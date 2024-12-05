from sentence_transformers import util, losses, SentenceTransformer, SentenceTransformerTrainingArguments, SimilarityFunction, SentenceTransformerTrainer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers
from collections import OrderedDict
from huggingface_hub import HfApi
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import evaluate
import pyarrow
import random
import torch
import os

prompt = {'retrieval': 'Retrieve semantically similar text: '}
prompt_name = 'retrieval'

class SBertBI(object):

	def __init__(self, model_name_or_path, device, similarity_fn: str = 'cos'):
		self.model_name_or_path = model_name_or_path
		self.device = device
		try:
			hf_api = HfApi()
			model_info = hf_api.model_info(model_name_or_path)
			self.model = SentenceTransformer(model_name_or_path, device=self.device, prompts=prompt)
		except Exception:
			print('Model name does not exist, attempting as checkpoint.')
			if os.path.exists(model_name_or_path):
				self.model = SentenceTransformer(model_name_or_path, device=self.device, prompts={'retrieval': 'Retrieve semantically similar text: '})
			else:
				raise Exception('Model name does not exist, and neither does the local path provided')
		try:
			self.model.prompt = prompt
			self.model.default_prompt_name = prompt_name # ??
		except:
			pass
		match similarity_fn:
			case 'cos':
				self.model.similarity_fn_name = SimilarityFunction.COSINE
			case 'dot':
				self.model.similarity_fn_name = SimilarityFunction.DOT_PRODUCT

	def encode_id_text(self, id_text):
		pool = self.model.start_multi_process_pool()
		embeddings = self.model.encode_multi_process(id_text.values(), pool)
		self.model.stop_multi_process_pool(pool)
		return dict(zip(id_text.keys(), embeddings))

	def encode(self, text):
		pool = self.model.start_multi_process_pool()
		embeddings = self.model.encode_multi_process(text, pool)
		self.model.stop_multi_process_pool(pool)
		return embeddings

	def get_best(self, query, corpa):
		base_embedding = self.encode(query)
		corpa_embeddings = self.encode(corpa)
		similarity_scores = self.model.similarity(base_embedding, corpa_embeddings)
		score, index = torch.topk(similarity_scores, k=1)
		return corpa[index]

	def retrieve_rank(self, topics, answers):
		topic_embeddings = self.encode_id_text(topics)
		answer_embeddings = self.encode_id_text(answers)
		topic_embeddings_tensor = torch.tensor(np.stack(list(topic_embeddings.values())), device=self.device)
		answer_embeddings_tensor = torch.tensor(np.stack(list(answer_embeddings.values())), device=self.device)
		similarity_scores = util.semantic_search(topic_embeddings_tensor, answer_embeddings_tensor,
									query_chunk_size=100, corpus_chunk_size=50000,
									top_k=100, score_function=SimilarityFunction.to_similarity_fn(self.model.similarity_fn_name))
		topic_ids = list(topic_embeddings.keys())
		answer_ids = list(answer_embeddings.keys())
		topic_answer_rankings = {}
		for topic_index, ranking in enumerate(tqdm(similarity_scores, desc='Patching Similarity Scores...')):
			ordered_rankings = OrderedDict({d['corpus_id']: d['score'] for d in ranking})
			topic_answer_rankings[topic_ids[topic_index]] = OrderedDict((answer_ids[answer_index], score) for answer_index, score in ordered_rankings.items())
		return topic_answer_rankings

	def fine_tune(self, save_path, train_qrel, eval_qrel, test_qrel, topics, answers, epochs:int = 1, batch_size:int = 32):
		train_dataset = self.format_dataset(train_qrel, answers, topics)
		eval_dataset = self.format_dataset(eval_qrel, answers, topics)
		loss_fn = losses.CoSENTLoss(model=self.model) # This said on the website that it was a more incoming function that was promising https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html#sentence_transformers.losses.CoSENTLoss
		self.model.to(self.device)
		args = SentenceTransformerTrainingArguments(
			output_dir=save_path,
			num_train_epochs=epochs,
			per_device_train_batch_size=batch_size,
			per_device_eval_batch_size=batch_size,
			learning_rate=5e-5,
			warmup_ratio=0.1,
			fp16=False,
			bf16=True,
			batch_sampler=BatchSamplers.NO_DUPLICATES,
			eval_strategy='steps',
			eval_steps=100,
			save_strategy='steps',
			save_steps=100,
			save_total_limit=1,
			logging_strategy='steps',
			logging_steps=25,
			run_name=self.model_name_or_path + '-travel',
			load_best_model_at_end=True,
		)
		eval_qrel_dict = eval_qrel.to_dict()
		eval_relevant_docs={}
		for topic_id, answer_scores in eval_qrel_dict.items():
			eval_relevant_docs[topic_id] = set(answer_scores.keys())
		train_ir_evaluator = InformationRetrievalEvaluator(
			queries=topics,
			corpus=answers,
			relevant_docs=eval_relevant_docs,
			name='TravelStackExchangeEval',
			mrr_at_k=[1, 5, 10],
			ndcg_at_k=[1, 5, 10],
			map_at_k=[1, 5, 10],
			show_progress_bar=True,
		)
		train_ir_evaluator(self.model)
		trainer = SentenceTransformerTrainer(
			model=self.model,
			args=args,
			train_dataset=train_dataset,
			eval_dataset=eval_dataset,
			loss=loss_fn,
			evaluator=train_ir_evaluator,
			compute_metrics=self.compute_metrics,
		)
		trainer.train()
		test_dataset = self.format_dataset(test_qrel, answers, topics)
		trainer.predict(test_dataset)
		test_relevant_docs={}
		test_qrel_dict = test_qrel.to_dict()
		for topic_id, answer_scores in test_qrel_dict.items():
			test_relevant_docs[topic_id] = set(answer_scores.keys())
		test_ir_evaluator = InformationRetrievalEvaluator(
			queries=topics,
			corpus=answers,
			relevant_docs=test_relevant_docs,
			name='PuzzleStackExchangeEval',
			mrr_at_k=[1, 5, 10],
			ndcg_at_k=[1, 5, 10],
			map_at_k=[1, 5, 10],
			show_progress_bar=True,
		)
		test_ir_evaluator(self.model)
		self.model.save(save_path, self.model_name_or_path + '/TravelStackExchange')

	def compute_metrics(self, eval_preds):
		metric = evaluate.load('trec_eval')
		logits, labels = eval_preds
		predictions = np.argmax(logits, axis=1)
		return metric.compute(predictions=predictions, references=labels)

	def msmarco_dataset(self, qrel, answers, topics):
		qrel_dict = qrel.to_dict()
		triplets = []
		for topic_id, answer_scores in qrel_dict.items():
			for answer_id, score in answer_scores.items():
				if score > 1: # Should we be more unforgiving?
					triplets.append({'sentence1': topics[topic_id], 'sentence2': answers[answer_id], 'label': 1.0})
				elif score == 1:
					triplets.append({'sentence1': topics[topic_id], 'sentence2': answers[answer_id], 'label': 0.5})
				else:
					triplets.append({'sentence1': topics[topic_id], 'sentence2': answers[answer_id], 'label': 0.0})
			for i in range(len(answer_scores)):
				random_answer_id = random.choice(list(answers.keys()))
				if random_answer_id in qrel_dict[topic_id].keys():
					continue
				triplets.append({'sentence1': topics[topic_id], 'sentence2': answers[random_answer_id], 'label': 0.0})
		dataset = Dataset(pyarrow.Table.from_pylist(mapping=triplets))
		return dataset
