from sbert_ce_ir import SBertCE
from itertools import islice
from ranx import Qrels
import argparse
import my_util
import random
import torch
import os

qrel_split_dir='.'

def main():
	global qrel_split_dir
	parser = argparse.ArgumentParser()
	parser.add_argument('answers', help='Answers (representing document collection).')
	parser.add_argument('topics', help='Topics (representing OG queries).', nargs='+')
	parser.add_argument('-save', help='Path to save fine-tuned cross.', default=r'.\ft_cross')
	parser.add_argument('-model', help='Name of cross-encoder model to download from huggingface.', default='cross-encoder/ms-marco-MiniLM-L-6-v2')
	parser.add_argument('-q', '--qrel', help='Path to qrel_1.tsv')
	parser.add_argument('-qs', '--qrel_splits', help='Qrel splits (from qrel_1.tsv).', nargs=3)
	parser.add_argument('-qsp', '--qrel_split_dir', help='Directory to save Qrel splits (from qrel_1.tsv', default=r'.\qrel_splits')
	parser.add_argument('-e', '--epochs', type=int, help='Epochs to train', default=4)
	args = parser.parse_args()
	qrel_split_dir = os.path.abspath(args.qrel_split_dir)

	answer_dict = my_util.read_answers(args.answers)
	topic_dict_1 = my_util.read_topics(args.topics[0])

	train_qrel, eval_qrel, test_qrel = split_qrel_1(args.qrel, args.qrel_splits)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	cross = SBertCE(args.model, device)

	cross.fine_tune(args.save,
					train_qrel, eval_qrel, test_qrel,
					topic_dict_1, answer_dict,
					epochs=args.epochs, batch_size=128)

def split_qrel_1(qrel_filepath, qrel_splits, split: float = 0.9):
	if qrel_splits:
		train_qrel, eval_qrel, test_qrel = map(lambda q: Qrels.from_file(q, kind='trec'), qrel_splits)
		return train_qrel, eval_qrel, test_qrel
	elif qrel_filepath:
		qrel_dict = Qrels.from_file(qrel_filepath, kind='trec').to_dict()
		topic_ids = list(qrel_dict.keys())
		random.shuffle(topic_ids)
		qrel_dict = {query_id:qrel_dict[query_id] for query_id in topic_ids}

		query_count = len(qrel_dict)
		train_set_count = int(query_count * split)
		val_set_count = int((query_count - train_set_count) / 2)

		train_qrel = Qrels.from_dict(dict(islice(qrel_dict.items(), train_set_count)))
		eval_qrel = Qrels.from_dict(dict(islice(qrel_dict.items(), train_set_count, train_set_count + val_set_count)))
		test_qrel = Qrels.from_dict(dict(islice(qrel_dict.items(), train_set_count + val_set_count, None)))

		os.makedirs(r'.\qrel_splits', exist_ok=True)
		train_qrel.save(os.path.join(qrel_split_dir, 'train_qrel.tsv'), kind='trec')
		eval_qrel.save(os.path.join(qrel_split_dir, 'eval_qrel.tsv'), kind='trec')
		test_qrel.save(os.path.join(qrel_split_dir, 'test_qrel.tsv'), kind='trec')

		return train_qrel, eval_qrel, test_qrel
	else:
		raise Exception('Must supply either a single Qrel to be split, or split Qrels (Train, Validation, Test)')

def split_topics_1(topic_dict, train_qrel, eval_qrel, test_qrel):
	train_topics = {topic_id: topic_dict[topic_id] for topic_id in train_qrel.keys()}
	eval_topics = {topic_id: topic_dict[topic_id] for topic_id in eval_qrel.keys()}
	test_topics = {topic_id: topic_dict[topic_id] for topic_id in test_qrel.keys()}
	return train_topics, eval_topics, test_topics


if __name__ == '__main__':
	main()