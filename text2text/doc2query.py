import argparse as ap
import json
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from ranx import Qrels
import torch
import transformers
from bs4 import BeautifulSoup as bs
import re
from nltk.corpus import stopwords
import nltk
import multiprocessing as mp
import sys
import string
import os
from huggingface_hub import login
import my_util

os.environ['TRANSFORMERS_CACHE'] = '/mnt/netstore1_home/'
# os.environ['HF_HOME'] = '/mnt/netstore1_home/' # TODO: Preferred
login()

def main():
	argparser = ap.ArgumentParser()
	argparser.add_argument('answers', type=str, help='Answers.json file to generate queries from.')
	args = argparser.parse_args()

	answers_dict = read_answers(args.answers)

	msmarco_dict = []
	beir_dict = []
	llama_dict = []
	# Do all query-generation at the same time.
	try:
		msmarco_process = mp.Process(target=msmarco_doc2query, args=(answers_dict,msmarco_dict))
		beir_process = mp.Process(target=beir_doc2query, args=(answers_dict,beir_dict))
		llama_process = mp.Process(target=llama_doc2query, args=(answers_dict,llama_dict))

		msmarco_process.start()
		beir_process.start()
		llama_process.start()

		msmarco_process.join()
		beir_process.join()
		llama_process.join()

		merged_dict = {**msmarco_dict,**beir_dict,**llama_dict}
		with open('AllQueries.json', 'w', encoding='utf-8') as f:
			json.dump(merged_dict, f, indent=4)
	except KeyboardInterrupt:
		sys.exit()

def read_answers(answer_filepath):
	answer_list = json.load(open(answer_filepath, 'r', encoding='utf-8'))
	answer_dict = {}
	for answer in tqdm(answer_list, desc='Reading Answer Collection...', colour='yellow'):
		# TODO: why no use preprocess_text?
		# answer_dict[answer['Id']] = preprocess_text(answer['Text'])
		answer_dict[answer['Id']] = answer['Text']
	return answer_dict

def msmarco_doc2query(answers_dict, msmarco_dict):
	model_name = 'doc2query/msmarco-t5-base-v1'
	tokenizer = T5Tokenizer.from_pretrained(model_name)
	model = T5ForConditionalGeneration.from_pretrained(model_name)
	model.to('cuda:0')
	for answer_id, answer in tqdm(answers_dict.items(), desc='Generating MS-MARCO Query From Doc', colour='blue'):
		tokenized_answer = tokenizer.encode(answer, max_length=512, truncation=True, return_tensors='pt')
		tokenized_answer = tokenized_answer.to('cuda:0')
		tokenized_queries = model.generate(input_ids=tokenized_answer, max_length=128, do_sample=True, top_p=0.95, num_return_sequences=3)
		msmarco_dict.append({'Id': answer_id, 'Text': list(tokenizer.decode(tokenized_query, skip_special_tokens=True) for tokenized_query in tokenized_queries)})
	with open('MSMARCO_Queries.json', 'w', encoding='utf-8') as outfile:
		json.dump(msmarco_dict, outfile, indent=4)

def beir_doc2query(answers_dict, beir_dict):
	model_name = 'BeIR/query-gen-msmarco-t5-large-v1'
	tokenizer = T5Tokenizer.from_pretrained(model_name)
	model = T5ForConditionalGeneration.from_pretrained(model_name)
	model.to('cuda:0')
	for answer_id, answer in tqdm(answers_dict.items(), desc='Generating BeIR Query From Doc', colour='blue'):
		tokenized_answer = tokenizer.encode(answer, max_length=512, truncation=True, return_tensors='pt')
		tokenized_answer = tokenized_answer.to('cuda:0')
		tokenized_queries = model.generate(input_ids=tokenized_answer, max_length=128, do_sample=True, top_p=0.95, num_return_sequences=3)
		beir_dict.append({'Id': answer_id, 'Text': list(tokenizer.decode(tokenized_query, skip_special_tokens=True) for tokenized_query in tokenized_queries)})
	with open('BeIR_Queries.json', 'w', encoding='utf-8') as outfile:
		json.dump(beir_dict, outfile, indent = 4)

def llama_doc2query(answers_dict, llama_dict):
	model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
	# TODO: This has a warning about sequential use, slowing down efficiency.
	pipeline = transformers.pipeline(
		"text-generation",
		model=model_name,
		model_kwargs={"torch_dtype": torch.bfloat16},
		device_map="auto"
	)
	pipeline.model.generation_config.pad_token_id = pipeline.model.generation_config.eos_token_id
	messages = [
		{"role": "system", "content": "You are question generator assistant for travelling answers. When given an answer you will generate a corresponding question. Do not explicitly acknowledge the task or respond directly to the user, just do as told and generate a question."},
		# {'role': 'user', 'content': 'The EU\'s currency is known as the Euro'},
		# {'role': 'assistant', 'content': 'what is the official currency of the EU?'}
		# These examples used for few-show was a regenerated query and answer pair from ChatGPT
		{'role': 'user', 'content': "Practices regarding complimentary tap water in Europe vary widely, with no universal custom. While free water isn’t exclusive to Finland or Scandinavia, laws and traditions differ by country. some places, serving tap water is required by law, such as the UK (for premises serving alcohol), France (where pitchers are often provided automatically with meals), Hungary, and Spain. In Finland, Norway, Sweden, Denmark, and Slovenia, free water is very common. In countries like Switzerland, free tap water is offered inconsistently, while in the Netherlands, Germany, Luxembourg, Italy, and Belgium, it’s less common, and patrons typically order paid drinks. Some restaurants in these regions may refuse or appear surprised if asked for free water. Even in countries where laws mandate free tap water, exceptions occur, such as in mountain lodges or upscale venues. High-end restaurants may expect customers to purchase drinks, sometimes offering filtered or carbonated water as a paid alternative. Lastly, in places like Austria, France, and Italy, serving a glass of water alongside coffee is customary and generally well-accepted."},
		{'role': 'assistant', 'content': "How frequently do restaurants in Europe provide complimentary drinking water upon request? When I visited Helsinki, I noticed restaurants often provided free water with orders. This included places like McDonald’s, where my friend requested tap water, and it was served without charge. Some restaurants even encouraged this practice, offering water refill stations with clean glasses or placing glass jugs of water near the soft drink area for self-service. I haven’t observed this elsewhere in Europe, though my travels are limited. Is free water for customers a common practice across Europe, or is it specific to Finland or Scandinavia?"}
	]
	for answer_id, answer in tqdm(list(answers_dict.items()), desc='Generating LLaMa Query From Doc', colour='blue'):
		outputs = pipeline(messages + [{'role': 'user', 'content': answer}], max_new_tokens=256, num_return_sequences=3)
		llama_dict.append({'Id': answer_id, 'Text': [output['generated_text'][-1] for output in outputs]})
	with open('collections/LLaMa_Queries.json', 'w', encoding='utf-8') as outfile:
		json.dump(llama_dict, outfile, indent = 4)

if __name__ == '__main__':
	main()