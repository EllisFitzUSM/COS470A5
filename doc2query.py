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

try:
    stopwords = stopwords.words('english')
except:
    nltk.download('stopwords')
    stopwords = stopwords.words('english')

os.environ['TRANSFORMERS_CACHE'] = '/mnt/netstore1_home/'
login()

def main():
	argparser = ap.ArgumentParser()
	argparser.add_argument('answers', type=str, help='Answers.json file to generate queries from.')
	args = argparser.parse_args()

	answers_dict = read_answers(args.answers)

	# msmarco_doc2query(answers_dict)
	# beir_doc2query(answers_dict)
	try:
		# msmarco_process = mp.Process(target=msmarco_doc2query, args=(answers_dict,))
		# beir_process = mp.Process(target=beir_doc2query, args=(answers_dict,))
		llamam_process = mp.Process(target=llama_doc2query, args=(answers_dict,))

		# msmarco_process.start()
		# beir_process.start()
		llamam_process.start()

		# msmarco_process.join()
		# beir_process.join()
		llamam_process.join()
	except KeyboardInterrupt:
		sys.exit()

def read_answers(answer_filepath):
	answer_list = json.load(open(answer_filepath, 'r', encoding='utf-8'))
	answer_dict = {}
	for answer in tqdm(answer_list, desc='Reading Answer Collection...', colour='yellow'):
		# answer_dict[answer['Id']] = preprocess_text(answer['Text'])
		answer_dict[answer['Id']] = answer['Text']
	return answer_dict

def msmarco_doc2query(answers_dict):
	model_name = 'doc2query/msmarco-t5-base-v1'
	tokenizer = T5Tokenizer.from_pretrained(model_name)
	model = T5ForConditionalGeneration.from_pretrained(model_name)
	model.to('cuda:0')
	queries_dict = []
	for answer_id, answer in tqdm(answers_dict.items(), desc='Generating Query From Doc', colour='blue'):
		tokenized_answer = tokenizer.encode(answer, max_length=512, truncation=True, return_tensors='pt')
		tokenized_answer = tokenized_answer.to('cuda:0')
		tokenized_queries = model.generate(input_ids=tokenized_answer, max_length=128, do_sample=True, top_p=0.95, num_return_sequences=3)
		queries_dict.append({'Id': answer_id, 'Text': list(tokenizer.decode(tokenized_query, skip_special_tokens=True) for tokenized_query in tokenized_queries)})
	with open('MSMARCO_Queries.json', 'w', encoding='utf-8') as outfile:
		json.dump(queries_dict, outfile, indent=4)

def beir_doc2query(answers_dict):
	model_name = 'BeIR/query-gen-msmarco-t5-large-v1'
	tokenizer = T5Tokenizer.from_pretrained(model_name)
	model = T5ForConditionalGeneration.from_pretrained(model_name)
	model.to('cuda:0')
	queries_dict = []
	for answer_id, answer in tqdm(answers_dict.items(), desc='Generating Query From Doc', colour='blue'):
		tokenized_answer = tokenizer.encode(answer, max_length=512, truncation=True, return_tensors='pt')
		tokenized_answer = tokenized_answer.to('cuda:0')
		tokenized_queries = model.generate(input_ids=tokenized_answer, max_length=128, do_sample=True, top_p=0.95, num_return_sequences=3)
		queries_dict.append({'Id': answer_id, 'Text': list(tokenizer.decode(tokenized_query, skip_special_tokens=True) for tokenized_query in tokenized_queries)})
	with open('BeIR_Queries.json', 'w', encoding='utf-8') as outfile:
		json.dump(queries_dict, outfile, indent = 4)

def llama_doc2query(answers_dict):

	model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

	pipeline = transformers.pipeline(
		"text-generation",
		model=model_name,
		model_kwargs={"torch_dtype": torch.bfloat16},
		device_map="auto"
	)

	messages = [
		{"role": "system", "content": "You are query generator for travel passages. When given a passage you will generate a query that is answered with the passage."},
		{'role': 'user', 'content': 'The EU\'s currency is known as the Euro'},
		{'role': 'assistant', 'content': 'what is the official currency of the EU?'}
	]

	queries_dict = []
	for answer_id, answer in tqdm(list(answers_dict.items())[:10], desc='Generating Query From Doc', colour='blue'):
		outputs = pipeline(messages + [{'role': 'user', 'content': preprocess_text(answer)}], max_new_tokens=256)
		queries_dict.append({'Id': answer_id, 'Text': outputs[0]['generated_text'][-1]})
	with open('LLaMa_Queries.json', 'w', encoding='utf-8') as outfile:
		json.dump(queries_dict, outfile, indent = 4)


def preprocess_text(text_string):
	res_str = bs(text_string, "html.parser").get_text(separator=' ')
	res_str = re.sub(r'http(s)?://\S+', ' ', res_str)
	res_str = re.sub(r'[^\x00-\x7F]+', '', res_str)
	res_str = res_str.translate({ord(p): ' ' if p in r'\/.!?-_' else None for p in string.punctuation})
	res_str = ' '.join([word for word in res_str.split() if word not in stopwords])
	return res_str

if __name__ == '__main__':
	main()