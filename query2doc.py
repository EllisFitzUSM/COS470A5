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
import util
import itertools

os.environ['TRANSFORMERS_CACHE'] = '/mnt/netstore1_home/'
# os.environ['HF_HOME'] = '/mnt/netstore1_home/' # TODO: Preferred
login()

def main():
	argparser = ap.ArgumentParser()
	argparser.add_argument('topics', type=str, help='Answers.json file to generate queries from.')
	args = argparser.parse_args()

	topics_dict = util.read_topics(args.topics)
	topics_dict = dict(itertools.islice(topics_dict.items(), 10))

	llama_list = []
	try:
		llama_process = mp.Process(target=llama_query2doc, args=(topics_dict,llama_list))
		llama_process.start()
		llama_process.join()
	except KeyboardInterrupt:
		sys.exit()

def llama_query2doc(topics_dict, llama_dict):
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
		{"role": "system", "content": "You are question answerer assistant for travelling questions. When given an question you will generate a corresponding anwswer. Do not explicitly acknowledge the task or respond directly to the user, just do as told and generate an answer."},
		# These examples used for few-shot was a regenerated query and answer pair from ChatGPT
        {'role': 'user', 'content': "How frequently do restaurants in Europe provide complimentary drinking water upon request? When I visited Helsinki, I noticed restaurants often provided free water with orders. This included places like McDonald’s, where my friend requested tap water, and it was served without charge. Some restaurants even encouraged this practice, offering water refill stations with clean glasses or placing glass jugs of water near the soft drink area for self-service. I haven’t observed this elsewhere in Europe, though my travels are limited. Is free water for customers a common practice across Europe, or is it specific to Finland or Scandinavia?"},
        {'role': 'assistant', 'content': "Practices regarding complimentary tap water in Europe vary widely, with no universal custom. While free water isn’t exclusive to Finland or Scandinavia, laws and traditions differ by country. some places, serving tap water is required by law, such as the UK (for premises serving alcohol), France (where pitchers are often provided automatically with meals), Hungary, and Spain. In Finland, Norway, Sweden, Denmark, and Slovenia, free water is very common. In countries like Switzerland, free tap water is offered inconsistently, while in the Netherlands, Germany, Luxembourg, Italy, and Belgium, it’s less common, and patrons typically order paid drinks. Some restaurants in these regions may refuse or appear surprised if asked for free water. Even in countries where laws mandate free tap water, exceptions occur, such as in mountain lodges or upscale venues. High-end restaurants may expect customers to purchase drinks, sometimes offering filtered or carbonated water as a paid alternative. Lastly, in places like Austria, France, and Italy, serving a glass of water alongside coffee is customary and generally well-accepted."}
	]
	for topic_id, topic in tqdm(list(topics_dict.items()), desc='Generating LLaMa Answer From Query', colour='blue'):
		prompt = pipeline.tokenizer.apply_chat_template(messages + [{'role': 'user', 'content': topic}], add_generation_prompt=True, tokenize=True)
		outputs = pipeline(prompt, max_new_tokens=512, num_return_sequences=1)
		llama_dict.append({'Id': topic_id, 'Text': outputs[0]['generated_text'][-1]})
	with open('GeneratedAnswers.json', 'w', encoding='utf-8') as outfile:
		json.dump(llama_dict, outfile, indent = 4)

if __name__ == '__main__':
	main()