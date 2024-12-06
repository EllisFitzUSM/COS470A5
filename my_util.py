from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords
from tqdm import tqdm
import string
import nltk
import json
import re

try:
	stopwords = stopwords.words('english')
except:
	nltk.download('stopwords')
	stopwords = stopwords.words('english')

def get_stopwords():
	return stopwords

def read_answers(answer_path):
	answer_list = json.load(open(answer_path, 'r', encoding='utf-8'))
	answer_dict = {}
	for answer in tqdm(answer_list, desc='Reading Answer Collection...', colour='green'):
		answer_dict[answer['Id']] = preprocess_text(answer['Text'])
	return answer_dict

def read_topics(topic_path, includes = ['Title', 'Body', 'Tags']):
	topic_list = json.load(open(topic_path, 'r', encoding='utf-8'))
	topic_dict = {}
	for topic in tqdm(topic_list, desc='Reading Topic Collection...', colour='blue'):
		topic_dict[topic['Id']] = ' '.join([preprocess_text(topic[include]) for include in includes])
	return topic_dict

def preprocess_text(text_string):
	res_str = bs(text_string, "html.parser").get_text(separator=' ')
	res_str = re.sub(r'http(s)?://\S+', ' ', res_str)
	res_str = re.sub(r'[^\x00-\x7F]+', '', res_str)
	res_str = res_str.translate({ord(p): ' ' if p in r'\/.!?-_' else None for p in string.punctuation})
	# res_str = ' '.join([word for word in res_str.split() if word not in stopwords]) # Testing this...
	return res_str