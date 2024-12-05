from nltk.corpus import stopwords
import nltk
import json
from tqdm import tqdm

try:
    stopwords = stopwords.words('english')
except:
    nltk.download('stopwords')
    stopwords = stopwords.words('english')

def read_answers(answer_path):
	answer_list = json.load(open(answer_path, 'r', encoding='utf-8'))
	answer_dict = {}
	for answer in tqdm(answer_list, desc='Reading Answer Collection...', colour='green'):
		answer_dict[answer['Id']] = answer['Text']
	return answer_dict