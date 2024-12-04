import argparse as ap
import json
from tqdm import tqdm
from util import read_answers

def main():
	parser = ap.ArgumentParser()
	parser.add_argument('query_path', help='Path to the generated query file.')
	parser.add_argument('answer_path', help='Path to the original answer file.')
	args = parser.parse_args()

	answer_dict = read_answers(args.answer_path)
	query_dict = read_queries(args.query_path)
	best_query_dict = rank_queries(answer_dict, query_dict)
	with open('Queries.json', 'w') as outfile:
		json.dump(best_query_dict, outfile, indent=4)

def read_queries(query_path):
	query_list = json.load(open(query_path, 'r', encoding='utf-8'))
	query_dict = {}
	for query in tqdm(query_list, desc='Reading Query Collection...', colour='red'):
		query_dict[query['Id']] = query['Text']
	return query_dict

def rank_queries(answer_dict, query_dict):
	pass




if __name__ == '__main__':
	main()