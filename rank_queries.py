import argparse as ap
import json
from tqdm import tqdm
from util import read_answers
from sbert_ce_ir import SBertCE

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
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	cross_encoder_model = SBertCE('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
	new_query_dict = {}
	for answer_id, answer in answer_dict.items():
		generated_queries = [query['content'] for query in query_dict[answer_id]['Text']]
		best = cross_encoder_model.get_best(answer, generated_queries)
		new_query_dict[answer_id] = best
	with open('BestQueries.json', 'w') as outfile:
		json.dump(new_query_dict, outfile, indent=4)




if __name__ == '__main__':
	main()