from sbert_ce_ir import SBertCE
from bm25_ir import BM25
from ranx import Qrels
from ranx import Run
import argparse
import my_util
import os

results_dir='.'

def main():
    global results_dir
    parser = argparse.ArgumentParser()
    parser.add_argument('answers', help='Answers (representing document collection).')
    parser.add_argument('topics', help='Topics (representing OG queries).', nargs='+')
    parser.add_argument('-test', help='Test Topics Qrel')
    parser.add_argument('-sym', '--symmetric', help='If the cross-encoder semantic sim is symmetric. This would either A) answers are in query format or B) queries are in answer format.', action='store_true', default=False)
    parser.add_argument('-name', help='Custom name to be appended to run names.', default='')
    parser.add_argument('-pti', '--pt_index', help='Path to PyTerrier index.', default=r'./pt_index')
    parser.add_argument('-ft', '--ft_cross', help='Path to FT index.', default=None)
    parser.add_argument('-res', '--results_dir', help='Path to directory to save results.', default=r'.\results')
    # parser.add_argument() # Here I would like a flag to turn on to ReRank run with cross encoder? sumn like that
    args = parser.parse_args()
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    bm25 = get_bm25(args.answers, args.pt_index)
    cross = get_cross_encoder(args.ft_cross, args.symmetric)
    answer_dict = my_util.read_answers(args.answers)
    retrieve(bm25, cross, answer_dict, args.topics, args.test, args.ft_cross)

def retrieve(bm25, cross, answers_dict, topics_paths, test_qrel, ft_cross):
    skip = 1
    if test_qrel is not None:
        qrel = Qrels.from_file(test_qrel, kind='trec').to_dict()
        topic_1_dict = my_util.read_topics(topics_paths[0])
        test_topics = {topic_id: topic_1_dict[topic_id] for topic_id in qrel}
        bm25_rankings = bm25.rank(test_topics)
        cross_rankings = cross.rerank(bm25_rankings, test_topics, answers_dict, 128)
        save_run(bm25_rankings, f'res_BM25_test')
        save_run(cross_rankings, f'res_BM25_Cross{"_ft" if ft_cross is not None else ""}_ReRank_test')
        if test_qrel in topics_paths:
            topics_paths = topics_paths[1:]
        skip = 2
    for index, topic_path in enumerate(topics_paths):
        topics_dict = my_util.read_topics(topic_path)
        bm25_rankings = bm25.rank(topics_dict)
        cross_rankings = cross.rerank(bm25_rankings, topics_dict, answers_dict, 128)
        save_run(bm25_rankings, f'res_BM25_{index + skip}')
        save_run(cross_rankings, f'res_BM25_Cross{"_ft" if ft_cross is not None else ""}_ReRank_{index + skip}')

def get_bm25(answers_path, pt_index):
    return BM25(answers_path, pt_index)

def get_cross_encoder(ft_cross=None, symmetric=False):
    if ft_cross is not None:
        cross = SBertCE(ft_cross, device='cuda')
    elif symmetric:
        cross = SBertCE('cross-encoder/quora-roberta-large', device='cuda')
    else:
        cross = SBertCE('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
    return cross

def save_run(run_dict, name):
    saved_run = Run.from_dict(run_dict)
    local_file = f'{name}.tsv'
    saved_run.save(path=os.path.join(os.path.abspath(results_dir), local_file), kind='trec')

if __name__ == '__main__':
    main()