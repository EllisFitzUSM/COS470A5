import argparse
import os
# import pyterrier as pt
import pyterrier as pt
import util
import json
import pandas as pd
from tqdm import tqdm
import string
from sbert_bi_ir import *
from sbert_ce_ir import *
from ranx import Run

index_path = r'./pt_index'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gen_queries', help='Generated Queries (representing document collection).')
    parser.add_argument('topics', help='Topics (representing OG queries).', nargs='+')
    args = parser.parse_args()

    index_ref = get_bm25_index_ref(args.gen_queries)
    index: pt.IndexFactory.of = pt.IndexFactory.of(index_ref)
    bi = SBertBI('sentence-transformers/all-mpnet-base-v2', device='cuda')
    ce = SBertCE('cross-encoder/quora-roberta-large', device='cuda')
    gen_query_dict = util.read_answers(args.gen_queries)
    print(index.getCollectionStatistics().toString())

    for index, topic_path in enumerate(args.topics):
        topic_df = topic_file_to_dataframe(topic_path)
        bm25_retrieval(index_ref, topic_df, index + 1)

        topics_dict = util.read_topics(topic_path) # TODO: ! THIS NEEDS TO PREPROCESS TEXT
        bi_results = bi.retrieve_rank(topics_dict, gen_query_dict)
        bi_run = Run.from_dict(bi_results, kind='trec')
        bi_run.save(f'res_BiEncoder_{index + 1}.tsv', kind='trec')
        ce_results = ce.retrieve_rerank(bi_run, topics_dict, gen_query_dict)
        ce_run = Run.from_dict(ce_results, kind='trec')
        ce_run.save(f'res_CrossEncoder_{index + 1}.tsv', kind='trec')

def get_bm25_index_ref(gen_queries_path):
    index_abs_path = os.path.abspath(index_path)
    if not os.path.exists(index_abs_path):
        os.makedirs(index_abs_path, exist_ok=True)
        pt_indexer = pt.IterDictIndexer(index_abs_path,
                                        verbose=True,
                                        overwrite=True,
                                        stopwords=util.get_stopwords(),
                                        tokeniser='english')
        docs_df: pd.DataFrame = pd.DataFrame(json.load(open(gen_queries_path, 'r', encoding='utf-8')))
        docs_df.rename({'Id': 'docno', 'Text': 'text'}, axis='columns', inplace=True)
        pt_indexer.index(docs_df[['text', 'docno']].to_dict(orient='records'))
    return os.path.join(index_abs_path, 'data.properties')

def topic_file_to_dataframe(topic_path, includes = ['Title', 'Body', 'Tags']):
    topics = json.load(open(topic_path, 'r', encoding='utf-8'))                                          # Parse JSON into Dict
    topics_list = []                                                                                      # Initialize Topics List
    for topic in tqdm(topics, colour='green', desc='Converting Topics File into DataFrame'):
        topic_id = topic['Id']
        topic_text = ' '.join([util.preprocess_text(topic[include]) for include in includes])
        topics_list.append([topic_id, topic_text])                                                                                        # Union Title, Body, and Tags in Topic

    return pd.DataFrame(topics_list, columns=['qid', 'query'])

def bm25_retrieval(index, topics_df, topic_number):
    bm25: pt.terrier.Retriever = pt.terrier.Retriever(index, num_results=100, wmodel='BM25')
    # metaindex = index.getMetaIndex()
    bm25_result: pd.DataFrame = bm25.transform(topics_df)
    pt.io.write_results(bm25_result, f'res_BM25_{topic_number}.tsv', format='trec')


if __name__ == '__main__':
    main()