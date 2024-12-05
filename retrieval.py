import argparse
import os
import pyterrier as pt
import util
import json
import pandas as pd
from tqdm import tqdm
import string

index_path = r'./pt_index'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gen_queries', help='Generated Queries (representing document collection).')
    parser.add_argument('topics', help='Topics (representing OG queries).')
    args = parser.parse_args()

    index: pt.IndexFactory.of = pt.IndexFactory.of(get_bm25_index(args.gen_queries))
    print(index.getCollectionStastics().toString())

def topic_file_to_dataframe(topic_path, includes = ['Title', 'Body', 'Tags']):
    topics = json.load(open(topic_path, 'r', encoding='utf-8'))                                          # Parse JSON into Dict
    topics_list = []                                                                                      # Initialize Topics List
    for topic in tqdm(topics, colour='green', desc='Converting Topics File into DataFrame'):
        topic_id = topic['Id']
        topic_text = ' '.join([util.preprocess_text(topic[include]) for include in includes])
        topics_list.append([topic_id, topic_text])                                                                                        # Union Title, Body, and Tags in Topic

    return pd.DataFrame(topics_list, columns=['qid', 'query'])

def get_bm25_index(gen_queries_path):
    index_abs_path = os.path.abspath(index_path)
    if not os.path.exists(index_abs_path):
        os.makedirs(index_abs_path, exist_ok=True)
        pt_indexer = pt.IterDictIndexer(index_abs_path,
                                        verbose=True,
                                        overwrite=True,
                                        stopwords=util.get_stopwords(),
                                        tokeniser='english')
        docs_df: pd.DataFrame = pd.DataFrame(util.read_answers(gen_queries_path))
        docs_df.rename({'Id': 'docno', 'Text': 'text'}, axis='columns', inplace=True)
        pt_indexer.index(docs_df[['text', 'docno']].to_dict(orient='records'))
    return os.path.join(index_abs_path, 'data.properties')

def bm25_retrieval(index_ref, topics_df, topic_number) -> None:
    index: pt.IndexFactory.of = pt.IndexFactory.of(index_ref)
    bm25: pt.terrier.Retriever = pt.terrier.Retriever(index, num_results=100, wmodel='BM25')
    # metaindex = index.getMetaIndex()
    bm25_result: pd.DataFrame = bm25.transform(topics_df)
    pt.io.write_results(bm25_result, f'res_BM25_{topic_number}.tsv', format='trec')


if __name__ == '__main__':
    main()