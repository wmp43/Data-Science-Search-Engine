"""
File is for building a vector table, from queue to text and ner
processing pipeline to upsert to vector table
"""
import traceback
from src.base_models import Article
from src.api import WikipediaAPI
from src.rds_crud import ArticleTable, VectorTable
from src.text_processor import BaseTextProcessor
from config import (rds_host, rds_dbname, rds_user, rds_password, rds_port, ner_articles, SECTIONS_TO_IGNORE)
from tqdm import tqdm
from typing import List
import json
import uuid
import pandas as pd
import re
import mwparserfromhell
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def cluster_api_call(text):
    try:
        payload, api_url = {'text': text}, "http://127.0.0.1:5000/clustering_api"
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            cluster_vec = response.json()
            return cluster_vec
        else:
            print("Error in clustering API call:", response.status_code, response.text)
    except Exception as e: print(f'Error {e}'), traceback.print_exc()


def process_and_update_article_tbl(art_txt, art_id, article_table):
    """
    :param art_txt: article text
    :param article_table: table to update
    :return:
    """
    # Call clustering API
    try:
        cvector = cluster_api_call(art_txt)
        print(cvector)
        if isinstance(cvector, list) and isinstance(cvector[0], list):
            cvector = [item for sublist in cvector for item in sublist]
        article_table.update_text_cvector(art_txt, cvector, art_id)
    except Exception as e:
        print(f'Error {e}')
        traceback.print_exc()


# todo: Update article tbl with cleaned text and clustering
def threaded_article_pipeline(article_data, article_table: ArticleTable, vector_table: VectorTable, processor: BaseTextProcessor):
    try:
        # Process Article text -> txt pipe
        threaded_page_id, threaded_article_title, threaded_final_text = article_data
        threaded_article = Article(threaded_article_title, threaded_page_id, threaded_final_text, text_processor=processor)
        threaded_article.process_text_pipeline(SECTIONS_TO_IGNORE)

        # update article tbl with cleaned text & clustering vec
        cleaned_text = ' '.join([text for _, text in threaded_article.text_dict.items()])
        process_and_update_article_tbl(cleaned_text, threaded_page_id, article_table)


        # process embedding
        threaded_article.process_embedding_pipeline()
        # process ner
        threaded_article.process_metadata_pipeline()

        vector_records = []
        for idx, (threaded_keys, threaded_vector_tuple, threaded_metadata) in enumerate(zip(threaded_article.text_dict.keys(), threaded_article.embedding_dict.values(), threaded_article.metadata_dict.values())):
            embedding, encoding = threaded_vector_tuple[0][0], threaded_vector_tuple[1]
            vector_record = (str(uuid.uuid4()), threaded_page_id, threaded_article.title, embedding, encoding, json.dumps(threaded_metadata))
            vector_records.append(vector_record)

            if idx % 100 == 0:
                print(f'IDX: {idx} processed')

        vector_table.batch_upsert(vector_records)

    except Exception as e:
        print(f'Error processing {threaded_article_title}: {e}')
        traceback.print_exc()



def main():
    wiki_api, processor = WikipediaAPI(), BaseTextProcessor()
    rds_args = (rds_dbname, rds_user, rds_password, rds_host, rds_port)
    vector_tbl, article_tbl = VectorTable(*rds_args), ArticleTable(*rds_args)
    article_df = article_tbl.get_all_data_pd()
    articles_data = article_df[['id', 'title', 'raw_text']].values.tolist()

    THREADED = True
    if THREADED:
        futures = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            for article_data in tqdm(articles_data):
                futures.append(executor.submit(threaded_article_pipeline, article_data, article_tbl, vector_tbl, processor))
            for future in as_completed(futures):
                print(f"Task completed with result")

if __name__ == "__main__":
    main()

