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
import re
import mwparserfromhell
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def data_validation():
    # todo: Data Validation Function that certifies upserted records follow data format
    pass


# todo: build functionality for only vector table build and not articles
def threaded_article_pipeline(title: str, vector_table: VectorTable, article_table: ArticleTable, processor: BaseTextProcessor):
    try:
        # Processing the article
        threaded_article_title = re.sub(' ', '_', title)
        threaded_article_title, threaded_page_id, threaded_final_text = wiki_api.fetch_article_data(threaded_article_title)
        threaded_article = Article(threaded_article_title, threaded_page_id, threaded_final_text, processor)
        threaded_article.process_text_pipeline(processor, SECTIONS_TO_IGNORE)
        threaded_article.process_embedding_pipeline(processor)
        threaded_article.process_metadata_pipeline(processor)

        # Preparing batch records
        article_record = (str(uuid.uuid4()), threaded_article.title, threaded_article.text)
        vector_records = []

        for idx, (threaded_keys, threaded_vector_tuple, threaded_metadata) in enumerate(zip(threaded_article.text_dict.keys(), threaded_article.embedding_dict.values(), threaded_article.metadata_dict.values())):
            embedding, encoding = threaded_vector_tuple[0][0], threaded_vector_tuple[1]
            vector_record = (str(uuid.uuid4()), article_record[0], threaded_article.title, embedding, encoding, json.dumps(threaded_metadata))
            vector_records.append(vector_record)

            if idx % 100 == 0:
                print(f'IDX: {idx} processed')

        article_table.batch_upsert([article_record])
        vector_table.batch_upsert(vector_records)

    except Exception as e:
        print(f'Error processing {title}: {e}')
        traceback.print_exc()

# expanded_articles = expand_article_list(ner_articles)
wiki_api, processor = WikipediaAPI(), BaseTextProcessor()
rds_args = (rds_dbname, rds_user, rds_password, rds_host, rds_port)
vector_tbl, article_tbl = VectorTable(*rds_args), ArticleTable(*rds_args)
THREADED = True

if THREADED:
    futures = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for article in ner_articles[:5]:
            futures.append(executor.submit(threaded_article_pipeline, article, vector_tbl, article_tbl, processor))
        for future in as_completed(futures):
            print(f"Task completed with result: {future.result()}")
else:
    for article_title in tqdm(set(ner_articles), desc='Progress'):
        article_title = re.sub(' ', '_', article_title)
        title, page_id, final_text = wiki_api.fetch_article_data(article_title)
        article = Article(title, page_id, final_text)
        article.process_text_pipeline(processor, SECTIONS_TO_IGNORE)
        article.get_categories(processor)
        article.process_embedding_pipeline(processor)
        article.process_metadata_pipeline(processor)
        for keys, vector, encoding, metadata in zip(article.text_dict.keys(), article.embedding_dict.values(),
                                                    article.metadata_dict.values()):
            vector_tbl.add_record(article.title, str(uuid.uuid4()), vector, encoding, article.categories, metadata)
