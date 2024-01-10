"""
File is for building a vector table, from queue to text and ner
processing pipeline to upsert to vector table
"""
from src.base_models import Article
from src.api import WikipediaAPI
from typing import List
from src.tables import ArticlesTable
from src.text_processor import BaseTextProcessor
import spacy
from spacy.tokens import DocBin
from config import (rds_host, rds_dbname, rds_user, rds_password, rds_port, ner_articles, SECTIONS_TO_IGNORE)
from tqdm import tqdm
import json
import uuid
import re
import random
import mwparserfromhell
import requests

import mwparserfromhell
import requests

def get_wikitext(page_title):
    """ Fetch the raw wikitext of a Wikipedia page. """
    try:
        response = requests.get(
            'https://en.wikipedia.org/w/api.php',
            params={
                'action': 'query',
                'format': 'json',
                'titles': page_title,
                'prop': 'revisions',
                'rvprop': 'content'
            }
        ).json()
        page = next(iter(response['query']['pages'].values()))
        if 'revisions' in page:
            return page['revisions'][0]['*']
    except Exception as e:
        print(f"Error fetching wikitext for {page_title}: {e}")
    return ""


def extract_see_also(wikitext):
    """ Extract 'See also' section using mwparserfromhell. """
    parsed = mwparserfromhell.parse(wikitext)
    for section in parsed.get_sections(levels=[2]):
        heading = section.filter_headings()[0].title.strip()
        if heading == "See also":
            return [link.title.strip_code().strip() for link in section.filter_wikilinks()]
    return []


def get_categories(page_title):
    """ Fetch the categories of a Wikipedia page. """
    try:
        response = requests.get(
            'https://en.wikipedia.org/w/api.php',
            params={
                'action': 'query',
                'format': 'json',
                'titles': page_title,
                'prop': 'categories'
            }
        ).json()
        page_id = next(iter(response['query']['pages']))
        categories = response['query']['pages'][page_id]['categories']
        return [cat['title'] for cat in categories]
    except Exception as e:
        print(f"Error fetching categories for {page_title}: {e}")
        return []


def expand_article_list(original_list):
    """ Expand the article list based on 'See also' sections. """
    expanded_list = set(original_list)  # Use a set to avoid duplicates
    for art in original_list:
        wikitext = get_wikitext(art)
        if wikitext:
            see_also_titles = extract_see_also(wikitext)
            expanded_list.update(see_also_titles)
    return list(expanded_list)



expanded_articles = expand_article_list(ner_articles)
wiki_api,  processor = WikipediaAPI(), BaseTextProcessor()


while tqdm(set(expanded_articles),desc='Progress'):
    title = re.sub(' ', '_', expanded_articles.pop(1))
    title, page_id, final_text = wiki_api.fetch_article_data(title)
    article = Article(title, page_id, final_text)
    article.process_text_pipeline(processor, SECTIONS_TO_IGNORE)
    article.process_embedding_pipeline(processor)
    article.process_metadata_pipeline(processor)



# def build_corpus(current_list: List) -> List:
#     """
#     :param current_list: The current built list of articles
#     :return: a larger list of articles
#     """
#     wiki_api, result_list = WikipediaAPI(), []
#     for article_title in current_list:
#         title, page_id, final_text = wiki_api.fetch_article_data(article_title)
#         print(final_text)
#         # find see also of each article
#         # Append to result_list
#         return result_list
#
#
# corp_list = build_corpus(ner_articles)




