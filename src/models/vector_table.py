"""
File is for building a vector table, from queue to text and ner
processing pipeline to upsert to vector table
"""
from src.base_models import Article
from src.api import WikipediaAPI
from src.tables import ArticleTable, VectorTable
from src.text_processor import BaseTextProcessor
from config import (rds_host, rds_dbname, rds_user, rds_password, rds_port, ner_articles, SECTIONS_TO_IGNORE)
from tqdm import tqdm
import json
import uuid
import re
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
wiki_api, processor, vector_tbl = WikipediaAPI(), BaseTextProcessor(), VectorTable()

while tqdm(set(expanded_articles), desc='Progress'):
    title = re.sub(' ', '_', expanded_articles.pop(1))
    title, page_id, final_text = wiki_api.fetch_article_data(title)
    article = Article(title, page_id, final_text)
    article.process_text_pipeline(processor, SECTIONS_TO_IGNORE)
    article.get_categories(processor)
    article.process_embedding_pipeline(processor)
    article.process_metadata_pipeline(processor)
    for keys, vector, encoding, metadata in zip(article.text_dict.keys(), article.embedding_dict.values(),
                                                article.metadata_dict.values()):
        vector_tbl.add_record(article.title, uuid.uuid4(), vector, encoding, article.categories, metadata)
