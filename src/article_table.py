import traceback
from src.api import WikipediaAPI
from src.rds_crud import ArticleTable
from config import (rds_host, rds_dbname, rds_user, rds_password, rds_port, ner_articles, formatting_map)
from tqdm import tqdm
from typing import List
import uuid
import spacy
import re
import statistics
import requests
import mwparserfromhell


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
    articles = []
    for section in parsed.get_sections(levels=[2]):
        heading = section.filter_headings()[0].title.strip()
        if heading == "See also":
            links = section.filter_wikilinks()
            for link in links:
                title = link.title.strip_code().strip()
                if "#" in title:
                    articles.append(title.split("#")[0])
                elif "File" in title:
                    continue
                else:
                    articles.append(title)

            return articles
    return []


def expand_article_list(original_list):
    """ Expand the article list based on 'See also' sections. """
    expanded_list = set(original_list)  # Use a set to avoid duplicates
    for art in tqdm(original_list):
        wikitext = get_wikitext(art)
        if wikitext:
            see_also_titles = extract_see_also(wikitext)
            expanded_list.update(see_also_titles)
    return list(expanded_list)


# def data_validation(): pass
# todo: some method of showing pagerank
# todo: data validation for art_table
# todo: map connections between meta-articles and see also sections...


def fetch_and_store_article_data(articles: List, article_tbl: ArticleTable, wikipedia_caller: WikipediaAPI):
    for title in tqdm(articles, desc='Progress: '):
        try:
            formatted_title = re.sub(' ', '_', title)
            if formatted_title in formatting_map.keys(): formatted_title = formatting_map[formatted_title]
            article_title, page_id, article_text = wikipedia_caller.fetch_article_data_by_title(formatted_title)
            title_response = (article_title, page_id, article_text)
            if title_response and all(title_response):
                article_tbl.add_record(str(uuid.uuid4()), article_title, " ", article_text)
                print(f'Successfully added by title: {title}')
            else:
                missing_parts = [part for part, data in zip(["title", "page_id", "text"], title_response) if not data]
                print(f'Call Error at {formatted_title}, missing: {", ".join(missing_parts)}')
                # _article_title, _page_id, _article_text = wikipedia_caller.fetch_article_data_by_id(page_id)
                # id_response = (_article_title, _page_id, _article_text)
                # print(f'ID Res Tuple: {id_response}')
                # if id_response and all(id_response):
                #     article_tbl.add_record(str(uuid.uuid4()), _article_title, " ", _article_text)
                #     print(f'Successfully added by title: {_article_title} and id: {page_id}')
        except Exception as e:
            print(f'Error fetching/storing data for {title}: {e}')
            traceback.print_exc()


INGEST = True
if INGEST:
    wiki_api = WikipediaAPI()
    expanded_art_list = expand_article_list(ner_articles)
    article_table = ArticleTable(rds_dbname, rds_user, rds_password, rds_host, rds_port)
    # Expand articles, then fetch and store
    fetch_and_store_article_data(expanded_art_list, article_table, wiki_api)
