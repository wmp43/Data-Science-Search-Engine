"""
This file will upsert data in a new rds table for developing the embedding model
"""

from src.models import Article
from src.api import WikipediaAPI
from src.relational import EmbeddingModelTable
from src.text_processor import BaseTextProcessor
from config import (rds_host, rds_dbname, rds_user, rds_password, rds_port)
import tqdm

"""
Order of Operations and other details
"""
SECTIONS_TO_IGNORE = [
    "See also", "References", "External links", "Further reading", "Footnotes", "Bibliography", "Sources",
    "Citations",
    "Literature", "Footnotes", "Notes and references", "Photo gallery", "Works cited", "Photos", "Gallery", "Notes",
    "References and sources", "References and notes"]
list_of_relevant_articles = ['Machine_leanring', 'Normal_distribution'] #etc. must extend
wiki_api = WikipediaAPI()
processor = BaseTextProcessor()

for TITLE in tqdm(list_of_relevant_articles, desc='Fetching and processing articles'):
    title, page_id, final_text = wiki_api.fetch_article_data(TITLE)
    article = Article(title=title, id=page_id, text=final_text, text_processor=processor)
    article.process_text_pipeline(processor, SECTIONS_TO_IGNORE)



    # Call wikipedia API to get text
    # Run through text cleaning
    # Format so we have wiki_id, text, title
    # add to table


emb_tbl = EmbeddingModelTable(rds_dbname, rds_user, rds_password, rds_host, rds_port)
emb_tbl.add_record(455, 'sample_text', 'sample title')
emb_tbl.print_sample_data()

