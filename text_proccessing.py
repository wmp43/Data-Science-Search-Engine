"""
File used to test the text processing pipeline

"""
from src.text_processor import BaseTextProcessor, TextProcessor
from src.models import Article
from src.ingestion.api import WikipediaAPI

TITLE = 'Normal_distribution'
SECTIONS_TO_IGNORE = [
    "See also", "References", "External links", "Further reading", "Footnotes", "Bibliography", "Sources", "Citations",
    "Literature", "Footnotes", "Notes and references", "Photo gallery", "Works cited", "Photos", "Gallery", "Notes",
    "References and sources", "References and notes"]

wiki_api = WikipediaAPI()
processor = BaseTextProcessor()

title, page_id, final_text = wiki_api.fetch_article_data(TITLE)

article = Article(category='filler category', title=TITLE,
                  id=page_id, text=final_text, text_dict={},
                  metadata={}, text_processor=processor)

article = article.process_text_pipeline(processor, SECTIONS_TO_IGNORE)
article.show_headings(processor)

#print(article.text)
# print(article.text_dict.keys(),'\n\n\n')
#print(article.text_dict['Introduction'])
