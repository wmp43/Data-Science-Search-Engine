"""
This file will upsert data in a new rds table for developing the embedding model
"""
from src.base_models import Article
from src.api import WikipediaAPI
from src.relational import ArticlesTable
from src.text_processor import BaseTextProcessor
import spacy
from spacy.tokens import DocBin
from config import (rds_host, rds_dbname, rds_user, rds_password, rds_port, data_science_articles)
from tqdm import tqdm
import json
"""
This is a file to build a article table for ner or embeddings


Ingestion pipeline for full article and text data
1. Define List of Articles to Snag
2. Call API with title
3. Instantiate Article with returned items
4. Clean Text
5. Label Text
5. Store in DB
6. Use it for fine-tuning
"""

SECTIONS_TO_IGNORE = ["References", "External links", "Further reading", "Footnotes", "Bibliography", "Sources", "See also"
    "Citations", "Literature", "Footnotes", "Notes and references", "Photo gallery", "Works cited", "Photos", "Gallery",
    "Notes", "References and sources", "References and notes"]


"""
Ingest should only really be run if new article titles or labels have been devlop in the config file.
Takes around 45 seconds per 100 articles.
For Actual ingestion the see also section should be directly used to find related articles of our meta article list
Must limit to 2 edges from original in order to limit infinite search


56098
"""
wiki_api = WikipediaAPI()
processor = BaseTextProcessor()
INGEST = False
if INGEST:
    unique_id = -2
    emb_tbl = ArticlesTable(rds_dbname, rds_user, rds_password, rds_host, rds_port)
    for TITLE in tqdm(set(data_science_articles), desc='Progress'):
        title, page_id, final_text = wiki_api.fetch_article_data(TITLE)
        if page_id == -1:
            page_id = unique_id
            unique_id -= 1
        article = Article(title=title, id=page_id, text=final_text, text_processor=processor)
        article.process_text_pipeline(processor, SECTIONS_TO_IGNORE)
        json_record = article.process_metadata_labeling(processor)
        # total_text = ""
        # for (key, text), metadata in zip(article.text_dict.items(), article.metadata_dict.values()):
        #     print(metadata, type(metadata))
        #     total_text += text
        # cleaned_text = re.sub(r'[\n\t]', ' ', total_text)
        emb_tbl.add_record(json_record['id'], json_record['text'], json_record['title'], json.dumps(json_record['labels']))
    emb_tbl.close_connection()


"""
This builds a JsonL object that can be used in doccano labelling. 
Check out docker build to avoid issues
Username & password inside cover page of statistics for DS book
"""

BUILD_JSONL = False
if BUILD_JSONL:
    emb_df = ArticlesTable(rds_dbname, rds_user, rds_password, rds_host, rds_port)
    json_obj = emb_df.get_all_data_json()
    emb_df.close_connection()
    with open('doccano_data.jsonl', 'w', encoding='utf-8') as file:
        for item in json_obj:
            item['text'] = item['text'].replace('\\', '')
            json_string = json.dumps(item, ensure_ascii=False)
            file.write(json_string + '\n')
    print("File 'doccano_data.jsonl' has been created.")


"""
This builds a spacy format dataset for fine-tuning spaCy model.
Data format:
"""
BUILD_SPACY = True
if BUILD_SPACY:
    emb_df = ArticlesTable(rds_dbname, rds_user, rds_password, rds_host, rds_port)
    art_df = emb_df.get_all_data_pd()
    training_data = []
    for _, row in art_df.iterrows():
        entities = [tuple(entity) for entity in row['label']]
        training_data.append((row['text'], entities))
    """
    So now training_data is spacy formatted. Can now build the .spacy file for training
    
    the building of .spacy train data has some considerations:
    1. nlp var is a blank english model. We need something for ner: https://spacy.io/models
    2. Currently our training data is our entire dataset. So we will have to find new unique articles.
        - think I'm just gonna build a new list in config only iterate through non-training titles
    """
    nlp = spacy.load('en_core_web_sm')
    db = DocBin()
    for text, annotations in training_data:
        doc = nlp(text)
        ents = []
        for start, end, label in annotations:
            span = doc.char_span(start, end, label=label)
            ents.append(span)
            """
            Traceback (most recent call last):
            File "/Users/owner/myles-personal-env/Projects/wikiSearch/src/models/article_tbl.py", line 111, in <module>
            doc.ents = ents
            File "spacy/tokens/doc.pyx", line 790, in spacy.tokens.doc.Doc.ents.__set__
            File "spacy/tokens/doc.pyx", line 2005, in spacy.tokens.doc.get_entity_info
            TypeError: object of type 'NoneType' has no len()
            
            I guess my span metadata_pipeline doesn't work ideally. 
            I suppose we can rebuild this just from text articles table
            Also should include standard tags that are built from classical ner from spaCy
            """
        doc.ents = ents
        db.add(doc)
    db.to_disk("./train.spacy")