"""
This file will upsert data in a new rds table for developing the embedding model
"""
from src.base_models import Article
from src.api import WikipediaAPI
from src.tables import ArticleTable
from src.text_processor import BaseTextProcessor
import spacy
from spacy.tokens import DocBin
from config import (rds_host, rds_dbname, rds_user, rds_password, rds_port, ner_articles, ner_pattern, non_fuzzy_list)
from tqdm import tqdm
from typing import List, Dict
import json
import uuid
import random
import re
from collections import defaultdict

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='spacy')


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

SECTIONS_TO_IGNORE = ["References", "External links", "Further reading", "Footnotes", "Bibliography", "Sources",
                      "See also",
                      "Citations", "Literature", "Footnotes", "Notes and references", "Photo gallery", "Works cited",
                      "Photos", "Gallery",
                      "Notes", "References and sources", "References and notes"]

"""
Ingest should only really be run if new article titles or labels have been devlop in the config file.
Takes around 45 seconds per 100 articles.
For Actual ingestion the see also section should be directly used to find related articles of our meta article list
Must limit to 2 edges from original in order to limit infinite search
"""

BUILD_ARTICLE_LIST = False
if BUILD_ARTICLE_LIST:
    # To build finetuning data I think just getting all articles under see also for each
    # May not be necessary right now
    print('Implement Build Article List')


def convert_fuzzy_match(patterns: List[Dict], non_fuzzy_list: List[str]) -> List[Dict]:
    spacy_patterns = []
    for entry in patterns:
        label = entry['label']
        pattern = entry['pattern']

        if isinstance(entry['pattern'], str):
            if pattern in non_fuzzy_list: token_patterns = [{'LOWER': word.lower()} for word in entry['pattern'].split()]
            else: token_patterns = [{'LOWER': {'FUZZY1': word.lower()}} for word in entry['pattern'].split()]
        else:
            if pattern in non_fuzzy_list: token_patterns = [{'LOWER': word.lower()} for word in entry['pattern'].split()]
            else: token_patterns = [{'LOWER': {'FUZZY1': token['LOWER']}} for token in entry['pattern']]
        spacy_patterns.append({'label': label, 'pattern': token_patterns})
    return spacy_patterns


def clean_label(label_list):
    rectified_labels = []
    for entity in label_list:
        start, end, ent_type, ent_text = entity
        corrected_text = ent_text.strip()
        if corrected_text != ent_text:
            new_start = start + ent_text.find(corrected_text)
            new_end = new_start + len(corrected_text)
            rectified_labels.append([new_start, new_end, ent_type, corrected_text])
        else:
            rectified_labels.append(entity)
    return rectified_labels


# Data ingestion + tagging creation !!!!!!
wiki_api = WikipediaAPI()
processor = BaseTextProcessor()


categories = ["Mathematics", "Programming", "Probability & Statistics", "People",
              "Organizations", "Academic Disciplines", "Machine Learning", "Publications"]
category_counts = defaultdict(int)


INGEST = True
if INGEST:
    test_pattern_fuzzy = convert_fuzzy_match(ner_pattern, non_fuzzy_list)
    unique_id = -2
    count = 0.0
    emb_tbl = ArticleTable(rds_dbname, rds_user, rds_password, rds_host, rds_port)
    for TITLE in tqdm(set(ner_articles), desc='Progress'):
        title, page_id, final_text = wiki_api.fetch_article_data(TITLE)
        article = Article(title=title, id=page_id, text=final_text, text_processor=processor)
        article.process_text_pipeline(processor, SECTIONS_TO_IGNORE)
        article.process_metadata_labeling(processor, test_pattern_fuzzy)
        for (key, text), metadata in zip(article.text_dict.items(), article.metadata_dict.values()):
            text = re.sub(r'[\n\t]+', ' ', text)
            emb_tbl.add_record(str(uuid.uuid4()), key, article.title, text, json.dumps(metadata))
    emb_tbl.close_connection()

    for category, count in category_counts.items():
        print(f"Category '{category}': {count} tags")

"""
This builds a JsonL object that can be used in doccano labelling. 
Check out docker build to avoid issues
Username & password inside cover page of statistics for DS book
"""


"""
The titles in the db are skewed. Title is the text while text is the title...
"""

BUILD_JSONL = False
if BUILD_JSONL:
    emb_df = ArticleTable(rds_dbname, rds_user, rds_password, rds_host, rds_port)
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
Need to distinguish build of training and test data, .3 proba of test .7 proba of train
"""

BUILD_SPACY_DATA = True
if BUILD_SPACY_DATA:
    emb_df = ArticleTable(rds_dbname, rds_user, rds_password, rds_host, rds_port)
    art_df = emb_df.get_all_data_pd()
    art_df['label'] = art_df['label'].apply(clean_label)
    training_data = []
    for _, row in art_df.iterrows():
        entities = [tuple(entity) for entity in row['label']]
        training_data.append((row['title'], entities))
    nlp = spacy.load('en_core_web_sm')
    db_train, db_test = DocBin(), DocBin()
    for idx, (text, annotations) in enumerate(training_data):
        train_test_split = random.choices(population=['train', 'test'], weights=[0.75, 0.25])[0]
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label, pattern in annotations:
            span = doc.char_span(start, end, label=label)
            if span is not None: ents.append(span)
        ents = spacy.util.filter_spans(ents)
        doc.ents = ents
        if train_test_split == 'train':
            db_train.add(doc)
        else:
            db_test.add(doc)

    db_train.to_disk("./train.spacy")
    db_test.to_disk("./test.spacy")


