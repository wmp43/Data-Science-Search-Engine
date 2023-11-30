"""
Defines the article and document classes.
Each of these classes has an attribute that defines how they get stored

- Documents have a relational db class as an attribute

- Articles have a vector db class as an attribute.
"""
import numpy as np
import requests
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import requests
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import os
from ingestion.api import HuggingFaceSummaryAPI

"""
"""
token, endpoint = os.getenv("hf_token"), os.getenv("hf_endpoint")


@dataclass
class Category:
    super_category: Optional[str]  # Need to handle categories that exist within multiple super-categories
    id: Optional[int]
    title: f'Category:{str}'
    clean_title: Optional[str]
    return_items: Optional[List[str]] = field(default_factory=list)  # Assuming articles are stored as a list of strings

    def clean_title_method(self):
        # Preapre the title for the clf
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        text = self.title.replace('Category:', '').strip()
        text = self.title.replace('-', ' ')
        text = self.title.replace('_', ' ')
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
        self.clean_title = ' '.join(tokens)

    def build_title_embeddings(self):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(self.clean_title, show_progress_bar=True).reshape(1, -1)
        return embeddings


    def predict_relevancy(self, model_path: str, embeddings) -> bool:
        model = XGBClassifier()
        model.load_model(model_path)
        proba = model.predict_proba(embeddings)
        if proba > 0.45:
            return True
        else:
            return False

    def build_optionals(self, returned_items):
        self.id = uuid.uuid4()
        self.return_items = returned_items


class TextProcessor(ABC):
    @abstractmethod
    def stem_text(self, text):
        pass

    @abstractmethod
    def lemmatize_text(self, text):
        pass

    @abstractmethod
    def embed_vector(self, text):
        pass

    @abstractmethod
    def build_metadata(self, text):
        pass

    @abstractmethod
    def build_summary(self, article: Article(), api: HuggingFaceSummaryAPI()) -> List[str]:
        pass


class BaseTextProcessor(TextProcessor):
    """
    The base text processor will be step 1 for developing embeddings
    - Clean, lower, tokenize, embed.
    - There may be ways to process text that include lemmitization, stemming, etc.
    """

    def clean_text(self, text):
        # Actual implementation for processing text
        return text

    def stem_text(self, text):
        return text

    def lemmatize_text(self, text):
        return text

    def tokenize_text(self, text):
        return text

    def embed_vector(self, text):
        return text

    def build_metadata(self, text):
        """" Build metadata for every chunk
        :param text:
        :return:
        """
        return text

    def build_summary(self, article: Article(), api: HuggingFaceSummaryAPI()) -> str:
        hf_api = HuggingFaceSummaryAPI(token=hf_token, endpoint=hf_endpoint)
        # Edits the summary at text chunk index in order to maintain the proper text to summary
        for idx, text in enumerate(article.text):
            article.summary[idx] = hf_api.fetch_summary(text)


# todo: get the response str to upsert pipeline correct Nov 29th
@dataclass
class Article:
    category: str
    title: str
    id: str
    text: List[str]
    summary: List[str]
    metadata: Optional[Dict]
    embedding: Optional[List[float]]
    text_processor: TextProcessor = field(default=BaseTextProcessor())


    def process_text_pipeline(self, text, text_processor):
        # This should include a pipeline to process text
        clean_text = text_processor.clean_text(self.text)
        return clean_text

    def update_categories(self, new_category):
        # If article found in new
        return new_category

    def save_to_database(self, db):
        # Logic to save article to the database
        pass

    def __str__(self):
        return f"Article {self.id}: {self.title}"