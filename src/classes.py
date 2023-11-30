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

"""
Line 13 - 50ish is for wikipedia API calls
Line 50ish - 70ish is for the vector db
Line 124 to end is for article and text processing classes

"""

text, hf_token, hf_endpoint

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
    def clean_text(self, text):
        pass

    @abstractmethod
    def stem_text(self, text):
        pass

    @abstractmethod
    def lemmatize_text(self, text):
        pass

    @abstractmethod
    def tokenize_text(self, text):
        pass

    @abstractmethod
    def embed_vector(self, text):
        pass

    @abstractmethod
    def build_metadata(self, text):
        pass

    @abstractmethod
    def build_summary(self, text, hf_token, hf_endpoint):
        # Summary Building for storing in DB and return to
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
        return text

    def build_summary(self, text, hf_token, hf_endpoint):
        summary = hf_request(hf_token, hf_endpoint)
        return summary

# todo: get the response str to upsert pipeline correct Nov 29th
@dataclass
class Article:
    category: str
    title: str
    id: str
    text: List[str]
    metadata: Optional[Dict]
    embedding: Optional[List[float]]
    summary: str = field(default='')
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