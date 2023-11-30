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
import re
from ingestion.api import HuggingFaceAPI, OpenAIAPI

"""
vector = {
    "embedding": [[f32], [f32], [f32]]
    "id": [wiki_id_{idx}],
    "title": "bayes_theorem",
    "summary": [str,str,str],
    "metadata":{
        "categories": ['bayesian_statistics', 'posterior_probability', 'bayes_estimation'],
        "mentioned_people": ['william_bayes'],
        "mentioned_places": ['london'],
        "mentioned_topics": ['bayesian_economics', 'bayesian_deep_learning']
    }
}
"""
hf_token, hf_endpoint = os.getenv("hf_token"), os.getenv("hf_endpoint")
oai_token, oai_endpoint = os.getenv("oai_token"), os.getenv("oai_endpoint")




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


@dataclass
class Article:
    category: str
    title: str
    id: str
    text: List[str]
    metadata: Optional[List[Dict]]
    embedding: Optional[List[List[float]]]
    text_processor: TextProcessor = field(default=BaseTextProcessor())

    def process_text_pipeline(self, text_processor):
        # This should include a pipeline to process text
        clean_text = text_processor.clean_text(self.text)
        return clean_text

    def update_categories(self, new_category):
        # If article found in new
        return new_category

    def __str__(self):
        return f"Article {self.id}: {self.title}"


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
    def build_summary(self, article: Article(), api: HuggingFaceAPI()) -> List[str]:
        pass


class BaseTextProcessor(TextProcessor):
    """
    Need some way to split the returned text of tokenize and clean.
    t&c -> chunk -> hf_clean -> vectors
    [1] -> [many] -> [many cleaned] -> [[f32],[f32],[f32]]
    """

    def tokenize_and_clean(self):
        tokens = self.text.split()
        cleaned_tokens = [token for token in tokens if not re.match(r'\\[a-zA-Z]+', token)]
        cleaned_text = ' '.join(cleaned_tokens)
        return cleaned_text

    def hf_clean(self, article: Article(), HFapi: HuggingFaceAPI()):
        hf_api = HFapi(token=hf_token, endpoint=hf_endpoint)
        # Edits the summary at text chunk index in order to maintain the proper text to summary
        for idx, text in enumerate(article.text):
            article.text[idx] = hf_api.fetch_summary(text)

    def fetch_embeddings(self, article: Article(),OAIapi: OpenAIAPI()):
        """
        :param article:
        :param OAIapi:
        :return:
        """
        oai_api = OAIapi(token=openai_token, endpoint=openai_endpoint)
        for idx, text in enumerate(article.text):
            article.embedding[idx] = list(oai_api.fetch_embeddings(text))

    def build_metadata(self, article: Article()):
        """" Go through each chunk and build some metadata !!!!
        :param article: article object containing text,
        :return: dictionary with relevant metadata
        """
        for idx, text_chunk in enumerate(article.text):
            dict = {'categories': [article.category], "mentioned_people": [],
                    "mentioned_places": [], "mentioned_topics": []}

            # need to search through text and find relevant mentions

            article.metadata[idx] = dict
            # For each chunk of text



# todo: get the response str to upsert pipeline correct Nov 29th
