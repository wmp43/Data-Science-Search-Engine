import uuid
from typing import List, Dict, Any, Optional

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

"""
Line 13 - 50ish is for wikipedia API calls
Line 50ish - 70ish is for the vector db
Line 124 to end is for article and text processing classes

"""


@dataclass
class WikipediaAPI:
    def get_category_data(self, category):  # Wikimedia API Call for categories.
        URL = "https://en.wikipedia.org/w/api.php"
        PARAMS = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": "max",
            "format": "json"
        }

        HEADERS = {
            'User-Agent': 'Myles Pember (https://github.com/wmp43/wikiSearch/tree/master; wmp43@cornell.edu)'
        }

        response = requests.get(url=URL, params=PARAMS, headers=HEADERS)
        data = response.json()
        # pretty_json = json.dumps(data, indent=4)
        # print(pretty_json)
        response_list = [(category[9:], member['title'], member["pageid"], member["ns"]) for member in
                         data['query']['categorymembers']]
        return response_list

    def fetch_article_content(self, title):
        """
        Fetches the plain text content of a Wikipedia article by its title.

        This method retrieves the content of a Wikipedia page using the Wikipedia API. It
        extracts the plain text without any markup or HTML. Additionally, it constructs
        the direct URL to the Wikipedia page based on the title.

        :param title: The title of the Wikipedia article to fetch.
        :return: A tuple containing:
            - title (str): The normalized title of the article.
            - page_id (str): The page ID of the article in Wikipedia.
            - content (str): The plain text extract of the article content.
            - wiki_url (str): The direct URL to the Wikipedia page.
        """
        URL = "https://en.wikipedia.org/w/api.php"
        PARAMS = {
            "action": "query",
            "prop": "extracts",
            "titles": title,
            "explaintext": True,
            "format": "json"
        }

        response = requests.get(url=URL, params=PARAMS)
        data = response.json()
        pages = data["query"]["pages"]
        page_id = next(iter(pages))
        content = pages[page_id].get("extract", "")
        normalized_title = pages[page_id].get("title", "")
        wiki_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        return normalized_title, page_id, content, wiki_url


class VectorDB:
    def __init__(self, client, schema):
        self.clint = client
        self.schema = schema

    def connect_to_db(self, client):
        pass

    def get_by_article(self, article_id):
        pass

    def add(self, article_id, data):
        pass

    def update(self, article_id, data):
        pass


@dataclass
class Category:
    super_category: Optional[str]  # Need to handle categories that exist within multiple super-categories
    id: Optional[int]
    title: str
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



class RelationalDB(ABC):
    """
    Schema: id:int (PK) | wiki_id:int | category: str | wiki_url:str | super_categories: List[str] | sub_categories:List[str] | sub_articles: List[str]
    """

    @abstractmethod
    def connect(self, credentials: Dict[str, Any]):
        pass

    @abstractmethod
    def add_record(self, category: Category):
        pass

    @abstractmethod
    def get_record(self, category_id: int) -> Optional[Category]:
        pass

    @abstractmethod
    def update_record(self, category: Category):
        pass

    @abstractmethod
    def add_super_category(self, category_id: int, super_category: str):
        """
        Implementation for appending a super category to the super_category list.
        This handles the case for when a category can be reached from multiple super-categories
        Cat:Machine Leanring -> Cat:Neural Networks
        Cat: Artifical Intelligence -> Cat: Neural networks

        :param category_id: ID for category
        :param super_category: new super category
        :return:
        """
        pass


class LocalRelationalDB(RelationalDB):
    def connect(self, credentials: Dict[str, Any]):
        # Implement actual connection logic
        pass

    def add_record(self, category: Category):
        # Logic to add a category to the database
        pass

    def get_record(self, category_id: int) -> Optional[Category]:
        # Logic to retrieve a category from the database
        pass

    def update_record(self, category: Category):
        # Logic to update a category in the database
        pass

    def add_super_category(self, category_id: int, super_category: str):
        pass


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
    def build_summary(self, text):
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
        return text


@dataclass
class Article:
    category: str
    title: str
    id: str
    text: str
    embedding: Optional[List[float]] = field(default_factory=[0.0])
    wiki_url: str = field(default='')
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