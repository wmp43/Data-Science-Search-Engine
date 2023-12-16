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
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
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
    text: str
    text_dict: Optional[Dict[str, List[Tuple[str, np.ndarray]]]]
    metadata: Optional[Dict[str, List[Dict[str, any]]]]
    text_processor: TextProcessor = field(default=BaseTextProcessor())
    """
    {
    section_heading: [(cleaned_text: str, embedding: np.ndarry[float64]), 
                      (cleaned_text: str, embedding: np.ndarry[float64]),
                      (cleaned_text: str, embedding: np.ndarry[float64])]
    }
    
    Then each tuple from the values list can then build the vectorDB record:
    
    metadata_example = {
        "Introduction": [
            {
                "part": 1,
                "categories": ["Basic Concepts", "Overview"],
                "mentioned_people": ["John Smith"],
                "mentioned_places": ["Paris"],
                "mentioned_topics": ["Introduction to Statistics"]
            },
            {
                "part": 2,
                "categories": ["Overview"],
                "mentioned_people": ["Jane Doe"],
                "mentioned_places": ["New York"],
                "mentioned_topics": ["Historical Context"]
            }
        ],
        "Methodology": [
            {
                "part": 1,
                "categories": ["Research Methods"],
                "mentioned_people": ["Alice Johnson"],
                "mentioned_places": ["London"],
                "mentioned_topics": ["Survey Design"]
            },
            {
                "part": 2,
                "categories": ["Data Analysis"],
                "mentioned_places": ["Berlin"],
                "mentioned_topics": ["Data Collection Techniques"]
            }
        ],
    }

Vector record
    vector_record = {
        "embedding": np.ndarray
        "id": 1234,
        "title": "title of article",
        "metadata": {"section heading":{
                "categories": ['bayesian_statistics', 'posterior_probability', 'bayes_estimation'],
                "mentioned_people": ['william_bayes'],
                "mentioned_places": ['london'],
                "mentioned_topics": ['bayesian_economics', 'bayesian_deep_learning']
                }
        }
    }
    """

    def process_text_pipeline(self, text_processor, exclude_section):
        # This should include a pipeline to process text
        processor = text_processor()
        text_dict = processor.build_section_dict(self, exclude_section)
        clean_text_dict = processor.remove_curly_brackets(text_dict)
        self.text_dict = clean_text_dict


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

    def build_section_dict(self, article: Article(), exclude_sections: List) -> Dict:
        """
        Split on section headings
        :return: Dictionary with section headings as keys and text as value.

        """
        section_pattern = r'==[^=]*=='
        subsection_pattern = r'===[^=]*==='
        text = re.sub(subsection_pattern, '', article.text)
        parts = re.split(section_pattern, text)

        sections = {}
        sections["Introduction"] = parts[0].strip()

        section_headers = re.findall(section_pattern, text)
        for i, header in enumerate(section_headers):
            clean_header = header.strip("= ").strip()

            if clean_header not in exclude_sections:
                sections[clean_header] = parts[i + 1].strip() if i + 1 < len(parts) else ""
        return sections



    def remove_curly_brackets(self, section_dict) -> Dict:
        """Removes the curly braces (latex) from the values in each section."""
        cleaned_sections = {}

        for section, text in section_dict.items():
            stack = []
            to_remove = []
            text_list = list(text)

            for i, char in enumerate(text_list):
                if char == '{':
                    stack.append(i)
                elif char == '}':
                    if stack:
                        start = stack.pop()
                        if not stack:
                            to_remove.append((start, i))

            for start, end in reversed(to_remove):
                del text_list[start:end + 1]

            cleaned_sections[section] = ''.join(text_list)

        return cleaned_sections


"""    
For each text chunk, we should use langchain's functionality for text splitting and embedding
- need to figure out 


        {section: text, 
        section: text, 
        section: text,
        section: text}
    
"""

    def build_embeddings(self, article: Article(), OAIapi: OpenAIAPI(), section_dict):
        """
        :param article:
        :param OAIapi:
        :return:
        """
        oai_api = OAIapi(token=1, endpoint=1)
        for idx, text in enumerate(article.text):
            article.embedding[idx] = list(oai_api.fetch_embeddings(text))
        return 1

    def build_metadata(self, article: Article(), section_dict):
        """"
        :param article: article object containing text,
        :param section_dict: dictionary with sections
        :return: dictionary with relevant metadata
        """
        for idx, text_chunk in enumerate(article.text):
            dict = {'categories': [article.category], "mentioned_people": [],
                    "mentioned_places": [], "mentioned_topics": []}

            # need to search through text and find relevant mentions

            article.metadata[idx] = dict
            # For each chunk of text
        return None
