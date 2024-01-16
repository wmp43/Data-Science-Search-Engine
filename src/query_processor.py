from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import re
import os
import requests
import json
import tiktoken
import numpy as np
from src.base_models import Query
from langchain.text_splitter import (TokenTextSplitter,
                                     CharacterTextSplitter,
                                     SpacyTextSplitter)

from config import ner_pattern, entity_ruler_config, non_fuzzy_list
from spacy.lang.en import English


class QueryProcessor(ABC):
    """
    Article:TextProcessor
    as
    Query:QueryProcessor
    """

    @abstractmethod
    def embed_query(self, query):
        pass

    @abstractmethod
    def expand_query(self, query) -> str:
        pass

    @abstractmethod
    def rerank(self):
        pass


class BaseQueryProcessor(QueryProcessor):
    """
    This class implements the ABC of QueryProcessor
    Using an Abstract Base Class allows us to experiment with a few different methods
    of query expansion
    """
    def expand_query(self, query: str) -> str:
        """
        Expand query for better results
        :return: string?
        """
        pass

    def embed_query(self, query):
        """
        Method: Matches patterns to text and builds the return data structure.
        :param query: Query Object
        :param  model: The loaded fine-tuned model
        :return List as follows:
        data = [
        ("This is text with important information",[(start_span, end_span, label)]),
        ("important information and I promise it is important",[(start_span, end_span, label), (start_span, end_span, label)])
        ]
        """
        payload, api_url = {'article': query}, "http://127.0.0.1:5000/query_api"
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            encoded_query = response.json()
            return encoded_query
        else:
            print("Error in NER API call:", response.status_code, response.text)

    def rerank(self, results):
        """
        :return:
        """




