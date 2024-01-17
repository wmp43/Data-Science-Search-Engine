from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import requests
import cohere
import tiktoken
from config import ner_pattern, entity_ruler_config, non_fuzzy_list, COHERE_API_KEY
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
    def rerank(self, query, results):
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
        payload, api_url = {'query': query}, "http://127.0.0.1:5000/query_api"
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            encoded_query = response.json()
            return encoded_query
        else:
            print("Error in Query embed call:", response.status_code, response.text)

    def rerank(self, query, results):
        """
        :param: results -- needs to include text. The basic implementation can be found here:
        https://txt.cohere.com/rerank/
        :return:
        """
        # Results should be a list of text
        # List as dtype string has issues with rerank. Should just build new column 'text' in
        enc = tiktoken.get_encoding("cl100k_base")
        title, encoding_str, _metadata, _embedding = results
        encoding_list = [int(num) for num in encoding_str.strip('[]').split(',')]
        decoded_encoding = enc.decode(encoding_list)
        co = cohere.Client(COHERE_API_KEY)
        results = co.rerank(query=query, documents=decoded_encoding, top_n=10, model="rerank-multilingual-v2.0")
        return results




