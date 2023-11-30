"""
Defines the api call object.
"""

import requests
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json


class WikipediaAPI(BaseModel):
    def get_category_data(self, category: str) -> List:  # Wikimedia API Call for categories.
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

    def clean_text(self) -> List:
        """
        :return: Cleans and chunks into list of tokens
        """
        pass


class HuggingFaceSummaryAPI(BaseModel):
    token: str
    endpoint: str

    def fetch_summary(self, text_chunk: str) -> str:
        """
        :param text_chunk: Chunk of tokenized text that is going to get summarized
        :return: summary str
        """
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = {"inputs": text_chunk}

        response = requests.post(self.endpoint, headers=headers, data=json.dumps(payload))
        response_json = response.json()

        return response_json['text']


# class OpenAIAPI(BaseModel):

