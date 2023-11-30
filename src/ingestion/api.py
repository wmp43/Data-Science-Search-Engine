"""
Defines the api call object.
"""

import requests
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
from openai import OpenAI
import os

openai_token = os.getenv("openai_token")


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

    def get_article_data(self, article_title):
        """
        :param article_title: title of article
        :return: returns title, id, content of article
        """
        URL = "https://en.wikipedia.org/w/api.php"
        PARAMS = {
            "action": "query",
            "prop": "extracts",
            "titles": article_title,
            "explaintext": True,
            "format": "json"
        }

        response = requests.get(url=URL, params=PARAMS)
        data = response.json()
        pages = data["query"]["pages"]
        page_id = next(iter(pages))
        content = pages[page_id].get("extract", "")
        title = pages[page_id].get("title", "")
        return title, page_id, content


class HuggingFaceAPI(BaseModel):
    """
    Llama model deployed to HF for text cleaning and summarization

    """
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


class OpenAIAPI(BaseModel):
    """
    ahhh ahh ahhhhh ahhh ahhh
    """
    token: str

    def fetch_summary(self, text_chunk):
        client = OpenAI(
            organization='org-FkeFbkQ4XzxxQoN5PJ9APM7D'
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a mathematical text summarizer. Summarize the following:"},
                {"role": "user", "content": text_chunk}
            ]
        )

        summary = response['choices'][0]['message']['content']
        return summary



