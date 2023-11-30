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


    def clean_text(self):
        """
        :return: Cleans and chunks
        """
        pass



class HuggingFaceSummaryAPI(BaseModel):
    token: str
    endpoint: str

    def fetch_summary(self, text_chunk) -> str:
        """
        :param text_chunk: Chunk of tokenized text that is going to get summarized
        :return: summary str
        """
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = {"inputs": text_chunk}

        response = requests.post(self.endpoint, headers=headers, data=json.dumps(payload))
        response_json = response.json()
        return response_json