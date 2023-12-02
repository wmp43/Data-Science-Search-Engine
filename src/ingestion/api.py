"""
Defines the api call object.
"""

import requests
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
from openai import OpenAI
import os
import mwclient
import mwparserfromhell

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
        site = mwclient.Site('en.wikipedia.org')
        page = site.pages[article_title]

        # Dictionary to store section titles and their corresponding text
        sections = {}

        if page.exists:
            wikitext = page.text()
            parsed = mwparserfromhell.parse(wikitext)

            # Iterate over sections
            for section in parsed.get_sections(include_lead=True, flat=True):
                # Use section title or "Lead section" as the key
                title = section.filter_headings()[0].title.strip() if section.filter_headings() else "Lead section"
                # Clean up the section text
                text = section.strip_code().strip()
                sections[title] = text
        else:
            print(f"The page '{article_title}' does not exist.")

        return sections



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
        hf_token = os.getenv("hf_token")
        headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
        payload = {"inputs": f"{text_chunk}"}

        response = requests.post(self.endpoint, headers=headers, data=json.dumps(payload))
        response_json = response.json()

        return response_json[0]["generated_text"]


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



