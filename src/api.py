"""
Defines the api call object.
"""

import requests
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
from openai import OpenAI
import os
import re
import mwclient
import tiktoken
import mwparserfromhell


class WikipediaAPI(BaseModel):
    def fetch_category_data(self, category: str) -> List:  # Wikimedia API Call for categories.
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

    def fetch_article_data(self, article_title):
        URL = "https://en.wikipedia.org/w/api.php"
        PARAMS = {
            "action": "query",
            "prop": "extracts",
            "titles": article_title,
            "explaintext": True,
            "format": "json"
        }
        try:
            response = requests.get(url=URL, params=PARAMS)
            data = response.json()
            pages = data["query"]["pages"]

            page_id = next(iter(pages))
            content = pages[page_id].get("extract", "")
            tokens = content.split()
            cleaned_tokens = [token for token in tokens]
            cleaned_text = ' '.join(cleaned_tokens)
            spaceless_text = re.sub(r'[\n\t]+', '', cleaned_text)
            final_text = re.sub(r' {2}', ' ', spaceless_text)
            title = pages[page_id].get("title", "")
            return title, page_id, final_text

        except requests.exceptions.RequestException as e:
            print(f"HTTP Request failed: {e}")
        except Exception as e:
            print(f"Error processing article data: {e}")

        return None, None, None






# class HuggingFaceAPI(BaseModel):
#     """
#     Llama model deployed to HF for text cleaning and summarization
#
#     """
#     token: str
#     endpoint: str
#
#     def fetch_summary(self, text_chunk: str) -> str:
#         """
#         :param text_chunk: Chunk of tokenized text that is going to get summarized
#         :return: summary str
#         """
#         hf_token = os.getenv("hf_token")
#         headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
#         payload = {"inputs": f"{text_chunk}"}
#
#         response = requests.post(self.endpoint, headers=headers, data=json.dumps(payload))
#         response_json = response.json()
#
#         return response_json[0]["generated_text"]
#
#
# class OpenAIAPI(BaseModel):
#     """
#     ahhh ahh ahhhhh ahhh ahhh
#     """
#     token: str
#
#     def fetch_summary(self, text_chunk):
#         client = OpenAI(
#             organization='org-FkeFbkQ4XzxxQoN5PJ9APM7D'
#         )
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a mathematical text summarizer. Summarize the following:"},
#                 {"role": "user", "content": text_chunk}
#             ]
#         )
#
#         summary = response['choices'][0]['message']['content']
#         return summary
#
#
#
# def remove_nested_curly_braces(text):
#     stack = []
#     to_remove = []
#     text_list = list(text)
#
#     for i, char in enumerate(text_list):
#         if char == '{':
#             stack.append(i - 1)
#         elif char == '}':
#             if stack:
#                 start = stack.pop()
#                 if not stack:
#                     to_remove.append((start, i))
#
#     for start, end in reversed(to_remove):
#         del text_list[start:end + 1]
#
#     return ''.join(text_list)