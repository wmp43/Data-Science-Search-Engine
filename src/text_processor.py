# src.text_processor.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from src.models import Article, Category
import re
import numpy as np

class TextProcessor(ABC):

    @abstractmethod
    def build_section_dict(self, article, exclude_sections):
        pass

    @abstractmethod
    def remove_curly_brackets(self, section_dict):
        pass

    @abstractmethod
    def build_embeddings(self, article):
        pass

    @abstractmethod
    def build_metadata(self, article, section_dict):
        pass

class BaseTextProcessor(TextProcessor):
    """
    Need some way to split the returned text of tokenize and clean.
    t&c -> chunk -> hf_clean -> vectors
    [1] -> [many] -> [many cleaned] -> [[f32],[f32],[f32]]
    """

    def build_section_dict(self, article: Article, exclude_sections: List) -> Dict:
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

    def build_embeddings(self, article: Article):
        """
        :param article:
        :param OAIapi:
        :return:
        """
        return 1

    def build_metadata(self, article: Article, section_dict:{}):
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
