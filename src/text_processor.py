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

    def extract_headings(self, article):
        normalized_text = re.sub(r'={3,}', '==', article.text)
        section_pattern = r'(==\s*[^=]+?\s*==)'
        headings = re.findall(section_pattern, normalized_text, re.MULTILINE)
        print("Extracting headings...")
        for heading in headings:
            print(heading)
        if not headings:
            print("No major headings found.")


    def build_section_dict(self, article: Article, exclude_sections: List[str]) -> Dict[str, str]:
        """
        Split on section headings and exclude specified headings.
        :param article: Article object containing the text.
        :param exclude_sections: List of headings to exclude.
        :return: Dictionary with section headings as keys and text as value.
        """
        normalized_text = re.sub(r'={3,}', '==', article.text)
        normalized_text = re.sub(r'\{\{[^}]*?\}\}', '', normalized_text)

        text = re.sub(r'\{\{[^}]*?\}\}', '', article.text)

        section_pattern = r'(==\s*[^=]+?\s*==)'
        parts = re.split(section_pattern, normalized_text)
        sections = {'Introduction': parts[0].strip()}

        for i in range(1, len(parts), 2):
            section_title = parts[i].strip("= ").strip()

            if section_title not in exclude_sections:
                sections[section_title] = parts[i + 1].strip()

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
