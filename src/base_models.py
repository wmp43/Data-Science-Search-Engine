# src.base_classes.py
import numpy as np
from dataclasses import dataclass, field
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Dict, Any, Optional, Tuple
from config import ner_pattern, non_fuzzy_list
from pydantic import BaseModel
import uuid

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

    # def build_title_embeddings(self):
    #     model = SentenceTransformer('all-MiniLM-L6-v2')
    #     embeddings = model.encode(self.clean_title, show_progress_bar=True).reshape(1, -1)
    #     return embeddings
    #
    # def predict_relevancy(self, model_path: str, embeddings) -> bool:
    #     model = XGBClassifier()
    #     model.load_model(model_path)
    #     proba = model.predict_proba(embeddings)
    #     if proba > 0.45:
    #         return True
    #     else:
    #         return False
    #
    # def build_optionals(self, returned_items):
    #     self.id = uuid.uuid4()
    #     self.return_items = returned_items


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
        "embedding": np.ndarray,
        "encoding":
        "id": 1234,
        "title": "title of article",
        "section": section_heading
        "metadata": [ent.text, ent.label]
        }
    }
    """


@dataclass
class Article:
    title: str
    id: str
    text: str
    text_dict: Dict[str, str] = field(default_factory=dict)  # Text Dict and embedding Dict match on section heading.
    embedding_dict: Dict[str, Tuple] = field(default_factory=dict)  # [0] embedding for vec search -- [1] encoding
    metadata_dict: Dict[str, Dict] = field(default_factory=dict)
    text_processor: any = None

    def process_text_pipeline(self, exclude_section):
        text_dict = self.text_processor.build_section_dict(self, exclude_section)
        clean_text_dict = self.text_processor.remove_curly_brackets(text_dict)
        chunked_text_dict = self.text_processor.chunk_text_dict(clean_text_dict)
        self.text_dict = chunked_text_dict
        return self

    def process_embedding_pipeline(self):
        """
        This method may only be invoked after the process_text_pipeline method.
        This will return a dictionary with section headings and token lens for
        Cost approximation and text chunking
        :param text_processor: BaseTextProcessor Class
        :return: self
        """
        embed_dict = self.text_processor.build_embeddings(self)
        self.embedding_dict = embed_dict
        return self

    def process_metadata_pipeline(self):
        """
        December 19th
        https://spacy.io/usage/rule-based-matching
        This method should build a metadata object for each embedded section.
        By extension I guess it should also support
        :param text_processor: TextProcessor object to build metadata
        :return: self
        """
        metadata_dict = self.text_processor.build_metadata(self)
        self.metadata_dict = metadata_dict
        return self

    def get_categories(self, text_processor):
        cats_list = text_processor.build_categories(self)
        self.categories = cats_list

    def process_metadata_labeling(self, text_processor, pattern):
        """
        :param text_processor: Base Text Processor
        :return: Builds the entities for metadata labeling for ner
        """
        entities_dict = text_processor.build_training_metadata(self, pattern)
        self.metadata_dict = entities_dict

    def show_headings(self, text_processor):
        text_processor.extract_headings(self)
        return self

    def show_token_len_dict(self, text_processor) -> Dict:
        len_dict = text_processor.build_token_len_dict(self)
        return len_dict

    def __str__(self):
        return f"Article {self.id}: {self.title}"

    def json_serialize(self):
        """
        This function should prepare the contents of this data class to be passed in a POST request and upserted to db
        :return: Bool
        """
        article_dict = {
            'title': self.title,
            'id': self.id,
            'text': self.text,
            'text_dict': self.text_dict,
            'embedding_dict': self.embedding_dict,
            'metadata_dict': self.metadata_dict,
            'text_processor': None
        }

        return article_dict

    @staticmethod
    def json_deserialize(data: dict):
        return Article(**data)


@dataclass
class Query:
    raw_query: str
    expanded_query: any = None
    query_processor: any = None  # This should really be a QueryProcessor class
    query_visualizer: any = None  # query visualizer class
    vector_tbl: any = None  # this should be VectorTable class
    query_tbl: any = None  # tbl with the queries
    embedding: any = None   # embedding from model
    language_results: str = None  # results from language_model(vector search + query)
    search_results: str = None   #


    def process(self):
        """
        from text to embedding
        :return: None
        """
        EXPAND = False
        if EXPAND:
            # todo: implement query expansion
            query_expanded = self.query_processor.expand_query(self.raw_query)
            self.expanded_query = query_expanded
            query_embedded = self.query_processor.embed_query(query_expanded)
        else:
            query_embedded = self.query_processor.embed_query(self.raw_query)
        self.embedding = query_embedded[0]

    def execute(self):
        """
        sends self.embedding to vector table
        :param vector_tbl:
        :return: results -- what dtype?
        """
        search_res_list = self.vector_tbl.query_vectors(self.embedding, top_n=20)
        # returns a list of tuples need to figure out what is in each tuple to pass correct items
        self.search_results = search_res_list
        # returns list of tuples:
        # [("Title 1", "Encoding 1", "Metadata 1", "Vector 1"),
        # ("Title 2", "Encoding 2", "Metadata 2", "Vector 2")]

    def re_ranker(self):
        """
        Re rank results from the returned db
        :return:
        """
        EXPAND = False
        if EXPAND: ranked_results = self.query_processor.rerank(self.expanded_query, self.results)
        else: ranked_results = self.query_processor.rerank(self.raw_query, self.results)
        self.search_results = ranked_results

    def network_graph(self):
        self.query_visualizer.plot_graph(self.results)


    def language_model(self):
        # Calls the language model and returns a string
        response = self.query_processor.call_langauge_model(self.search_results, )
        self.language_results = response



    def query_to_tbl(self):
        self.query_tbl.add_record(self.raw_query,  # raw query
                                  self.embedding,  # embedding
                                  self.expanded_query,  # Expanded query if using Hyde
                                  " This should be the LLM response",
                                  # todo: LLM response. Deploy to Sagemaker w/ api call?
                                  uuid.uuid4())


