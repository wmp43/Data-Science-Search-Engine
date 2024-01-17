# src.base_classes.py
import numpy as np
from dataclasses import dataclass, field
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Dict, Any, Optional, Tuple
from config import ner_pattern, non_fuzzy_list
from pydantic import BaseModel
import plotly.express as px
import tiktoken
from sklearn.manifold import TSNE
import json
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
    """

    """
    text: str
    query_processor: any = None  # This should really be a QueryProcessor class
    vector_tbl: any = None  # this should be VectorTable class
    embedding: any = None
    results: any = None

    def process(self):
        """
        from text to embedding
        :return: None
        """
        EXPAND = False
        if EXPAND:
            query_expanded = self.query_processor.expand_query(self.text)
            query_embedded = self.query_processor.embed_query(query_expanded)
        else:
            query_embedded = self.query_processor.embed_query(self.text)
        self.embedding = query_embedded[0]

    def execute(self):
        """
        sends self.embedding to vector table
        :param vector_tbl:
        :return: results -- what dtype?
        """
        res_list = self.vector_tbl.query_vectors(self.embedding)  # optionally top-n returns
        self.results = res_list
        # returns list of tuples:
        # [("Title 1", "Encoding 1", "Metadata 1", "Vector 1"),
        # ("Title 2", "Encoding 2", "Metadata 2", "Vector 2")]

    def re_ranker(self):
        """
        Re rank results from the returned db
        :return:
        """
        reranked_res = self.query_processor.rerank(self.text, self.results)
        print(reranked_res)
        self.results = reranked_res


class QueryVisualization:
    def __init__(self, query_results):
        self.query_results = query_results
        self.graph = nx.Graph()


    """
    Network Viz
    """
    #todo: make sure this works. Check out vizualizations.py
    def _process_metadata(self):
        """Process metadata and add edges to the graph."""
        for _, _, metadata_str in self.query_results:
            if metadata_str:  # Check if metadata is not empty
                metadata = json.loads(metadata_str)  # Convert JSON string to dict
                for key, values in metadata.items():
                    for value in values:
                        self.graph.add_edge(key, value)

    def plot_graph(self):
        """Plot the graph using NetworkX."""
        self._process_metadata()  # Process metadata and construct the graph
        plt.figure(figsize=(10, 8))
        nx.draw(self.graph, with_labels=True)
        plt.show()

    """
    Scatter vizualtion
    """
    def _reduce_dimensions(self, vectors, method='PCA', n_components=3):
        """
        Reduce the dimensions of the vectors to 3 using PCA or t-SNE.
        """
        if method == 'PCA':
            model = PCA(n_components=n_components)
        elif method == 'TSNE':
            model = TSNE(n_components=n_components)
        else:
            raise ValueError("Invalid dimensionality reduction method")

        reduced_vectors = model.fit_transform(vectors)
        return reduced_vectors

    def plot_3d_scatter_plotly(self, reduction_method='PCA'):
        """
        Plot a 3D scatter plot of the vectors using dimensionality reduction with Plotly Express.
        """
        vectors = [json.loads(vec_str) for _, _, _, vec_str in self.query_results if vec_str]
        vectors = np.array(vectors)
        reduced_vectors = self._reduce_dimensions(vectors, method=reduction_method)
        fig = px.scatter_3d(reduced_vectors, x=0, y=1, z=2,
                            title="3D Scatter Plot",
                            labels={'0': 'Reduced Dim 1', '1': 'Reduced Dim 2', '2': 'Reduced Dim 3'})
        fig.show()
