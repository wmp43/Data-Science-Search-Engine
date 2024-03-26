from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import requests
import cohere
from config import COHERE_API_KEY
import plotly.express as px
from sklearn.manifold import TSNE
import json
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import numpy as np


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
        prompt = 'you are a data science instructor, write a passage to answer the question'
        HyDE: which is query -> prompt + query to lm -> res
        :return: response from lm
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
        payload, api_url = {'query': query}, "http://127.0.0.1:5010/query_api"
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            embedded_q = response.json()
            return embedded_q
        else:
            print("Error in Query embed call:", response.status_code, response.text)

    def rerank(self, query, input_results):
        """
        :param: results -- needs to include text. The basic implementation can be found here:
        https://txt.cohere.com/rerank/
        :return:
        """
        # Results should be a list of text
        # List as dtype string has issues with rerank. Should just build new column 'text' in
        documents = [text for _, text, _, _ in input_results]
        co = cohere.Client(COHERE_API_KEY)
        results = co.rerank(query=query, documents=documents, top_n=10, model="rerank-multilingual-v2.0")
        # todo: return these results to table on frontend
        return results

    def call_language_model(self, context_len=2) -> str:
        """todo: ensure this was correct placement of langauge model call"""
        #  Pass context length of returned articles and query or expanded_query depending




class QueryVisualizer:
    def __init__(self):
        self.graph = nx.Graph()
        self.color_map = {
            'Probability & Statistics': 'green',
            'Machine Learning': 'blue',
            'Mathematics': 'red',
            'Programming': 'orange',
            'Data': 'purple',
            'People': 'gray',
            'Organizations': 'purple',
            'Professional Disciplines': 'pink',
            'Academic Disciplines': 'black'}

    """
    Network Viz
    """
    # def _process_metadata(self, query_results):
    #     for title, _, metadata_str, _ in query_results:  # Unpacking four values
    #         if isinstance(metadata_str, dict):
    #             metadata = metadata_str
    #         elif isinstance(metadata_str, str):
    #             try:
    #                 metadata = json.loads(metadata_str)
    #             except json.JSONDecodeError: continue
    #         else: continue
    #
    #         for key, values in metadata.items():
    #             for value in values:
    #                 self.graph.add_node(title)
    #                 self.graph.add_node(value)
    #                 self.graph.add_edge(title, value)

    def _process_metadata(self, query_results):
        """Process metadata and add edges to the graph."""
        for title, _, metadata_str, _ in query_results:
            if isinstance(metadata_str, str):
                try: metadata = json.loads(metadata_str)
                except json.JSONDecodeError: continue
            elif isinstance(metadata_str, dict):
                metadata = metadata_str
            else: continue  # Skip if metadata_str is neither dict nor str
            self.graph.add_node(title, type='article')

            # Iterate through each metadata category and add nodes and edges
            for category, items in metadata.items():
                for item in items:
                    self.graph.add_node(item, type=category)
                    self.graph.add_edge(title, item)

    def plot_graph(self, query_results):
        """Plot the graph using NetworkX with colored nodes and adjusted sizes."""
        self._process_metadata(query_results)  # Process metadata and construct the graph

        # Determine the color for each node based on its type
        colors = [self.color_map.get(self.graph.nodes[node]['type'], 'grey') for node in self.graph.nodes()]
        node_sizes = [self.graph.degree[node] * 100 for node in self.graph.nodes()]
        plt.figure(figsize=(15, 12))
        pos = nx.spring_layout(self.graph, k=0.30)
        nx.draw_networkx_nodes(self.graph, pos, node_color=colors, node_size=node_sizes)
        nx.draw_networkx_edges(self.graph, pos)
        nx.draw_networkx_labels(self.graph, pos)
        legend_handles = [mpatches.Patch(color=color, label=label) for label, color in self.color_map.items()]
        plt.legend(handles=legend_handles, loc='upper left')
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


