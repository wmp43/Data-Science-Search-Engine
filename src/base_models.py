# src.base_classes.py
import numpy as np
from dataclasses import dataclass, field
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from pydantic import BaseModel




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


test_pattern = [
    {"label": "Probability & Statistics", "pattern": "normal distribution"},
    {"label": "Probability & Statistics", "pattern": "poisson distribution"},
    {"label": "Probability & Statistics", "pattern": "estimates of location"},
    {"label": "Probability & Statistics", "pattern": "estimates of variability"},
    {"label": "Probability & Statistics", "pattern": "confidence intervals"},
    {"label": "Probability & Statistics", "pattern": "central limit theorem"},
    {"label": "Probability & Statistics", "pattern": "bayes theorem"},
    {"label": "Probability & Statistics", "pattern": "conditional probability"},
    {"label": "Probability & Statistics", "pattern": "correlation"},
    {"label": "Probability & Statistics", "pattern": "inference"},
    {"label": "Probability & Statistics", "pattern": "selection bias"},
    {"label": "Probability & Statistics", "pattern": "standard error"},
    {"label": "Probability & Statistics", "pattern": "bootstrap"},
    {"label": "Probability & Statistics", "pattern": "confidence intervals"},
    {"label": "Probability & Statistics", "pattern": "t distribution"},
    {"label": "Probability & Statistics", "pattern": "binomial distribution"},
    {"label": "Probability & Statistics", "pattern": "chi-square distribution"},
    {"label": "Probability & Statistics", "pattern": "f distribution"},
    {"label": "Probability & Statistics", "pattern": "exponential distribution"},
    {"label": "Probability & Statistics", "pattern": "probability distribution"},
    {"label": "Probability & Statistics", "pattern": "probability mass function"},
    {"label": "Probability & Statistics", "pattern": "cumulative density function"},
    {"label": "Probability & Statistics", "pattern": "probability density function"},
    {"label": "Probability & Statistics", "pattern": "hypothesis testing"},
    {"label": "Probability & Statistics", "pattern": "experimentation"},
    {"label": "Probability & Statistics", "pattern": "a/b testing"},
    {"label": "Probability & Statistics", "pattern": "p value"},
    {"label": "Probability & Statistics", "pattern": "anova"},
    {"label": "Probability & Statistics", "pattern": "regression analysis"},
    {"label": "Probability & Statistics", "pattern": "time series analysis"},
    {"label": "Probability & Statistics", "pattern": "markov models"},
    {"label": "Probability & Statistics", "pattern": "monte carlo simulation"},
    {"label": "Probability & Statistics", "pattern": "factor analysis"},
    {"label": "Probability & Statistics", "pattern": "causal inference"},
    {"label": "Probability & Statistics", "pattern": "probability space"},
    {"label": "Probability & Statistics", "pattern": "survival analysis"},
    {"label": "Probability & Statistics", "pattern": "logistic regression"},
    {"label": "Probability & Statistics", "pattern": "cluster analysis"},
    {"label": "Probability & Statistics", "pattern": "dimensionality reduction"},
    {"label": "Probability & Statistics", "pattern": "p-value"},
    {"label": "Probability & Statistics", "pattern": "z-test"},
    {"label": "Probability & Statistics", "pattern": "data normalization"},
    {"label": "Probability & Statistics", "pattern": "feature scaling"},
    {"label": "Probability & Statistics", "pattern": "data imputation"},
    {"label": "Probability & Statistics", "pattern": "sampling methods"},
    {"label": "Probability & Statistics", "pattern": "statistical significance"},
    {"label": "Probability & Statistics", "pattern": "descriptive statistics"},
    {"label": "Probability & Statistics", "pattern": "inferential statistics"},
    {"label": "Probability & Statistics", "pattern": "random variables"},
    {"label": "Probability & Statistics", "pattern": "sampling distributions"},
    {"label": "Probability & Statistics", "pattern": "point estimation"},
    {"label": "Probability & Statistics", "pattern": "variance analysis"},
    {"label": "Probability & Statistics", "pattern": "covariance analysis"},
    {"label": "Probability & Statistics", "pattern": "statistical power"},
    {"label": "Probability & Statistics", "pattern": "likelihood functions"},
    {"label": "Probability & Statistics", "pattern": "random variables"},
    {"label": "Machine Learning", "pattern": "supervised learning"},
    {"label": "Machine Learning", "pattern": "regression"},
    {"label": "Machine Learning", "pattern": "bayesian network"},
    {"label": "Machine Learning", "pattern": "linear regression"},
    {"label": "Machine Learning", "pattern": "directed acyclic graph"},
    {"label": "Machine Learning", "pattern": "classification"},
    {"label": "Machine Learning", "pattern": "unsupervised learning"},
    {"label": "Machine Learning", "pattern": "reinforcement learning"},
    {"label": "Machine Learning", "pattern": "neural networks"},
    {"label": "Machine Learning", "pattern": "deep learning"},
    {"label": "Machine Learning", "pattern": "convolutional neural networks"},
    {"label": "Machine Learning", "pattern": "recurrent neural networks"},
    {"label": "Machine Learning", "pattern": "long short-term memory networks"},
    {"label": "Machine Learning", "pattern": "transfer learning"},
    {"label": "Machine Learning", "pattern": "feature extraction"},
    {"label": "Machine Learning", "pattern": "feature selection"},
    {"label": "Machine Learning", "pattern": "decision tree"},
    {"label": "Machine Learning", "pattern": "random forest"},
    {"label": "Machine Learning", "pattern": "support-vector machine"},
    {"label": "Machine Learning", "pattern": "gradient boosting"},
    {"label": "Machine Learning", "pattern": "adaptive boosting"},
    {"label": "Machine Learning", "pattern": "ensemble methods"},
    {"label": "Machine Learning", "pattern": "k-means clustering"},
    {"label": "Machine Learning", "pattern": "hierarchical clustering"},
    {"label": "Machine Learning", "pattern": "dimensionality reduction"},
    {"label": "Machine Learning", "pattern": "principal component analysis"},
    {"label": "Machine Learning", "pattern": "singular value decomposition"},
    {"label": "Machine Learning", "pattern": "t-distributed stochastic neighbor embedding"},
    {"label": "Machine Learning", "pattern": "natural language processing"},
    {"label": "Machine Learning", "pattern": "text mining"},
    {"label": "Machine Learning", "pattern": "perceptron"},
    {"label": "Machine Learning", "pattern": "word embeddings"},
    {"label": "Machine Learning", "pattern": "bag of words"},
    {"label": "Machine Learning", "pattern": "tf-idf"},
    {"label": "Machine Learning", "pattern": "computer vision"},
    {"label": "Machine Learning", "pattern": "image recognition"},
    {"label": "Machine Learning", "pattern": "object detection"},
    {"label": "Machine Learning", "pattern": "anomaly detection"},
    {"label": "Machine Learning", "pattern": "predictive modeling"},
    {"label": "Machine Learning", "pattern": "cross-validation"},
    {"label": "Machine Learning", "pattern": "model evaluation"},
    {"label": "Machine Learning", "pattern": "model selection"},
    {"label": "Machine Learning", "pattern": "hyperparameter tuning"},
    {"label": "Machine Learning", "pattern": "overfitting"},
    {"label": "Machine Learning", "pattern": "underfitting"},
    {"label": "Machine Learning", "pattern": "bias-variance tradeoff"},
    {"label": "Machine Learning", "pattern": "data augmentation"},
    {"label": "Machine Learning", "pattern": "data preprocessing"},
    {"label": "Machine Learning", "pattern": "data splitting"},
    {"label": "Machine Learning", "pattern": "performance metrics"},
    {"label": "Machine Learning", "pattern": "accuracy"},
    {"label": "Machine Learning", "pattern": "precision"},
    {"label": "Machine Learning", "pattern": "recall"},
    {"label": "Machine Learning", "pattern": "f1 score"},
    {"label": "Machine Learning", "pattern": "confusion matrix"},
    {"label": "Machine Learning", "pattern": "ROC curve"},
    {"label": "Machine Learning", "pattern": "AUC score"},
    {"label": "Machine Learning", "pattern": "total operating characteristic"},
    {"label": "Machine Learning", "pattern": "receiver operating characteristic"},
    {"label": "Machine Learning", "pattern": "area under the curve"},
    {"label": "Mathematics", "pattern": "stochastic process"},
    {"label": "Mathematics", "pattern": "calculus"},
    {"label": "Mathematics", "pattern": "linear algebra"},
    {"label": "Mathematics", "pattern": "matrix"},
    {"label": "Mathematics", "pattern": "vector"},
    {"label": "Mathematics", "pattern": "markov chains"},
    {"label": "Mathematics", "pattern": "gradient descent"},
    {"label": "Mathematics", "pattern": "matrix multiplication"},
    {"label": "Mathematics", "pattern": "derivative"},
    {"label": "Mathematics", "pattern": "integral"},
    {"label": "Mathematics", "pattern": "taylor series"},
    {"label": "Mathematics", "pattern": "random walk"},
    {"label": "Mathematics", "pattern": "wiener process"},
    {"label": "Mathematics", "pattern": "poisson process"},
    {"label": "Mathematics", "pattern": "fourier transform"},
    {"label": "Mathematics", "pattern": "laplace transform"},
    {"label": "Mathematics", "pattern": "differential equations"},
    {"label": "Mathematics", "pattern": "bayesian inference"},
    {"label": "Mathematics", "pattern": "probability theory"},
    {"label": "Mathematics", "pattern": "statistical modeling"},
    {"label": "Mathematics", "pattern": "optimization theory"},
    {"label": "Mathematics", "pattern": "discrete mathematics"},
    {"label": "Mathematics", "pattern": "combinatorics"},
    {"label": "Mathematics", "pattern": "graph theory"},
    {"label": "Mathematics", "pattern": "numerical methods"},
    {"label": "Mathematics", "pattern": "game theory"},
    {"label": "Mathematics", "pattern": "eigenvalues and eigenvectors"},
    {"label": "Mathematics", "pattern": "convex optimization"},
    {"label": "Mathematics", "pattern": "topology"},
    {"label": "Mathematics", "pattern": "real analysis"},
    {"label": "Mathematics", "pattern": "complex analysis"},
    {"label": "Mathematics", "pattern": "number theory"},
    {"label": "Mathematics", "pattern": "set theory"},
    {"label": "Mathematics", "pattern": "information theory"},
    {"label": "Mathematics", "pattern": "chaos theory"},
    {"label": "Mathematics", "pattern": "nonlinear dynamics"},
    {"label": "People", "pattern": "machine learning engineers"},
    {"label": "People", "pattern": "researchers"}
]


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
        "encoding": np.ndarray # built from the text dict. Can use tiktoken.encode(text)
        "id": 1234,
        "title": "title of article",
        "metadata": {"section heading":{
                "Probability & Statistics": ['bayesian_statistics', 'posterior_probability', 'bayes_estimation'],
                "Machine Learning": ['overfitting'],
                "Mathematics": ['matrix multiplication'],
                "People": ['machine learning engineer'],
                "Categories":['Category_1', 'Category_2']
                }
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
    metadata_dict: Dict[str, Dict[str, any]] = field(default_factory=dict)
    text_processor: any = None

    # def print_attribute_types(self):
    #     for attribute, value in self.__dict__.items():
    #         print(f"{attribute} ({type(value)})")
    #         if isinstance(value, dict):
    #             for key, val in value.items():
    #                 print(f"  {key}: {type(val)}")
    #                 if isinstance(val, dict):  # For nested dictionaries
    #                     for sub_key, sub_val in val.items():
    #                         print(f"    {sub_key}: {type(sub_val)}")

    def process_text_pipeline(self, text_processor, exclude_section):
        text_dict = text_processor.build_section_dict(self, exclude_section)
        clean_text_dict = text_processor.remove_curly_brackets(text_dict)
        chunked_text_dict = text_processor.chunk_text_dict(clean_text_dict)
        self.text_dict = chunked_text_dict
        return self

    def process_embedding_pipeline(self, text_processor):
        """
        This method may only be invoked after the process_text_pipeline method.
        This will return a dictionary with section headings and token lens for
        Cost approximation and text chunking
        :param text_processor: BaseTextProcessor Class
        :return: self
        """
        embed_dict = text_processor.build_embeddings(self)
        self.embedding_dict = embed_dict
        return self

    def process_metadata_pipeline(self, text_processor):
        """
        December 19th
        https://spacy.io/usage/rule-based-matching
        This method should build a metadata object for each embedded section.
        By extension I guess it should also support
        :param text_processor: TextProcessor object to build metadata
        :return: self
        """
        metadata_dict = text_processor.build_metadata(self, pattern=test_pattern)
        self.metadata_dict = metadata_dict

    def process_metadata_labeling(self, text_processor):
        json_record = text_processor.build_metadata_json(self, test_pattern)
        return json_record

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


class Query(BaseModel):
    """

    """
    text: str

    def encode_query(self):
        # embed the query
        # search the db for cosine_sim argmax
        pass
