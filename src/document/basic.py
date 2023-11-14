from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import spacy
import chromadb

chroma_client = chromadb.Client()


nlp = spacy.load("en_core_web_sm")


# Functionality for Query Object
class Query(BaseModel):
    raw_query: str
    processed_query: Optional[str] = None
    metadata: Optional[Dict[str, str]] = {}

    def __init__(self, **data):
        super().__init__(**data)
        self.process_query()

    def process_query(self):
        # Logic to process the raw query
        # For example, tokenizing, removing unnecessary parts, normalization, etc.
        self.processed_query = self.func(self.raw_query)

    def func(self):
        # Place holder for
        pass


# Functionality for document Object
class Document(BaseModel):
    id: str
    title: str
    content: str
    metadata: Optional[Dict[str, str]] = Field(default_factory=dict)
    tokens: Optional[List[str]] = None
    vector_embedding: Optional[List[float]] = None

    def __init__(self, **data):
        super().__init__(**data)

    """
    Section 1
    - Text processing
    def tokenize, def remove_html
    maybe: def lemmatization, def stem 
    """

    def tokenize(self):
        """
        Tokenizes the document's content using SpaCy and updates the tokens attribute.
        """
        doc = nlp(self.content)
        self.tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

    def remove_html(self):
        # remove html from content
        # may not even be needed depending on acquisition method
        pass

    def lemmatize(self):
        pass

    def stem(self):
        pass

    """"
    Section 2
    - Vecs
    def vec_embed, def classify_type, def extract_metadata, def ner
    """""

    def vec_embed(self):
        # Vectorization of tokens logic
        pass

    def extract_metadata(self):
        # Function for ner, clf for doc type
        # generate metadata. May need multiple functions for this
        pass

    def clf_document(self):
        # Classify documents to major groups
        pass

    """Section 3
    - Db interaction
    def save_to_db, def update_in_db, def delete_from_db, def retreive_from_db
    """

    def update_in_db(self):
        pass

    def delete_from_db(self):
        pass

    def save_to_db(self):
        # Sending vec & metadata to db
        pass

    def retreive_from_db(self):
        # Retrieve from db
        pass

    """
    Section 4
    - version Control
    def checkout def download, def upload
    """
