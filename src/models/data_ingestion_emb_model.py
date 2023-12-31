"""
This file will upsert data in a new rds table for developing the embedding model
"""

from src.models import Article
from src.api import WikipediaAPI
from src.relational import EmbeddingDB
from config import (rds_host, rds_dbname, rds_user, rds_password, rds_port)


"""
Order of Operations and other details
"""
