# Imports
from src.api import WikipediaAPI
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import (rds_host, rds_dbname, rds_user, rds_password, rds_port)
from src.relational import ArticlesTable
import json


"""
This is the file to develop a custom ner model to attach to our model_apis in order to build metadata
for the entire corpus.

Noetica AI is hiring a NLP Engineer - https://www.useparallel.com/noeticaai/careers/657741ab0222ded56f973340
CTO's thesis can actually be used here - https://academiccommons.columbia.edu/doi/10.7916/znbv-rz34
See Chapter 3: Partially Supervised Named Entity Recognition via the Expected Entity Ratio
"""


# Helpers:
def dataframe_to_spacy_format(df):
    training_data = []
    for _, row in df.iterrows():
        entities = {'entities': row['label']}
        training_data.append((row['text'], entities))
    return training_data


"""
Precprocess is to build a dataset from our articles table that matches the required format
"""
PREPROCESS = True
if PREPROCESS:
    emb_df = ArticlesTable(rds_dbname, rds_user, rds_password, rds_host, rds_port)
    art_df = emb_df.get_all_data_pd()
    spacy_data = dataframe_to_spacy_format(art_df)
    print(spacy_data[0])