import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.rds_crud import ArticleTable
from config import rds_port, rds_host, rds_user, rds_password, rds_dbname
import ast
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS
from itertools import cycle


"""
Helper funcs
"""
def convert_string_to_array(string):
    try:
        return ast.literal_eval(string)
    except ValueError:
        return np.nan


"""
Loading Data
"""

rds_args = (rds_dbname, rds_user, rds_password, rds_host, rds_port)
art_tbl = ArticleTable(*rds_args)
df = art_tbl.get_all_data_pd()


df['clustering_vec'] = df['clustering_vec'].apply(convert_string_to_array)
df['clustering_vec'] = df['clustering_vec'].apply(lambda x: np.array(x) if isinstance(x, list) else x)
vectors = np.stack(df['clustering_vec'].values)


"""
All these baselines were created after dim reduc.

DBSCAN
Arguments and results 
eps = mean_distance * 0.1 || min_samples = 5
- Good Baseline

eps = mean_distance * 1.2 || min_samples = 10
- Also Good

HDBSCAN
- Default Args
"""

DB_SCAN = False
HDB_SCAN = True
OPTIC = False
if DB_SCAN:
    # eps=mean_distance*0.11, min_samples=10
    db = DBSCAN(eps=0.5, min_samples=10).fit(vectors)
elif HDB_SCAN:
    # cluster_selection_epsilon=0.10,  min_cluster_size=10 -- this was post dim reduc
    db = HDBSCAN(cluster_selection_epsilon=0.05, min_cluster_size=5, min_samples=15).fit(vectors)
elif OPTIC:
    db = OPTICS().fit(vectors)

labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters (excluding noise): %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

"""
Dim Reduc and Viz

- TSNE has been alright
- Considering Umap

The function of this script is to allow for an interactive and visual clustering experience
Select your clustering Algo || Select Paramters || Select Dim Reduc Method
|/\\/|/\\/|/\\/|/\\/
Graph

|/\\/|/\\/|/\\/|/\\/
Suggested paramters: DBSCAN || eps = mean_distance * 1.2, min_samples = 10 || TSNE


"""

dim_reduc = TSNE(n_components=3, random_state=42)
res = dim_reduc.fit_transform(vectors)
df['dim_1'] = res[:, 0]
df['dim_2'] = res[:, 1]
df['dim_3'] = res[:, 2]
reduced_dims = df[['dim_1', 'dim_2', 'dim_3']]
data = reduced_dims.values

PLOT_NOISE = True
if not PLOT_NOISE:
    df_clustered = df[labels == -1]
    labels_filtered = labels[labels == -1]
    n_clusters_ = len(set(labels_filtered))
    palette = cycle(px.colors.qualitative.Plotly)
    colors = [next(palette) for _ in range(n_clusters_)]
    fig = px.scatter_3d(df_clustered, x='dim_1', y='dim_2', z='dim_3', hover_name='title', color=labels_filtered, color_discrete_sequence=colors)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()
else:
    palette = cycle(px.colors.qualitative.Plotly)
    colors = [next(palette) for _ in range(n_clusters_)] + ['grey']
    fig = px.scatter_3d(df, x='dim_1', y='dim_2', z='dim_3', hover_name='title', color=labels, color_discrete_sequence=colors)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()
