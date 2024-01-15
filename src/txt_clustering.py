import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.rds_crud import ArticleTable
from config import rds_port, rds_host, rds_user, rds_password, rds_dbname
from sklearn.decomposition import PCA
import ast
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn import metrics



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
dim_reduc = TSNE(n_components=3, random_state=42)
res = dim_reduc.fit_transform(vectors)
print(res.shape)
df['dim_1'] = res[:, 0]
plt.hist(df['dim_1'])
plt.title('DIM 1 Hist')
print(f'dim_1 avg: {df["dim_1"].mean()}')
df['dim_2'] = res[:, 1]
plt.hist(df['dim_2'])
plt.title('DIM 2 Hist')
print(f'dim_2 avg: {df["dim_2"].mean()}')
df['dim_3'] = res[:, 2]
plt.hist(df['dim_3'])
plt.title('DIM 3 Hist')
print(f'dim_3: {df["dim_3"].mean()}')


"""
DBscan
"""

from scipy.spatial.distance import pdist



reduced_dims = df[['dim_1', 'dim_2', 'dim_3']]
data = reduced_dims.values  # Use values to get the correct numpy array shape
distances = pdist(data)
mean_distance = np.mean(distances)
print(f'Euc mean distance on reduced dim dimensions: {mean_distance}')

# Apply DBSCAN clustering
"""
Arguments and results 
eps = mean_distance * 0.1 || min_samples = 5
- Good Baseline

eps = mean_distance * 1.2 || min_samples = 10
- Also Good



"""
DB_SCAN = True
if DB_SCAN:
    db = DBSCAN(eps=mean_distance*0.10, min_samples=10).fit(reduced_dims)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

from itertools import cycle
palette = cycle(px.colors.qualitative.Plotly)
colors = [next(palette) for _ in range(n_clusters_)] + ['grey']
fig = px.scatter_3d(df, x='dim_1', y='dim_2', z='dim_3', hover_name='title', color=labels, color_discrete_sequence=colors)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
