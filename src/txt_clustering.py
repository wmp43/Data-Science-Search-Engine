import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.rds_crud import ArticleTable
from config import rds_port, rds_host, rds_user, rds_password, rds_dbname
from sklearn.decomposition import PCA
import ast
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
import plotly.express as px

"""
Helper funcs
"""
def convert_string_to_array(string):
    try:
        return ast.literal_eval(string)
    except ValueError:
        return np.nan  # or return None, or handle it in some other way

"""
Loading Data
"""
rds_args = (rds_dbname, rds_user, rds_password, rds_host, rds_port)

art_tbl = ArticleTable(*rds_args)
df = art_tbl.get_all_data_pd()
df['clustering_vec'] = df['clustering_vec'].apply(convert_string_to_array)
df['clustering_vec'] = df['clustering_vec'].apply(lambda x: np.array(x) if isinstance(x, list) else x)
vectors = np.stack(df['clustering_vec'].values)
dim_reduc = MDS(n_components=3, random_state=42)
res = dim_reduc.fit_transform(vectors)
print(res.shape)
df['dim_1'] = res[:, 0]
print(f'dim_1 avg: {df["dim_1"].mean()}')
df['dim_2'] = res[:, 1]
print(f'dim_2 avg: {df["dim_2"].mean()}')
df['dim_3'] = res[:, 2]
print(f'dim_3: {df["dim_3"].mean()}')
fig = px.scatter_3d(df, x='dim_1', y='dim_2', z='dim_3', hover_name='title')
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()




# tsne = TSNE(n_components=3, random_state=42)
# tsne_results = tsne.fit_transform(list(df['clustering_vec']))
# tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2', 'TSNE3'])
# tsne_df['title'] = df['title']
#
#

