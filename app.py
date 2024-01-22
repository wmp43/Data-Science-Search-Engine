from flask import Flask, render_template, request, jsonify
from src.base_models import Query
from src.query_processor import BaseQueryProcessor, QueryVisualizer
from src.rds_crud import VectorTable, ArticleTable, QueryTable
from config import rds_user, rds_password, rds_host, rds_port, rds_dbname
import logging
from InstructorEmbedding import INSTRUCTOR
import traceback
import numpy as np

app = Flask(__name__)
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
rds_args = (rds_dbname, rds_user, rds_password, rds_host, rds_port)
vec_tbl, art_tbl, qry_tbl = VectorTable(*rds_args), ArticleTable(*rds_args), QueryTable(*rds_args)
model = INSTRUCTOR('hkunlp/instructor-large')


@app.route('/')
def index():
    app.logger.info('Main page requested')
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    data = request.json
    search_term = data['query']
    if not search_term: return "Search Term is Required", 400
    app.logger.info(f'Search requested for term: {search_term}')
    qp, qv = BaseQueryProcessor(), QueryVisualizer()
    """
    QueryTable insert
    """
    query_obj = Query(search_term, qp, qv, vec_tbl, qry_tbl, embedding=[], results=[])
    query_obj.process()
    query_obj.execute()
    query_obj.query_to_tbl()
    # todo: return LLM Response not just table of res
    return jsonify(query_obj.results)


# @app.route('/api/visualize/query')
# def visualize_query():
#     # Endpoint for query visualization
#     # Fetch and process data for query visualization
#     # Return the data in a suitable format for d3.js
#     pass  # Replace with your implementation
#
#
# @app.route('/api/visualize/article')
# def visualize_article():
#     # Endpoint for article visualization
#     # Fetch and process data for article visualization
#     pass  # Replace with your implementation
#
#
# @app.route('/api/visualize/corpus')
# def visualize_corpus():
#     # Endpoint for corpus visualization
#     pass  # Replace with your implementation

"""
This should be in inference endpoints but flask isn't allowing parallel serving
"""


@app.route('/query_api', methods=['POST'])
def query_api():
    query = request.json.get('query')
    cluster_arr = _build_query(query, model)
    if isinstance(cluster_arr, np.ndarray):
        cluster_arr = cluster_arr.tolist()
        return jsonify(cluster_arr)
    else:
        return jsonify({'error': 'Failed to process clustering'}), 500


def _build_query(query, model):
    try:
        instruction = "Represent the question for retrieving supporting documents: "
        embeddings = model.encode([[instruction, query]])
        return embeddings
    except Exception as e:
        print(f'Query Error (_build_query): {e}')
        traceback.print_exc()
        return None


if __name__ == '__main__':
    app.run()
