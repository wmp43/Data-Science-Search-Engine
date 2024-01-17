from flask import Flask, render_template, request, jsonify
from src.base_models import Query
from src.query_processor import BaseQueryProcessor, QueryVisualizer
from src.rds_crud import VectorTable
from config import rds_user, rds_password, rds_host, rds_port, rds_dbname

app = Flask(__name__)
rds_args = (rds_dbname, rds_user, rds_password, rds_host, rds_port)
vec_tbl = VectorTable(*rds_args)


"""
search_term = request.form['search_term']
qp, qv = BaseQueryProcessor(), QueryVisualizer()
query_obj = Query(search_term, qp, qv, vec_tbl, embedding=[], results=[])
This Query Object can also be used for query visualization

A Corpus visualization will be different.
"""

@app.route('/search', methods=['POST'])
def search():
    # Should go from query result to table and LLM response.
    search_term = request.form['search_term']
    qp, qv = BaseQueryProcessor(), QueryVisualizer()
    query_obj = Query(search_term, qp, qv, vec_tbl, embedding=[], results=[])
    query_obj.process()
    query_obj.execute()




@app.route('/visualize')
def hello_world():  # put application's code here
    # Flask Endpoint For Wiki Search text input
    return 'Hello World!'


if __name__ == '__main__':
    app.run()
