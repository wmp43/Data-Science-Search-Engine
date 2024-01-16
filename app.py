from flask import Flask
from src.base_models import Query
from src.query_processor import BaseQueryProcessor
from src.rds_crud import VectorTable
from config import rds_user, rds_password, rds_host, rds_port, rds_dbname

app = Flask(__name__)
rds_args = (rds_dbname, rds_user, rds_password, rds_host, rds_port)

@app.route('/search', methods=['POST'])
def search():
    # user -> query -> expanded -> embedded -> results -> re ranked -> LLM Res
    # this may be super slow....
    bqp, vec_tbl = BaseQueryProcessor(), VectorTable(*rds_args)
    input = Query(query, bqp)
    expanded = input.expand_query().encode_query()



@app.route('/visualize')
def hello_world():  # put application's code here
    # Flask Endpoint For Wiki Search text input
    return 'Hello World!'


@app.route('/versioning')
def hello_world():  # put application's code here
    # Flask Endpoint for doc versioning, upload
    return 'Hello World!'

"""
Status: 
- Data Cleaning method is API call, split and clean, regex, return text
- This pipeline lives in the jupyter notebook text_cleaning_test_1 and sans 1. 
- Issues: need to figure out how to remove latex. Start with chunking at section headings
- then figure out db. thinking about pgvector due to open source. Need host that allows for extensions


"""


if __name__ == '__main__':
    app.run()
