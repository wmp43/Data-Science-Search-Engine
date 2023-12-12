from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    # Flask Endpoint For Wiki Search Home
    return 'Hello World!'


@app.route('/search')
def hello_world():  # put application's code here
    # Flask Endpoint For Wiki Search text input
    return 'Hello World!'


@app.route('/versioning')
def hello_world():  # put application's code here
    # Flask Endpoint for doc versioning, upload
    return 'Hello World!'


@app.route('/visualize')
def hello_world():  # put application's code here
    # Flask Endpoint For Wiki Search text input
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
