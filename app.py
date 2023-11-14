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


if __name__ == '__main__':
    app.run()
