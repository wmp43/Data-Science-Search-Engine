from flask import Flask, request, jsonify
import tiktoken  # Assuming this is a module you have for encoding
from src.models import Article
from InstructorEmbedding import INSTRUCTOR

"""
Helpers
"""
def _build_embeddings(article, model):
    enc = tiktoken.get_encoding("cl100k_base")
    instruction = "Represent the technical paragraph for retrieval:"
    embed_dict = {}
    for idx, (section, content) in enumerate(article.text_dict.items()):
        encodings = enc.encode(content)
        embeddings = model.encode([[instruction, content]])
        embed_dict[section] = (embeddings, encodings)
    return embed_dict


model = INSTRUCTOR('hkunlp/instructor-large')

"""
Endpoints
"""
app = Flask(__name__)

@app.route('/hkunlp_embeddings_api', methods=['POST'])
def build_embeddings_api():
    article_data = request.json.get('article')
    article = Article.json_deserialize(article_data)
    embed_dict = _build_embeddings(article, model)
    return embed_dict


@app.route('/ner_api', methods=['POST'])
def build_embeddings_api():
    # todo: develop model and store json path
    model.load("SOMEJSONPATH")
    ner = model.predict("ARG_TEXT")
    return ner


if __name__ == '__main__':
    app.run(debug=True, port=5000)
