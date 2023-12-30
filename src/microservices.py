from flask import Flask, request, jsonify
import tiktoken  # Assuming this is a module you have for encoding
from models import Article
from InstructorEmbedding import INSTRUCTOR


app = Flask(__name__)

model = INSTRUCTOR('hkunlp/instructor-large')


@app.route('/build_embeddings', methods=['POST'])
def build_embeddings():
    article_data = request.json.get('article')
    print(article_data)
    article = Article.json_deserialize(article_data)
    embed_dict = _build_embeddings(article, model)
    return jsonify(embed_dict)


def _build_embeddings(article, model):
    enc = tiktoken.get_encoding("cl100k_base")
    instruction = "Represent the technical paragraph for retrieval:"
    embed_dict = {}
    for idx, (section, content) in enumerate(article.text_dict.items()):
        encodings = enc.encode(content)
        embeddings = model.encode([[instruction, content]])
        embed_dict[section] = (embeddings, encodings)
    return embed_dict


if __name__ == '__main__':
    app.run(debug=True, port=5000)