from flask import Flask, request, jsonify
import tiktoken  # Assuming this is a module you have for encoding
from InstructorEmbedding import INSTRUCTOR
import spacy
import numpy as np

"""
Helpers
"""
model = INSTRUCTOR('hkunlp/instructor-large')

"""
Endpoints
"""
app = Flask(__name__)


@app.route('/embeddings_api', methods=['POST'])
def embeddings_api():
    article_data = request.json.get('article')
    embed_dict = _build_embeddings(article_data, model)
    return jsonify(embed_dict)


def _build_embeddings(article: dict, model):
    enc = tiktoken.get_encoding("cl100k_base")
    instruction = "Represent the technical paragraph for retrieval:"
    embed_dict = {}
    for idx, (section, content) in enumerate(article.items()):
        encodings = enc.encode(content)
        embeddings = model.encode([[instruction, content]])
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        if isinstance(encodings, np.ndarray):
            encodings = encodings.tolist()
        embed_dict[section] = (embeddings, encodings)
    return embed_dict


@app.route('/ner_api', methods=['POST'])
def ner_api():
    ner_model = spacy.load('/Users/owner/myles-personal-env/Projects/wikiSearch/src/models/model-best')
    data = request.get_json()
    text = data.get("text")
    doc = ner_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return {"entities": entities}


if __name__ == '__main__':
    app.run(debug=True, port=5000)
