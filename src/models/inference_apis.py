from flask import Flask, request, jsonify
import tiktoken  # Assuming this is a module you have for encoding
from InstructorEmbedding import INSTRUCTOR
import spacy
import traceback
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
    """
    Could Also consider elastic search for DPR. This is likely the more professional option
    :return:  Embedding Dict
    """
    article_data = request.json.get('article')
    embed_dict = _build_embeddings(article_data, model)
    return jsonify(embed_dict)


def _build_embeddings(article: dict, model):
    enc = tiktoken.get_encoding("cl100k_base")
    instruction = "Represent the technical paragraph for retrieval: "
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
    article_data = request.json.get('article')
    ner_model = spacy.load('/Users/owner/myles-personal-env/Projects/wikiSearch/src/models/model-best')
    metadata_dict = _build_ner(article_data, ner_model)
    return jsonify(metadata_dict)


def _build_ner(article_text_dict: dict, ner_model):
    ner_dict = {}
    for section, content in article_text_dict.items():
        doc, meta_dict = ner_model(content), {}
        for ent in doc.ents:
            label, text = ent.label_, ent.text
            if label not in meta_dict: meta_dict[label] = []  # init
            if text not in meta_dict[label]: meta_dict[label].append(text)  # append
        ner_dict[section] = meta_dict
    return ner_dict


@app.route('/clustering_api', methods=['POST'])
def clustering_api():
    article_txt = request.json.get('text')
    cluster_arr = _build_clusters(article_txt, model)
    if isinstance(cluster_arr, np.ndarray):
        cluster_arr = cluster_arr.tolist()
        return jsonify(cluster_arr)
    else:
        return jsonify({'error': 'Failed to process clustering'}), 500


def _build_clusters(article_text, model):
    try:
        instruction = "Represent the technical document for clustering: "
        embeddings = model.encode([[instruction, article_text]])
        return embeddings
    except Exception as e:
        print(f'Clustering Error (_build_clusters): {e}')
        traceback.print_exc()  # This will print the full traceback
        return None


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
