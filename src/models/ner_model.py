# Imports
from src.api import WikipediaAPI
from src.text_processor import BaseTextProcessor
import re
from src.base_models import Article
import spacy
from spacy import displacy

"""
This is the file to develop a custom ner model to attach to our model_apis in order to build metadata
for the entire corpus.

Noetica AI is hiring a NLP Engineer - https://www.useparallel.com/noeticaai/careers/657741ab0222ded56f973340
CTO's thesis can actually be used here - https://academiccommons.columbia.edu/doi/10.7916/znbv-rz34
See Chapter 3: Partially Supervised Named Entity Recognition via the Expected Entity Ratio
"""


"""
The method to fine-tune the model is in config1.cfg with model tuning params
Terminal command is:
python -m spacy train /Users/owner/myles-personal-env/Projects/wikiSearch/src/models/config1.cfg 
    --output /Users/owner/myles-personal-env/Projects/wikiSearch/src/models/. 
    --paths.train /Users/owner/myles-personal-env/Projects/wikiSearch/src/models/train.spacy 
    --paths.dev /Users/owner/myles-personal-env/Projects/wikiSearch/src/models/test.spacy 
    --gpu-id 0

"""

NER_TEST = True
SECTIONS_TO_IGNORE = ["References", "External links", "Further reading", "Footnotes", "Bibliography", "Sources",
                      "See also",
                      "Citations", "Literature", "Footnotes", "Notes and references", "Photo gallery", "Works cited",
                      "Photos", "Gallery",
                      "Notes", "References and sources", "References and notes"]



if NER_TEST:
    """
    Some notes from visualiztion on new Wikipedia Articles.
    - Current model is over-fit. It remembers exact strings from the config pattern.
    - It selects things like engineering as Academic disciplines, when the phrase is Information Engineering
    - Make more specific Academic tags away from just engineering
    - Sometimes it doesnt even select exact matches
    - Need to try some different configs, maybe early stopping too
    """
    wiki_api = WikipediaAPI()
    processor = BaseTextProcessor()
    WIKI = 'Cluster_analysis'
    title, page_id, final_text = wiki_api.fetch_article_data_by_title(WIKI)
    article = Article(title=title, id=page_id, text=final_text, text_processor=processor)
    article.process_text_pipeline(processor, SECTIONS_TO_IGNORE)
    concat_text = str([re.sub(r'[\n\t]+', ' ', text) for text in article.text_dict.values()])
    ner_model = spacy.load('//src/models/model-best')
    doc = ner_model(concat_text)
    doc.user_data["title"] = 'Wikipedia Article: ' + WIKI + ' w/ ner tags'
    colors = {
        "Probability & Statistics": "#FFA07A",
        "Machine Learning": "#20B2AA",
        "Mathematics": "#778899",
        "Data": "#9370DB",
        "Organizations": "#FFD700",
        "People": "#F08080",
        "Programming": "#00FA9A",
        "Academic Disciplines": "#4682B4"}

    displacy.serve(doc, style="ent", options={"colors": colors}, port=1111)

