"""
This is a pipeline from Wikipedia API to vector embedding
Essentially the entire ingestion process to build vector db
"""
from src.text_processor import BaseTextProcessor
from src.base_models import Article
from src.api import WikipediaAPI
from spacy.lang.en import English
from config import test_pattern


"""
Outline
1. Random helpers to sensure the methods are working. Quick and dirty sub for legit tests 
Purpose
1. To Run through the process acquiring data, processing it, storing it



WIDTH OF ARTICLE SEARCH:
PROCESSED, NOT PROCESSED = [], [HUGE LIST OF RELEVANT ARTICLES]
WHILE NOT PROCESSED:
    IF ARTICLE IN SEEN_ARTICLES: CONTINUE
    ELSE: 
        INGESTION PIPELINE
        ADD SEE ALSO SECTION ARTICLES TO LIST
        
THIS COULD BE RELEVANT METHOD OF EXTRACTION. 
ALSO COULD CAP CONTENT: ONLY 2 MENTIONS AWAY FROM THE ROOT ARTICLE.
PREVENTS INFINITE SEARCH
"""

INGESTION = True


if INGESTION:
    TITLE = 'Machine_learning'
    SECTIONS_TO_IGNORE = [
        "See also", "References", "External links", "Further reading", "Footnotes", "Bibliography", "Sources",
        "Citations",
        "Literature", "Footnotes", "Notes and references", "Photo gallery", "Works cited", "Photos", "Gallery", "Notes",
        "References and sources", "References and notes"]

    wiki_api = WikipediaAPI()
    processor = BaseTextProcessor()

    title, page_id, final_text = wiki_api.fetch_article_data(TITLE)
    article = Article(title=TITLE, id=page_id, text=final_text, text_processor=processor)
    article.process_text_pipeline(processor, SECTIONS_TO_IGNORE)
    article.process_embedding_pipeline(processor)

    # Print the first two key-value pairs
    # for key, value in first_two_pairs:
    #     print(f"{key}: {value}")
    # for k, v in article.embedding_dict.items():
    #     print(f'embeddings: {type(v[0]), v[0].shape}')
    # print(len(article.text_dict["Introduction_1"]) + len(article.text_dict["Introduction_0"]) )
    # print(f'intro_0: {article.text_dict["Introduction_0"]}\n Intro_1: {article.text_dict["Introduction_1"]}')
    #ner(article1)

# def get_embedding(text, model="text-embedding-ada-002"):
#    text = text.replace("\n", " ")
#    return client.embeddings.create(input=[text], model=model).data[0].embedding
#
#
# for key, value in article.text_dict.items():
#     new_dict = {}
#     new_dict[key] = get_embedding(value)


# print(new_dict)

"""
1. metadata building function
2. text chunking function
3. text  embedding function
3. Need C.R.U.D operations for db
"""

# max_chars = 512
# text_splitter = CharacterTextSplitter(max_chars)
#
# chunked_text_dict = {}
#
# for section, content in article.text_dict.items():
#     chunks = text_splitter.split(content)
#     chunked_text_dict[section] = chunks

# import spacy
# from spacy.matcher import Matcher
# nlp = spacy.load("en_core_web_sm")
# matcher = Matcher(nlp.vocab)
# # Add match ID "HelloWorld" with no callback and one pattern
# pattern = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]
# matcher.add("HelloWorld", [pattern])
#
# doc = nlp("Hello, world! Hello world!")
# matches = matcher(doc)
# for match_id, start, end in matches:
#     string_id = nlp.vocab.strings[match_id]  # Get string representation
#     span = doc[start:end]  # The matched span
#     print(f'match_id: {match_id}\nstring id: {string_id}\nstart: {start}\nend: {end}\nspan: {span.text}')


# print(article.text)
# print(article.text_dict.keys(),'\n\n\n')
# print(article.text_dict['Introduction'])

# def have_same_keys_and_length(dict1, dict2):
#     # Check if the length of both dictionaries is the same
#     if len(dict1) != len(dict2):
#         return False
#
#     # Check if all keys in dict1 are in dict2
#     for key in dict1:
#         if key not in dict2:
#             return False
#
#     # Optionally, check if all keys in dict2 are in dict1 as well
#     for key in dict2:
#         if key not in dict1:
#             return False
#
#     return True


# def ner(article: Article):
#     metadata_dict = {}
#     nlp = English()
#     ruler = nlp.add_pipe("entity_ruler")
#     ruler.add_patterns(test_pattern)
#
#     for heading, content in article.text_dict.items():
#         sub_metadata_dict = {}
#         doc = nlp(content)
#         for ent in doc.ents:
#             if ent.label_ in sub_metadata_dict:
#                 sub_metadata_dict[ent.label_].add(ent.text)
#             else:
#                 sub_metadata_dict[ent.label_] = {ent.text}
#         # Convert sets to lists for final output
#         metadata_dict[heading] = {key: list(value) for key, value in sub_metadata_dict.items()}
#     article.metadata_dict = metadata_dict
#     print(metadata_dict)
#     return None