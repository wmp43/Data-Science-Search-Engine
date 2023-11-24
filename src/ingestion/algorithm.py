# File for the massive for loop that is comprehensive data ingestion from wikipedia
# Implementations of TextProcessingStrategy can be CleanAndTokenizeStrategy, etc.
# Much of this algo is current garbage.
from classes import WikipediaAPI, Category, Article, BaseTextProcessor, TextProcessor
from collections import deque
import uuid
# I wish there was a dynamic way to visualize this working, like





while list_of_categories:

    CAT1, CAT2, CAT3 = Category("meta_category", uuid.uuid4(), "Statistics"), Category("meta_category", uuid.uuid4(), "Machine_Learning"), Category("meta_category", uuid.uuid4(), "Operation_research")
    called_articles, called_categories = set(), set()
    BASE_WIKI_URL = 'https://en.wikipedia.org/'
    list_of_categories = deque([CAT1, CAT2, CAT3])
    vec_db_client = VectorDB()

    category = list_of_categories.popleft()
    if category.title in called_categories: pass
    called_categories.add(category.title)
    api_results = WikipediaAPI().get_category_data(category.title)
    # api_results = [(category, namespace, id, title), (category, namespace, id, title),....]
    for result in api_results: #iterate through the result categories
        if namespace == 14:  # if the result item is a category
            super_category, namespace, id, title = result[0], result[1], result[2], result[3]
            if title in called_categories: pass # Sub category was found via multiple super categoryies,
            # todo: relational db for category realtionships
            else:
                category_handler = Category(super_category, title, uuid.uuid4())
                list_of_categories.append(category_handler)
        else:
            super_category, namespace, id, title = result[0], result[1], result[2], result[3]
            # Could be a cagtegory object as first param rather than just the name.
            # I think you would get access to the Articles's Category's Content. Article.cateogry.super_categories
            article_handler = Article(category, title, id, "", None, f'{BASE_WIKI_URL}/{title}', "", BaseTextProcessor())
            if article_handler.title in called_articles:  # if we already found this article in different category,
                # we want to add to the category metadata of the vecdb
                article_handler.update_categories(article_handler.category)
                """
                These article methods must work across chunks in the db. A lack of breadth failure would chunk the app
                """
            else:
                called_articles.add(article_handler.title)
                raw_text = WikipediaAPI.get_article_data(article_handler.title)

                article_handler.update_text(raw_text)
                # article_handler.chunk_text(raw_text)
                # Would take raw text and break it to proper embedding length
                # this method would also remove sections we are not interested in storing
                summary, vector_embedding = article_handler.process_text_pipeline(BaseTextProcessor())  # This pipeline is from raw to embedding
                article_handler.update_record(summary, vector_embedding)

            # Should add functionality for when, there are multiple categories

    #     if category_handler.id in set_of_articles: # If the article is in db, we want to add category
    #         vec_db_record = VectorDB.get(article_id) #think oj json above
    #         if category not in vec_db_record['metadata']['categories']:
    #             vec_db_record['metadata']['categories'].append(category)
    #             vec_db_record.update()
    #     else:
    #
    #         set_of_articles.add((article_id, article_title))
    #
    # else: #Handle the article
    #     article = Article(result[3], result[2], None)  # Initialize Article without text
    #     article_content = WikipediaAPI.get_article_data(article.title)  # Assume this returns the content of the article
    #     article.update_text(article_content)

    #
    #     else: pass
    #     else:
    #         # Get text if not in db
    #         article_text = article_api_call(article_title)
    #         clean_text = article_text.clean().tokenize()
    #         # Build metadata: get category, find mentions
    #         metadata = clean_text.build_metadata()
    #         embedding = clean_text.vectorize()
    #         db.add(article_id, {"embedding": embedding, "metadata": metadata, "title": article_title})
    # elif result[1] == 'category':
    #     # If the result is a category, add to list for further processing
    #     new_category = result[2]
    #     if new_category not in set_of_categories:
    #         list_of_categories.append(new_category)