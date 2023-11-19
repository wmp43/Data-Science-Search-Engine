# File for the massive for loop that is comprehensive data ingestion from wikipedia
# Implementations of TextProcessingStrategy can be CleanAndTokenizeStrategy, etc.
# Much of this algo is current garbage.
from classes import WikipediaAPI, Category, Article, BaseTextProcessor, TextProcessor



# I wish there was a dynamic way to visualize this working, like

list_of_categories = ['machine_learning', 'data_science', 'statistics', 'databases', 'computer_science',
                      'artificial_intelligence', 'bayes_thereom', 'game_theory', 'econometrics',
                      'big_data']  # Categories to be iterated through
set_of_categories = set()
set_of_articles = set()
while list_of_categories:
    # Category(title, id, returned_list)
    # def check_status(self, id)
    #   if id in db: return False

    category = list_of_categories.pop(1)
    if category in set_of_categories: pass
    set_of_categories.add(category)
    api_results = WikipediaAPI.category_request(category)  # Build list of tuples from cateogry search: [(category, namespace, id, title)]
    # api_results = [(category, namespace, id, title), (category, namespace, id, title), (category, namespace, id, title), (category, namespace, id, title)]

    (category, namespace, id, title)
    for result in api_results:
        if namespace == 14:  # if the result object is a category handle the category
            # Currently this category object is completely useuless. Have to think about how we are going to store super-categories or not
            # This could be useful for our category dimension for a relational db
            super_category, namespace, id, current_category = result[0], result[1], result[2], result[3]
            category_handler = Category(super_category, namespace, id, category)
            if category_handler.title in set_of_categories:
                pass
            else:
                list_of_categories.append(category_handler.title)
        else:
            category, namespace, id, title = result[0], result[1], result[2], result[3]
            article_handler = Article(category, title, id, None, text_processor)
            if article_handler.title in set_of_articles:  # if we already found this article in different category,
                # we want to add to the category metadata of the vecdb
                article_handler.update_categories(category)
            else:
                set_of_articles.add(article_handler.title)
                raw_text = WikipediaAPI.get_article_data(article_handler.title)
                article_handler.update_text(raw_text)
                summary, vector_embedding = article_handler.process_text_pipeline(
                    BaseTextProcessor())  # This pipeline is from raw to embedding
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