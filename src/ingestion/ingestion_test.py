# from classes import WikipediaAPI, Article, Classifier
# from collections import deque
# import time
# import matplotlib.pyplot as plt
#
# not_called_categories = deque(['Category:machine_learning', 'Category:data_science', 'Category:statistics',
#                                'Category:databases', 'Category:computer_science', 'Category:artificial_intelligence',
#                                'Category:game_theory', 'Category:econometrics', 'Category:big_data'])
# called_categories = set()
# called_articles = set()
#
# idx = 0
# article_counts_per_pass = []
# categories_remaining_per_pass = []
#
# while not_called_categories:
#
#     category = not_called_categories.popleft()
#     if category in called_categories:
#         continue
#     else:
#         clf = Classifier().build_and_predict(category)
#         if clf is not True: pass
#     called_categories.add(category)
#     print(f'Current Category: {category}\nNumber of Categories Traversed: {len(called_categories)}\n'
#           f'Number of Categories Left: {len(not_called_categories)}'
#           f'\nNumber of Articles Traversed: {len(called_articles)}')
#     api_results = WikipediaAPI().get_category_data(category)
#     for result in api_results:
#         title, namespace = result[1], result[3]
#         if namespace == 14:
#             if title not in called_categories:
#                 not_called_categories.append(title)
#         else:
#             if title not in called_articles:
#                 called_articles.add(title)
#
#     article_counts_per_pass.append(len(called_articles))
#     categories_remaining_per_pass.append(len(not_called_categories))
#     time.sleep(1)
#     idx += 1
#
# plt.figure(figsize=(10, 6))  # Optional: Set the size of the plot
# plt.plot(article_counts_per_pass, label='Number of Articles Processed', color='blue')
# plt.plot(categories_remaining_per_pass, label='Number of Categories Remaining', color='red')
#
# plt.xlabel('Iterations')
# plt.ylabel('Count')
# plt.title('Progress of Articles and Categories Over Iterations')
# plt.legend()
# plt.grid(True)
# plt.show()

"""
Pat 2 to build df for clf
"""
import uuid
from sklearn.decomposition import PCA
from src.models import WikipediaAPI, Category
from collections import deque
import time
import pandas as pd

MODEL_PATH = ("/Users/owner/myles-personal-env/Projects/"
              "wikiSearch/src/ingestion/category_classification/category_clf.json")

not_called_categories = deque(['Category:Operations Research', 'Category:Statistics', 'Category: Machine Learning'])
called_categories = set()
called_articles = set()

idx = 0
article_counts_per_pass = []
categories_remaining_per_pass = []

relevant_categories_df = pd.DataFrame(columns=['Referenced_from', 'Category'])
non_relevant_categories_df = pd.DataFrame(columns=['Referenced_from', 'Category'])

while not_called_categories:
    category_obj = Category(None, id=uuid.UUID, title=not_called_categories.popleft(), clean_title=None)
    category: str = category_obj.title
    if category in called_categories:
        continue
    called_categories.add(category)
    print(f'Current Category: {category}\nNumber of Categories Traversed: {len(called_categories)}\n'
          f'Number of Categories Left: {len(not_called_categories)}'
          f'\nNumber of Articles Traversed: {len(called_articles)}')

    api_results = WikipediaAPI().get_category_data(category)
    category_obj.build_optionals(api_results)
    for result in category_obj.return_items:
        title: str = result[1]
        namespace = result[3]
        if namespace == 14 and title not in called_categories:
            res_cat = Category(super_category=category, id=uuid.UUID, title=title, clean_title=None)
            res_cat.clean_title_method()
            title_features = res_cat.build_title_embeddings()
            print(title_features)
            pca = PCA(n_components=10)
            res = pca.fit_transform(title_features)
            print(len(res))
            pred = res_cat.predict_relevancy(MODEL_PATH, res)
            if pred:
                not_called_categories.append((category, title))
                relevant_categories_df = relevant_categories_df.append({'Referenced_from': category, 'Category': title},
                                                                       ignore_index=True)

            else:
                nonrelevant_categories_df = relevant_categories_df.append({'Referenced_from': category, 'Category': title}, ignore_index=True)

        if title not in called_categories and namespace == 0:
            called_articles.add(title)


    if idx == 200: break
    time.sleep(1)
    idx += 1


relevant_categories_df.to_csv(f'relevant.csv', index=False)
non_relevant_categories_df.to_csv(f'non_relevant.csv', index=False)