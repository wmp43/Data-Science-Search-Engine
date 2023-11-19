from classes import WikipediaAPI, Article
from collections import deque
import time
import matplotlib.pyplot as plt

not_called_categories = deque(['Category:machine_learning', 'Category:data_science', 'Category:statistics',
                               'Category:databases', 'Category:computer_science', 'Category:artificial_intelligence',
                               'Category:game_theory', 'Category:econometrics', 'Category:big_data'])
called_categories = set()
called_articles = set()

idx = 0
article_counts_per_pass = []
categories_remaining_per_pass = []

while not_called_categories:

    category = not_called_categories.popleft()
    #if category is useful: run it, if not, don't
    if category in called_categories:
        continue
    called_categories.add(category)
    print(f'Current Category: {category}\nNumber of Categories Traversed: {len(called_categories)}\n'
          f'Number of Categories Left: {len(not_called_categories)}'
          f'\nNumber of Articles Traversed: {len(called_articles)}')
    api_results = WikipediaAPI().get_category_data(category)
    for result in api_results:
        title, namespace = result[1], result[3]
        if namespace == 14:
            if title not in called_categories:
                not_called_categories.append(title)
        else:
            if title not in called_articles:
                called_articles.add(title)

    article_counts_per_pass.append(len(called_articles))
    categories_remaining_per_pass.append(len(not_called_categories))
    time.sleep(1)
    idx += 1

plt.figure(figsize=(10, 6))  # Optional: Set the size of the plot
plt.plot(article_counts_per_pass, label='Number of Articles Processed', color='blue')
plt.plot(categories_remaining_per_pass, label='Number of Categories Remaining', color='red')

plt.xlabel('Iterations')
plt.ylabel('Count')
plt.title('Progress of Articles and Categories Over Iterations')
plt.legend()
plt.grid(True)
plt.show()