# import networkx as nx
# import matplotlib.pyplot as plt
#
# # Example data
# data = {
#     "embedding": [1.1, 3.4, 3.2, -4, -0.1, 5],
#     "id": "123-gha-312",
#     "title": "bayes_theorem",
#     "wikipedia_id": "123456", # Wiki ID is the ID that wikipedia gives us. This can be used
#     "metadata":{
#         "categories": ['bayesian_statistics', 'posterior_probability', 'bayes_estimation'],
#         "mentioned_people": ['william_bayes'],
#         "mentioned_places": ['london'],
#         "mentioned_topics": ['bayesian_economics', 'bayesian_deep_learning'],
#         "storage_date": dt.object,
#         "summary": 'bayes_theorem is the application of posterior probabilities to statistical modeling', #HF based on first chunk',
#         "url": "https://wikilinkforbayestheorem.com",
#     }
# }
#
# # Creating a graph
# G = nx.Graph()
# G.add_node(data["title"], type='article')
# for cat in data["metadata"]["categories"]:
#     G.add_node(cat, type='category')
#     G.add_edge(data["title"], cat)
# for person in data["metadata"]["mentioned_people"]:
#     G.add_node(person, type='person')
#     G.add_edge(data["title"], person)
# # ... similarly for places and topics
#
# # Customizing the visualization
# color_map = {'article': 'blue', 'category': 'green', 'person': 'red', 'place': 'purple', 'topic': 'orange'}
# colors = [color_map[G.nodes[node]['type']] for node in G]
#
# # Drawing the graph
# nx.draw(G, with_labels=True, node_color=colors)

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Example data for bayes_theorem
bayes_theorem_data = {
    "title": "bayes_theorem",
    "metadata": {
        "categories": ['bayesian_statistics', 'posterior_probability', 'bayes_estimation'],
        "mentioned_people": ['william_bayes'],
        "mentioned_places": ['london'],
        "mentioned_topics": ['bayesian_economics', 'bayesian_deep_learning']
    }
}

# Additional data for bayesian_economics
bayesian_economics_data = {
    "title": "bayesian_economics",
    "metadata": {
        "categories": ['economic_theory', 'bayesian_statistics'],
        "mentioned_people": ['john_maynard_keynes', 'william_bayes'],  # Including a similar mention
        "mentioned_places": ['cambridge'],
        "mentioned_topics": ['game_theory', 'decision_theory']
    }
}

decision_theory_data = {
    "title": "decision_theory",
    "metadata": {
        "categories": ['cognitive_science', 'game_theory', 'rational_choice_theory'],
        "mentioned_people": ['daniel_kahneman', 'amartya_sen'],
        "mentioned_places": ['harvard_university'],
        "mentioned_topics": ['behavioral_economics', 'risk_analysis']
    }
}

game_theory_data = {
    "title": "game_theory",
    "metadata": {
        "categories": ['applied_mathematics', 'economic_theory', 'decision_theory'],
        "mentioned_people": ['john_von_neumann', 'john_nash'],
        "mentioned_places": ['princeton_university'],
        "mentioned_topics": ['nash_equilibrium', 'zero_sum_game']
    }
}


bayesian_deep_learning_data = {
    "title": "bayesian_deep_learning",
    "metadata": {
        "categories": ['machine_learning', 'bayesian_statistics', 'neural_networks'],
        "mentioned_people": ['geoffrey_hinton', 'yann_lecun'],
        "mentioned_places": ['stanford_university'],
        "mentioned_topics": ['deep_learning', 'probabilistic_modeling']
    }
}




# Function to add nodes and edges to the graph
def add_to_graph(graph, article_data):
    graph.add_node(article_data["title"], type='article')
    for cat in article_data["metadata"]["categories"]:
        graph.add_node(cat, type='category')
        graph.add_edge(article_data["title"], cat)
    for person in article_data["metadata"]["mentioned_people"]:
        graph.add_node(person, type='person')
        graph.add_edge(article_data["title"], person)
    for place in article_data["metadata"]["mentioned_places"]:
        graph.add_node(place, type='place')
        graph.add_edge(article_data["title"], place)
    for topic in article_data["metadata"]["mentioned_topics"]:
        graph.add_node(topic, type='topic')
        graph.add_edge(article_data["title"], topic)


# def get_contrasting_color(node_color):
#     # Assuming node_color is a hex string, convert it to RGB
#     r, g, b = int(node_color[1:3], 16), int(node_color[3:5], 16), int(node_color[5:], 16)
#     # Calculate the brightness of the color
#     brightness = r * 0.299 + g * 0.587 + b * 0.114
#     # Return 'white' for dark colors, 'black' for bright colors
#     return 'white' if brightness < 123 else 'black'
#
# # Example usage
# font_colors = [get_contrasting_color(color) for color in colors]

# Creating a graph
G = nx.Graph()

# Adding bayes_theorem and bayesian_economics data to the graph
add_to_graph(G, bayes_theorem_data)
add_to_graph(G, bayesian_economics_data)
add_to_graph(G, decision_theory_data)
add_to_graph(G, game_theory_data)
add_to_graph(G, bayesian_deep_learning_data)

plt.figure(figsize=(15, 10))

# Choose a layout
pos = nx.spring_layout(G, k=0.15)  # k is the optimal distance between nodes

# Drawing the graph with the chosen layout
color_map = {'article': 'blue', 'category': 'green', 'person': 'red', 'place': 'purple', 'topic': 'orange'}
colors = [color_map[G.nodes[node]['type']] for node in G]


nx.draw(G, pos, with_labels=True, node_color=colors, node_size=1500, font_size=12, font_color='black')

# Create a legend
legend_handles = []
for node_type, color in color_map.items():
    patch = mpatches.Patch(color=color, label=node_type)
    legend_handles.append(patch)

plt.legend(handles=legend_handles, loc='upper left')

# Show plot
plt.show()



#Expanded graphing theory, can easily build network graphs from vector representations.
