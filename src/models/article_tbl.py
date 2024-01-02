"""
This file will upsert data in a new rds table for developing the embedding model
"""
from src.base_models import Article
from src.api import WikipediaAPI
from src.relational import ArticlesTable
from src.text_processor import BaseTextProcessor
from config import (rds_host, rds_dbname, rds_user, rds_password, rds_port)
from tqdm import tqdm
import json
"""
Ingestion pipeline for full article and text data
1. Define List of Articles to Snag
2. Call API with title
3. Instantiate Article with returned items
4. Clean Text
5. Store in DB
6. Hope it works
"""

SECTIONS_TO_IGNORE = [
    "See also", "References", "External links", "Further reading", "Footnotes", "Bibliography", "Sources",
    "Citations",
    "Literature", "Footnotes", "Notes and references", "Photo gallery", "Works cited", "Photos", "Gallery", "Notes",
    "References and sources", "References and notes"]
data_science_articles = [
    'Data_science',
    'Calculus_on_Euclidean_space',
    'Vector_calculus',
    'Statistic',
    'Statistics',
    'Bayesian_statistics',
    'Linear_algebra',
    'Calculus',
    'Discrete_mathematics',
    'Python_(programming_language)',
    'Data_mining',
    'Data_and_information_visualization',
    'Machine_learning',
    'Deep_learning',
    'Artificial_neural_network',
    'Support_vector_machine',
    'Random_forest',
    'Natural_language_processing',
    'Computer_vision',
    'Reinforcement_learning',
    'Supervised_learning',
    'Unsupervised_learning',
    'Weak_supervision',
    'Feature_engineering',
    'Model_selection',
    'Cross-validation_(statistics)',
    'Bootstrapping_(statistics)',
    'Statistical_hypothesis_testing',
    'Regression_analysis',
    'Time_series',
    'Dimensionality_reduction',
    'Cluster_analysis',
    'Big_data',
    'Data_cleansing',
    'Data_integration',
    'Predictive_modelling',
    'Decision_tree_learning',
    'Decision_theory',
    'Association_rule_learning',
    'Ensemble_learning',
    'Gradient_boosting',
    'Neural_network_software',
    'Neural_network',
    'Evolutionary_algorithm',
    'Genetic_programming',
    'Fuzzy_logic',
    'Information_theory',
    'Entropy_(information_theory)',
    'Kullback–Leibler_divergence',
    'Markov_decision_process',
    'Graph_theory',
    'Mathematical_optimization',
    'Convex_optimization',
    'Stochastic_gradient_descent',
    'Principal_component_analysis',
    'Factor_analysis',
    'Matplotlib',
    'NumPy',
    'PyTorch',
    'Categorical_variable',
    'Numerical_analysis',
    'Monte_Carlo_method',
    'Simulated_annealing',
    'Computational_complexity_theory',
    'Algorithm',
    'Critical_path_method',
    'Data_structure',
    'Distributed_computing',
    'Parallel_computing',
    'Quantum_computing',
    'High-performance_computing',
    'Computational_science',
    'Data_security',
    'Blockchain',
    'Edge_computing',
    'Bioinformatics',
    'Computational_biology',
    'Computational_neuroscience',
    'Robotics',
    'Quantum_machine_learning',
    'Causal_inference',
    'Design_of_experiments',
    'Statistical_learning_theory',
    'Algorithmic_bias',
    'Computational_learning_theory',
    'Data_warehouse',
    'Deep_reinforcement_learning',
    'Ethics_of_artificial_intelligence',
    'Feature_learning',
    'Game_theory',
    'Graphical_model',
    'Knowledge_representation_and_reasoning',
    'Learning_to_rank',
    'Machine_perception',
    'Machine_translation',
    'Multi-agent_system',
    'Multi-task_learning',
    'Pattern_recognition',
    'Predictive_analytics',
    'Recommender_system',
    'Speech_recognition',
    'Statistical_classification',
    'Structured_prediction',
    'Text_mining',
    'Anomaly_detection',
    'Autoencoder',
    'Bias–variance_tradeoff',
    'Nearest_neighbor_search',
    'Clustering_coefficient',
    'Convolutional_neural_network',
    'Decision_boundary',
    'Dimensionality_reduction',
    'Cluster_analysis',
    'Feature_selection',
    'Generative_adversarial_network',
    'Hyperparameter_optimization',
    'Loss_function',
    'Similarity_learning',
    'Large_language_model',
    'Curve_fitting',
    'Overfitting',
    'Precision_and_recall',
    'Recurrent_neural_network',
    'Regularization_(mathematics)',
    'Transfer_learning',
    'Training,_validation,_and_test_data_sets',
    'XGBoost',
    'Ilya_Sutskever',
    'Harvard',
    'Cornell',
    "OpenAI",
    "Massachusetts_Institute_of_Technology",
    'Google Brain',
    'Andrew_Ng',
    'Scikit-learn',
    'Stanford',
    'Probability_theory',
    'Probability_distribution',
    'Conditional_probability',
    "Bayes'_theorem",
    'Independence_(probability_theory)',
    'Random_variable',
    'Central_limit_theorem',
    'Variance',
    'Standard_deviation',
    'Covariance',
    'Correlation',
    'Sampling_distribution',
    'Chi-squared_test',
    'Analysis_of_variance',
    'Factorial_experiment',
    'Causal_inference',
    'Multivariate_statistics',
    'Nonparametric_statistics',
    'K-means_clustering',
    'Vector_quantization',
    'Euclidean_distance'

]

wiki_api = WikipediaAPI()
processor = BaseTextProcessor()
INGEST = True


if INGEST:
    unique_id = -2
    emb_tbl = ArticlesTable(rds_dbname, rds_user, rds_password, rds_host, rds_port)
    for TITLE in tqdm(set(data_science_articles), desc='Progress'):
        title, page_id, final_text = wiki_api.fetch_article_data(TITLE)
        if page_id == -1:
            page_id = unique_id
            unique_id -= 1
        article = Article(title=title, id=page_id, text=final_text, text_processor=processor)
        article.process_text_pipeline(processor, SECTIONS_TO_IGNORE)
        json_record = article.process_metadata_labeling(processor)
        # total_text = ""
        # for (key, text), metadata in zip(article.text_dict.items(), article.metadata_dict.values()):
        #     print(metadata, type(metadata))
        #     total_text += text
        # cleaned_text = re.sub(r'[\n\t]', ' ', total_text)
        emb_tbl.add_record(json_record['id'], json_record['text'], json_record['title'], json.dumps(json_record['labels']))
    emb_tbl.close_connection()