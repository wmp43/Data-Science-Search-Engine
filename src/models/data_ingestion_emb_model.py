"""
This file will upsert data in a new rds table for developing the embedding model
"""
from src.models import Article
from src.api import WikipediaAPI
from src.relational import EmbeddingModelTable
from src.text_processor import BaseTextProcessor
from config import (rds_host, rds_dbname, rds_user, rds_password, rds_port)
from tqdm import tqdm
import re
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
    'Association_rule_learning',
    'Ensemble_learning',
    'Gradient_boosting',
    'Neural_network_software',
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
    'Categorical_variable',
    'Numerical_analysis',
    'Monte_Carlo_method',
    'Simulated_annealing',
    'Computational_complexity_theory',
    'Algorithm',
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
    'Artificial_intelligence_ethics',
    'Computational_learning_theory',
    'Data_mining_algorithms',
    'Data_warehouse',
    'Database_indexing',
    'Deep_reinforcement_learning',
    'Ethics_of_artificial_intelligence',
    'Explainable_AI',
    'Feature_learning',
    'Game_theory',
    'Graphical_model',
    'Knowledge_representation',
    'Learning_to_rank',
    'Machine_learning_in_healthcare',
    'Machine_perception',
    'Machine_translation',
    'Multi-agent_system',
    'Multi-task_learning',
    'Pattern_recognition',
    'Predictive_analytics',
    'Recommender_system',
    'Semantic_web',
    'Speech_recognition',
    'Statistical_classification',
    'Structured_prediction',
    'Text_mining',
    'Text_to_speech',
    'Time_series_analysis',
    'Anomaly_detection',
    'Autoencoder',
    'Bias-variance_tradeoff',
    'Classification_algorithm',
    'Clustering_algorithm',
    'Convolutional_neural_network',
    'Data_preprocessing',
    'Decision_boundary',
    'Dimensionality_reduction_algorithm',
    'Ensemble_methods',
    'Feature_selection',
    'Generative_adversarial_network',
    'Hyperparameter_optimization',
    'Imbalanced_data',
    'Loss_functions',
    'Metric_learning',
    'Model_evaluation',
    'Model_fitting',
    'Neural_network_architecture',
    'Outlier_detection',
    'Overfitting',
    'Precision_and_recall',
    'Recurrent_neural_network',
    'Regularization',
    'Transfer_learning',
    'Training_dataset',
    'Validation_dataset',
    'XGBoost',
    'Probability_theory',
    'Probability_distribution',
    'Conditional_probability',
    'Bayes_theorem',
    'Statistical_independence',
    'Random_variable',
    'Central_limit_theorem',
    'Variance',
    'Standard_deviation',
    'Covariance',
    'Correlation',
    'Sampling_distribution',
    'Chi_squared_test',
    'Student_t-test',
    'ANOVA',
    'Factorial_experiment',
    'Causal_statistical_inference',
    'Multivariate_statistics',
    'Non-parametric_statistics'
]
wiki_api = WikipediaAPI()
processor = BaseTextProcessor()
INGEST = True

if INGEST and len(data_science_articles) == len(set(data_science_articles)):
    counter = 0
    emb_tbl = EmbeddingModelTable(rds_dbname, rds_user, rds_password, rds_host, rds_port)
    for TITLE in tqdm(data_science_articles, desc='Progress'):
        title, page_id, final_text = wiki_api.fetch_article_data(TITLE)
        if page_id == -1:
            counter += 1
            print(f'title not found {title}, Counter: {counter}')
        article = Article(title=title, id=page_id, text=final_text, text_processor=processor)
        article.process_text_pipeline(processor, SECTIONS_TO_IGNORE)
        total_text = ""
        for content in article.text_dict.values():
            total_text += content
        # text is stored in total_text, title in article.title, and article.id
        cleaned_text = re.sub(r'[\n\t]', ' ', total_text)
        emb_tbl.add_record(article.id, cleaned_text, article.title)
    emb_tbl.close_connection()


