import pandas as pd
import numpy as np
import sklearn
"""
Due to the exponentially increasing information gain with each category searched:
Must develop a method to classify if a category is useful for wikiSearch
For example, there are obvious cases of no use, and less obvious cases:
Clearly Not Useful:
- Demographics of Moldova
- Demographics of Monaco
- Demographics of ___________(Nation)
- History of Animal Testing
- Animal euthanasia
- heritage Documentation Programs
- Music Review Sites
- List of Horse racing Results
- Austrian Demographers
- New Zealand Bioinformatics
Any place with a location, many of the stray categories have nations in the title



Deceptively Not Useful: May contain a keyword, but do not hold any useful information for a DS/ML IR system
- United States Census Bureau
- Statistician Generals of South Africa
- Hockey Statistics
- Google Chrome
- Messenger (Software)
- Tiktok
- Air Command and Control Systems
- Defunct Computer System companies
- Cricket Statisticians



Deceptively Useful:
- 
Clearly Useful:
- Validity (statistics)
- Central Limit Theorem
- Bayesian Inference
- Statistical Hypothesis testing
- Discrete Distributions
- Continous Distributions
- Approximation Algorithms
- Graph Algorithms
- Heuristic Algorithms
- Search algorithms
- Arrays
- 
"""