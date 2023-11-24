import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA





df = pd.read_csv('/Users/owner/myles-personal-env/Projects/wikiSearch/src/ingestion/category_classification/dataframes/embedding_df.csv')
df['Label'].fillna(0, inplace=True)
X = df.drop(columns=['Label', 'Cleaned_Category', 'Index'])
# pca = PCA(n_components=10)
# tfidf_pca = pca.fit_transform(X)
y = df['Label'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define a set of parameters for grid search
xgb_param_grid = {
    'max_depth': [2],
    'n_estimators': [60],
    'learning_rate': [0.15],
    'min_child_weight': [2.5],
    'gamma': [3],
    'scale_pos_weight': [0.75],
    'lambda': [2.5],
    'alpha': [1.5],
    'colsample_bytree': [0.5]
}

# Initialize the classifier
xgb_clf = XGBClassifier(random_state=42, objective='binary:logistic')
lr_clf = LogisticRegression(random_state=42)
import sklearn
print(sorted(sklearn.metrics.SCORERS.keys()))
# Set up GridSearchCV
grid_search = GridSearchCV(estimator=xgb_clf, param_grid=xgb_param_grid, scoring='balanced_accuracy', cv=3, verbose=2)

# Fit the GridSearchCV
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best parameters and the best estimator
best_params = grid_search.best_params_
best_clf = grid_search.best_estimator_

print(f'Best Parameters: {best_params}\n')

# Predict on the test set using the best estimator
y_pred_test = best_clf.predict(X_test)
y_pred_train = best_clf.predict(X_train_resampled)

best_clf.save_model('category_clf.json')

# Evaluate the classifier using the best estimator
print('Test CLF:')
print(classification_report(y_test, y_pred_test))
print('Train CLF:')
print(classification_report(y_train_resampled, y_pred_train))

y_probs_test = best_clf.predict_proba(X_test)[:, 1]


import numpy as np
# # Define a range of thresholds to test
thresholds = np.linspace(0.4, 0.55, 4)

# Evaluate each threshold
for thresh in thresholds:
    # Apply threshold to probability to create new predictions
    y_pred_new_threshold = (y_probs_test > thresh).astype(int)
    print(f'Threshold: {thresh}')
    print(classification_report(y_test, y_pred_new_threshold))