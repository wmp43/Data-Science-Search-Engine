import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.decomposition import PCA


# Need to set categories that contain the word stub to label 0
# Need to find a balancing method for the positive case or just manually find more categories



# df = pd.read_csv('/Users/owner/myles-personal-env/Projects/wikiSearch/src/ingestion/category_df_labelled.csv')
#
# nlp = spacy.load("en_core_web_sm")
# def contains_ner(text, ent_type):
#     doc = nlp(text)
#     return any(ent.label_ == ent_type for ent in doc.ents)
#
# df['Contains Place'] = df['Category'].apply(lambda x: contains_ner(x, "GPE"))
# df['Contains Person'] = df['Category'].apply(lambda x: contains_ner(x, "PERSON"))
# df['Contains Organization'] = df['Category'].apply(lambda x: contains_ner(x, "ORG"))
# df['Contains Year'] = df['Category'].apply(lambda x: 1 if re.search(r'\b(1|2)\d{3}\b', x) else 0)
# df['Contains Date'] = df['Category'].apply(lambda x: 1 if re.search(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}', x) else 0)
# df['Text Length'] = df['Category'].apply(len)
# df['Label'].fillna(0, inplace=True)
#
# df.to_csv('final_fe_cat_labelled.csv')
df = pd.read_csv('../final_fe_cat_labelled.csv')
df['Category'] = df['Category'].str.replace('^Category:', '', regex=True)


# Convert category text to a numerical feature vector using TF-IDF
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Category'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

pca = PCA(n_components=50)
tfidf_pca = pca.fit_transform(tfidf_df)

# Convert the PCA-transformed data into a DataFrame
tfidf_pca_df = pd.DataFrame(tfidf_pca, columns=[f'PCA_{i+1}' for i in range(tfidf_pca.shape[1])])

# Combine the PCA-transformed TF-IDF features with the original DataFrame
df = pd.concat([df.reset_index(drop=True), tfidf_pca_df.reset_index(drop=True)], axis=1)


X = df.drop(columns=['Label', 'Unnamed: 0'])
y = df['Label'].astype(int)  # Target is the 'Label' column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.columns)

preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', TfidfVectorizer(stop_words='english'), 'Category')
    ],
    remainder='passthrough'
)

# Create a pipeline with the preprocessor, feature selection, and XGBoost classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# Define a parameter grid to search over
param_grid = {
    'xgb__max_depth': [3, 5, 7],  # Depth of trees, to control overfitting
    'xgb__n_estimators': [100, 200],  # Number of trees
    'xgb__learning_rate': [0.1, 0.01],  # Learning rate, to control overfitting
}

# Set up GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=2)

# Train the model with GridSearchCV to find the best parameters
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Predict with the best estimator
best_estimator = grid_search.best_estimator_
predictions_test = best_estimator.predict(X_test)
predictions_train = best_estimator.predict(X_train)

# Print classification reports
print("Test Predictions")
print(classification_report(y_test, predictions_test))
print("Train Predictions")
print(classification_report(y_train, predictions_train))



import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Predict probabilities for the test set
y_scores = best_estimator.predict_proba(X_test)[:, 1]

# Calculate Precision-Recall curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_scores)

# Calculate ROC curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_scores)

# Calculate area under the ROC curve
roc_auc = auc(fpr, tpr)

# Plot Precision-Recall curve
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

# Plot ROC curve
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

