import pandas as pd
#import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import re



DATA_PATH = ('/Users/owner/myles-personal-env/Projects/'
             'wikiSearch/src/ingestion/category_classification/dataframes/category_df_labelled.csv')

df = pd.read_csv(DATA_PATH)

TRY_2 = True
if TRY_2:
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # nltk.download('wordnet')

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))


    # Function to clean text data
    def clean_text(text):
        # Remove 'Category:' prefix
        text = text.replace('Category:', '').strip()
        text = text.replace('-',' ')
        # Convert to lowercase
        text = text.lower()
        # Tokenize the text
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize the tokens
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
        # Join the tokens back into a string
        return ' '.join(tokens)


    df['Cleaned_Category'] = df['Category'].apply(clean_text)
    df.drop(columns=['Category'], inplace=True)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['Cleaned_Category'].tolist(), show_progress_bar=True)

    embeddings_df = pd.DataFrame(embeddings)

    final_df = pd.concat([embeddings_df, df], axis=1)
    final_df.to_csv('embedding_df.csv')

# This was the first try and kinda sucked ass

TRY_1 = False
if TRY_1:
    nlp = spacy.load("en_core_web_sm")
    def contains_ner(text, ent_type):
         doc = nlp(text)
         return any(ent.label_ == ent_type for ent in doc.ents)

    df['Contains Place'] = df['Category'].apply(lambda x: contains_ner(x, "GPE"))
    df['Contains Person'] = df['Category'].apply(lambda x: contains_ner(x, "PERSON"))
    df['Contains Organization'] = df['Category'].apply(lambda x: contains_ner(x, "ORG"))
    df['Contains Year'] = df['Category'].apply(lambda x: 1 if re.search(r'\b(1|2)\d{3}\b', x) else 0)
    df['Contains Date'] = df['Category'].apply(lambda x: 1 if re.search(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}', x) else 0)
    df['Text Length'] = df['Category'].apply(len)
    df['Label'].fillna(0, inplace=True)
    df['Category'] = df['Category'].str.replace('^Category:', '', regex=True)
    #
    print(df['Label'].value_counts())
    print(df.shape, df.columns)

    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Category'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    pca = PCA(n_components=75)
    tfidf_pca = pca.fit_transform(tfidf_df)
    tfidf_pca_df = pd.DataFrame(tfidf_pca, columns=[f'PCA_{i+1}' for i in range(tfidf_pca.shape[1])])
    df = pd.concat([df.reset_index(drop=True), tfidf_pca_df.reset_index(drop=True)], axis=1)
    df.to_csv('final_cat_clf_df.csv', index=False)