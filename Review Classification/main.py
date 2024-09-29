import pandas as pd
import string
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def data_analysis(data):
    print("Looking at the overall organization of the dataset\n", data.head())
    print("Checking if data is balanced", data['Sentiment'].value_counts())
    print("\nChecking for mising values\n", data.isnull().sum())

def data_preprocessing(df_yelp, df_amazon, df_imdb):
    column_name = ['Review', 'Sentiment']
    df_yelp.columns = column_name
    df_amazon.columns = column_name
    df_imdb.columns = column_name

    data = pd.concat([df_yelp, df_amazon, df_imdb], ignore_index=True)
    data_analysis(data)

    return data

def data_cleaning(sentence):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)

    tokens = []
    # if the word is a proper noun, there is no lemma for it, so we just take the lowercase
    # otherwise we first get the lemma of the word and then lowercase
    for token in doc:
        if token.lemma_ != "-PRON-":
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)

    cleaned_tokens = []
    #removing stopwords and punctuation
    for token in tokens:
        if token not in list(STOP_WORDS) and token not in list(string.punctuation):
            cleaned_tokens.append(token)

    return cleaned_tokens

def evaluate_results(y_test, y_pred):
    print("Accuracy score\n")
    print(accuracy_score(y_test, y_pred), "\n")
    print("\nConfusion matrix\n")
    print(confusion_matrix(y_test, y_pred), "\n")
    print("\nOther metrics\n")
    print(classification_report(y_test, y_pred), "\n")

if __name__ == "__main__":
    df_yelp = pd.read_csv('yelp_labelled.txt', sep='\t', header=None)
    df_amazon = pd.read_csv('amazon_cells_labelled.txt', sep='\t', header=None)
    df_imdb = pd.read_csv('imdb_labelled.txt', sep='\t', header=None)

    data = data_preprocessing(df_yelp, df_amazon, df_imdb)

    x = data['Review']
    y = data['Sentiment']

    #tokenization done according to data_cleaning
    tfidf = TfidfVectorizer(tokenizer = data_cleaning, token_pattern=None)
    classifier = LinearSVC()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0, shuffle = True)

    model = Pipeline([("tfidf", tfidf), ("classifier", classifier)])
    model.fit(x_train, y_train)

    y_pred = model.predict(y_test)

    evaluate_results(y_test, y_pred)

