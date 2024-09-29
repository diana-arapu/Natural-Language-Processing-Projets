import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# making the reviews in lowercase and stemming
def data_preprocessing(df):
    corpus = []
    ps = SnowballStemmer(language='english') 

    for i in range(len(df)):
        review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
        review = " ".join(review)

        corpus.append(review)

    return corpus

def evaluate_results(y_test, y_pred):
    print("Accuracy score\n")
    print(accuracy_score(y_test, y_pred), "\n")
    print("\nConfusion matrix\n")
    print(confusion_matrix(y_test, y_pred), "\n")
    print("\nOther metrics\n")
    print(classification_report(y_test, y_pred), "\n")


if __name__ == "__main__":
    df = pd.read_csv("Restaurant_Reviews.tsv", sep='\t', quoting=3)

    corpus = data_preprocessing(df)

    cv = CountVectorizer(max_features=1500)

    x = cv.fit_transform(corpus).toarray()
    y = df.iloc[:, 1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    evaluate_results(y_test, y_pred)