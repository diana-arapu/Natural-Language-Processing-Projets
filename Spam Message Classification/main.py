import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def data_analysis(df):
    print("Looking at the overall organization of the dataset\n", df.head())
    print("\nChecking for mising values\n", df.isna().sum())
    print("\nMore details...\n", df.describe())
    print("\nChecking if the data is balanced\n", df['label'].value_counts() * 100 / len(df))

def data_visualization(data):
    plt.hist(data[data['label'] == 'ham']['length'], label = 'ham', bins = 100, alpha = 0.7)
    plt.hist(data[data['label'] == 'spam']['length'], label = 'spam', bins = 100, alpha = 0.7)
    plt.ylabel("Message frequency")
    plt.xlabel("Message length")
    plt.title("Message length frequency")
    plt.legend()
    plt.show()

    plt.hist(data[data['label'] == 'ham']['punct'], label = 'ham', bins = 100, alpha = 0.7)
    plt.hist(data[data['label'] == 'spam']['punct'], label = 'spam', bins = 100, alpha = 0.7)
    plt.ylabel("Punctuation frequency")
    plt.xlabel("Punctuation counts")
    plt.title("Punctuation count frequency in messages")
    plt.legend()
    plt.show()

#making the dataset balanced: 50% ham messages, 50% spam messages
def data_preprocessing(df):
    ham = df[df['label'] == 'ham']
    spam = df[df['label'] == 'spam']
    ham = ham.sample(spam.shape[0])
    data = ham.append(spam, ignore_index=True)
    return data

def evaluate_results(y_test, y_pred1, y_pred2):
    print("Accuracy score\n")
    print("TD-IDF + Random Forests: ", accuracy_score(y_test, y_pred1), "\n")
    print("TD-IDF + SVM: ", accuracy_score(y_test, y_pred2), "\n")
    print("\nConfusion matrix\n")
    print("TD-IDF + Random Forests: \n", confusion_matrix(y_test, y_pred1), "\n")
    print("TD-IDF + SVM: \n", confusion_matrix(y_test, y_pred2), "\n")
    print("\nOther metrics\n")
    print("TD-IDF + Random Forests: \n", classification_report(y_test, y_pred1), "\n")
    print("TD-IDF + SVM: \n", classification_report(y_test, y_pred2), "\n")


if __name__ == "__main__":
    df = pd.read_csv('spam.tsv', sep='\t')
    #data_analysis(df)

    data = data_preprocessing(df)
    #data_visualization(data)

    x_train, x_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size = 0.3, random_state = 0, shuffle = True)

    #building and training the first model: TD-IDF Vectorizer + Random Forest
    classifier1 = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", RandomForestClassifier(n_estimators=100))])
    classifier1.fit(x_train, y_train)
    y_pred1 = classifier1.predict(x_test)

    #building and training the second model: TD-IDF Vectorizer + Support Vector Machine
    classifier2 = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", SVC(C = 100, gamma ='auto'))])
    classifier2.fit(x_train, y_train)
    y_pred2 = classifier2.predict(x_test)

    #Comparing results
    evaluate_results(y_test, y_pred1, y_pred2)