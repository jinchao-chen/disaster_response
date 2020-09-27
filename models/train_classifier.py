import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import GridSearchCV

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
import nltk
import pickle
import sys

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")


def load_data(database_filepath):
    """Read training dataset from database

    Args:
        database_filepath (str): database location
    
    Returns:
        X (numpy.ndarray): T
        Y (numpy.ndarray):
        cat_names (list): list of the categories

    To-dos: 
        - modify read_sql_table to make it more flexialbe
        - identify the cat columns
    """
    engine = create_engine("sqlite:///{:}".format(database_filepath))
    df = pd.read_sql_table(table_name="DisasterResponse", con=engine)
    X = df["message"].values

    cat_names = df.columns[5::].values
    Y = df[cat_names].values

    return X, Y, cat_names


def tokenize(text):
    """Tokenize input text message
    pre-processing the text message through: normalize, tokenize, remove stop words, stem and lemmatize

    Args: 
        text (str): the content of the message
    
    Return:
        words (list): list of tokenized message
    
    """
    # Normalize the text
    text = re.sub(r"\W", " ", text.lower().strip())

    # Tokenize
    words = word_tokenize(text)

    # Remove stop word
    words = [w for w in words if w not in stopwords.words("english")]

    # Stemming and lemmatize
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word).strip() for word in words]

    return words


def build_model():
    """ Create ML pipeline

    Args: 
        None

    Returns:
        cv(sklearn.pipeline): grid search model
    """

    # Build ML pipeline
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )
    # Use gridsearch to finetune the model
    parameters = {
        "clf__estimator__max_depth": [10, 50],
        "tfidf__sublinear_tf": [True, False],
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    """Evaluate the created model

    Args:
        model (sklearn.pipeline): 
        X_test (numpy.ndarray): Testing features
        Y_test (numpy.ndarray): Testing labels
        category_names (numpy.ndarray): Names of the features
    
    Returns:
        None
    """

    Y_pred = model.predict(X_test)

    for idx, cat in enumerate(category_names):
        print("------------------------")
        print("{:}".format(cat.upper()))
        print(classification_report(Y_test[idx], Y_pred[:, idx]))
    pass


def save_model(model, model_filepath):
    """ Dump the trained model in a pickle file

    Args:
        model (sklearn.pipeline): Trained ML model
        model_filepath (str): Pickle file path name
    
    Returns:
        None
    """
    with open(model_filepath, "wb") as file:
        pickle.dump(model, file)

    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
