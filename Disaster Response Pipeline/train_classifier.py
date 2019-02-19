import sys
import pandas as pd
import numpy as np
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from sqlalchemy import create_engine

import pickle

def load_data(database_filepath):
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('dataset', con=engine)

    
    df = df.dropna() 

    X = df.loc[:, 'message']
    Y = df.iloc[:, 4:]
    categories = list(Y)

    return X.values, Y.values, categories


def tokenize(text):
    
    tokens = word_tokenize(text)
    wl = WordNetLemmatizer()
   
    tokens = [wl.lemmatize(t).lower().strip() for t in tokens]

    return tokens


def build_model(parameters={}):
    
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
                            ('tfidf', TfidfTransformer()),
                            ('classifier', MultiOutputClassifier(RandomForestClassifier(**parameters)))])
    return pipeline


def optimal_params(model, X_train, Y_train):
    
    parameters = {
        'classifier__estimator__n_estimators': [50, 100, 150],
        'classifier__estimator__max_features': ['sqrt',],
        'classifier__estimator__criterion': ['entropy', 'gini']
    }

    cv = GridSearchCV(model, param_grid = parameters, verbose=1)
    cv.fit(X_train, Y_train)

    return cv.best_params_

def evaluate_model(model, X_test, Y_test, category_names):
    
    predictions = model.predict(X_test)
    
   
    print("Accuracy scores for each category\n")
    print("*-" * 30)

    for i in range(36):
        print("Category:", category_names[i],"\n", classification_report(Y_test[:, i], predictions[:, i]))


def save_model(model, model_filepath):
    
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
        
        print('Grid search started')
        optimal_parameters = optimal_params(model, X_train, Y_train)
        random_forest_params = {
                    'n_estimators': optimal_parameters['classifier__estimator__n_estimators'],
                    'max_features': optimal_parameters['classifier__estimator__max_features'],
                    'criterion': optimal_parameters['classifier__estimator__criterion'],
                }

        print("Optimal parameters")
        print(random_forest_params)

        print('Building random forest model with optimal parameters')
        model = build_model(random_forest_params)

        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()