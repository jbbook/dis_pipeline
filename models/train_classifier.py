import sys
from sqlalchemy import create_engine

import re
import numpy as np
import pandas as pd
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#needs to be defined here, error if defined in tokenize. do not yet understand
#https://stackoverflow.com/questions/44911539/pickle-picklingerror-args0-from-newobj-args-has-the-wrong-class-with-hado
stop_words = set(stopwords.words('english'))

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier

def load_data(database_filepath):
    """Load messages from database file.

    Return:
    df['message'] - messages Series
    df[categories] - category DF
    categories - list of category names.
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('messages', engine)
    categories = df.columns.drop(['genre', 'id', 'message', 'original'])

    return df['message'], df[categories], categories 


def tokenize(text):
    """Tokenize and preprocess text."""
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    #remove urls
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    text = replace_punctuation(text)

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    filtered_tokens = [w for w in clean_tokens if not w in stop_words]

    return filtered_tokens

def replace_punctuation(text):
    """Remove backslash,' and " in texts. Replace other punctuation with white spaces."""
    text = re.sub(r"\\", "", text)    
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text

def build_model():
    """Build model using Pipeline and GridsearchCV. Return model."""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)), 
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    #parameters for GridSearchCV. Best params only, due to comp. time restrictions
    parameters = {
        'clf__estimator__n_estimators': [100],
        'clf__estimator__learning_rate': [1.],
        'vect__ngram_range': [(1,2)],
        'tfidf__use_idf': [True],
        'vect__max_df': [1.0],
    }
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=6, scoring='f1_macro', verbose=1)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """Print classification report and confusion matrix for all categories."""
    Y_test = Y_test.to_numpy()
    Y_pred = model.best_estimator_.predict(X_test)
    for i in range(len(category_names)):
        print('######### Category {} #########'.format(category_names[i]))
        print('Classification Report:')
        print(classification_report(Y_test.T[i], Y_pred.T[i]))
        print('Confusion Matrix: \n TN FP \n FN TP')
        print(confusion_matrix(Y_test.T[i], Y_pred.T[i]))
        print('\n')



def save_model(model, model_filepath):
    """Save model into pickle file."""
    outfile = open(model_filepath, 'wb')
    pickle.dump(model.best_estimator_, outfile)


def main():
    """Main function of train_classifier.py."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()

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