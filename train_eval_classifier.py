from typing import List
from scipy.stats import sem 
from numpy import mean 
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix
import numpy as np
from joblib import dump
from matplotlib import pyplot as plt

max_accuracy = -1
max_train_index = [] 
max_test_index = []


def preprocess_tweet(tweet: str) -> str:
    """
    Performs Twitter-specific pre-processing steps.
    We do not remove punctuation right now because that would remove hashtags.
    """
    import re
    import contractions
    url_regex = r"\w+:\/\/\S+"
    handle_regex = r"@[\w\d_]+"
    space_regex = r'\s+'
    number_regex = r'\d+'

    # replace long whitespace with single space
    cleaned_tweet = re.sub(space_regex, ' ', tweet)
    cleaned_tweet = re.sub(url_regex, '', cleaned_tweet)  # remove urls
    # remove user handles
    cleaned_tweet = re.sub(handle_regex, '', cleaned_tweet)
    cleaned_tweet = re.sub(number_regex, '', cleaned_tweet)  # remove numbers

    cleaned_tweet = contractions.fix(cleaned_tweet).lower()
    return cleaned_tweet


def tokenize(tweet: str, method: str = "tweet") -> List[str]:
    """
    Returns tokens of a tweet, tokenized using a specified method, with all the tokens stemmed.
    tweet is a string
    method is a string
    """
    from nltk.tokenize import TweetTokenizer, word_tokenize
    from nltk.stem import PorterStemmer
    import emoji
    import string

    tokens = []
    if method == "tweet":
        tweet_tokenizer = TweetTokenizer()
        tokens = tweet_tokenizer.tokenize(tweet)
    elif method == "word":
        tokens = word_tokenize(tweet)

    # remove tokens that are just punctuation
    punctuation_list = [punct for punct in string.punctuation]
    tokens_punctuation_removed = [
        t for t in tokens if t not in punctuation_list]

    # remove emojis
    tokens_emojis_removed = [
        t for t in tokens_punctuation_removed if t not in emoji.UNICODE_EMOJI]

    # also perform stemming
    return [PorterStemmer().stem(t) for t in tokens_emojis_removed]

# from sklearn.model_selection import cross_validate
# def cross_validation(model, _X, _y, _cv=5):
#       '''Function to perform 5 Folds Cross-Validation
#        Parameters
#        ----------
#       model: Python Class, default=None
#               This is the machine learning algorithm to be used for training.
#       _X: array
#            This is the matrix of features.
#       _y: array
#            This is the target variable.
#       _cv: int, default=5
#           Determines the number of folds for cross-validation.
#        Returns
#        -------
#        The function returns a dictionary containing the metrics 'accuracy', 'precision',
#        'recall', 'f1' for both training set and validation set.
#       '''
#       _scoring = ['accuracy', 'precision', 'recall', 'f1']
#       results = cross_validate(estimator=model,
#                                X=_X,
#                                y=_y,
#                                cv=_cv,
#                                scoring=_scoring,
#                                return_train_score=True)
      
#       return {"Training Accuracy scores": results['train_accuracy'],
#               "Mean Training Accuracy": results['train_accuracy'].mean()*100,
#               "Training Precision scores": results['train_precision'],
#               "Mean Training Precision": results['train_precision'].mean(),
#               "Training Recall scores": results['train_recall'],
#               "Mean Training Recall": results['train_recall'].mean(),
#               "Training F1 scores": results['train_f1'],
#               "Mean Training F1 Score": results['train_f1'].mean(),
#               "Validation Accuracy scores": results['test_accuracy'],
#               "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
#               "Validation Precision scores": results['test_precision'],
#               "Mean Validation Precision": results['test_precision'].mean(),
#               "Validation Recall scores": results['test_recall'],
#               "Mean Validation Recall": results['test_recall'].mean(),
#               "Validation F1 scores": results['test_f1'],
#               "Mean Validation F1 Score": results['test_f1'].mean()
#               }

# def train_clf(train_tweets: List[str], train_labels: List[int], 
#               test_tweets: List[str], test_labels: List[int],  train_i, test_i,
#               classifier_filename: str = None,
#               vectorizer_filename: str = None, 
#               error_log_filename: str = None) -> None:
    
#     """
#     Trains a classifier on `train_tweets` and 
#       saves the vectorizer and model to disk if respective filenames are passed in.
#     Prints out the classification report/scores of the trained classifier on `test_tweets`.
#     Also plots the confusion matrix and, if a filename is provided, misprediction information 
#       is written to another file.
#     """
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.metrics import classification_report, plot_confusion_matrix
#     from sklearn.model_selection import KFold
#     import numpy as np
#     from joblib import dump
#     from matplotlib import pyplot as plt

#     global max_accuracy
#     global max_train_index
#     global max_test_index

#     vectorizer = TfidfVectorizer(
#         tokenizer=tokenize,
#         preprocessor=preprocess_tweet,
#         ngram_range=(1, 3),
#         min_df=5,  # only consider terms that occur in at least 5 tweets
#         max_df=0.75  # only consider terms that occur in less than 75% of all tweets
#     )

#     # train

#     X_vectorized = vectorizer.fit_transform(train_tweets) 
#     # print(X_vectorized)
#     clf = RandomForestClassifier(n_estimators=400, min_samples_split=2,
#                                  min_samples_leaf=1, max_features='sqrt', 
#                                  max_depth=None, bootstrap=False, 
#                                  class_weight='balanced')
#     # clf_result = cross_validation(clf, X, y, 5)
#     clf.fit(X_vectorized, train_labels)

#     print("fits model")
#     # test and print classification report
#     X_test_vectorized = vectorizer.transform(test_tweets)
#     y_pred = clf.predict(X_test_vectorized)
#     clf_report = classification_report(test_labels, y_pred, output_dict = True)
#     accuracy_dic = clf_report['accuracy']
#     if accuracy_dic > max_accuracy:
#         max_accuracy = accuracy_dic
#         max_train_index = train_i
#         max_test_index = test_i

#     print("CLF Report")
#     print(clf_report)
    
    # if error_log_filename:
    #     with open(error_log_filename, 'w') as f:
    #         for i in range(len(y_pred)):
    #             if y_pred[i] != test_labels[i]:
    #                 f.write("***\n")
    #                 f.write(f"EXPECTED: {test_labels[i]}, PREDICTED: {y_pred[i]}\n")
    #                 f.write(f"TWEET: {test_tweets[i]}\n\n")
    #     print(f"Wrote incorrectly predicted tweets to {error_log_filename}.")

    
    # confusion matrix
    # plot_confusion_matrix(clf, X_test_vectorized, test_labels)
    # plt.show()
    
    
    # persist classifier and vectorizer
    # MODEL_ROOT_DIR = os.path.join('..', 'models')

    # if classifier_filename:
    #     full_clf_filename = os.path.join(MODEL_ROOT_DIR, f"{classifier_filename}.joblib")
    #     print(f"Saving classifier to {full_clf_filename}")
    #     dump(clf, full_clf_filename)
        
    # if vectorizer_filename:
    #     full_vect_filename = os.path.join(MODEL_ROOT_DIR, f"{vectorizer_filename}.joblib")
    #     print(f"Saving vectorizer to {full_vect_filename}")
    #     dump(vectorizer, full_vect_filename)


def evaluate_model(X, y, repeats):
    # X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
    cv = RepeatedKFold(n_splits=2, n_repeats=2, random_state=1)
    model = LogisticRegression()
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores
    # print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

if __name__ == '__main__':
    import os
    import pandas as pd


 
    # read tweets   
    filename = os.path.join('..', 'data', 'out',
                            'test_clean(RoundUp).csv') #CHANGED THE FILE NAME
    tweets_df = pd.read_csv(
        filename, usecols=['Content', 'Average Rating', 'User'])
    
    # format tweets to pass into model
    X_total = tweets_df['Content'].fillna(' ').values #ADDED THE .fillna(' ')
    y_total = tweets_df['Average Rating'].values

    num_tweets = len(y_total)
    split_idx = int(num_tweets * 0.8) 
    print("Total number of tweets:", num_tweets)
    print("Split index:", split_idx)

    X = X_total[:split_idx]
    y = y_total[:split_idx]
    X_test_final = X_total[split_idx:]
    y_test_final = y_total[split_idx:]

    true_best_model = None 
    max_accuracy = -1
    
    kfRepeat = RepeatedKFold(n_splits=2,n_repeats=2, random_state=1)
    # enumerate splits
    outer_results = list()

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=preprocess_tweet,
        ngram_range=(1, 3),
        min_df=5,  # only consider terms that occur in at least 5 tweets
        max_df=0.75  # only consider terms that occur in less than 75% of all tweets
    )
    best_params = dict()
    for train_ix, test_ix in kfRepeat.split(X_total):
        # split data
        beginning_train = train_ix[0]
        end_train = train_ix[len(train_ix)-1]
        beginning_test = test_ix[0]
        end_test = test_ix[len(test_ix)-1]

        X_train, X_test = X_total[beginning_train:end_train+1], X_total[beginning_test:end_test+1]
        y_train, y_test = y_total[beginning_train:end_train+1], y_total[beginning_test:end_test+1]
        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=2, shuffle=True, random_state=1)
        # define the model
        X_vectorized = vectorizer.fit_transform(X_train) 

        model = RandomForestClassifier(n_estimators=400, min_samples_leaf=1, max_features='sqrt', 
                                    max_depth=None, bootstrap=False, 
                                    class_weight='balanced')
        print("Passed rfc")
        # define search space
        space = dict()
        # space['n_estimators'] = [100, 200, 300, 400, 500] More thorough
        space['n_estimators'] = [100, 300, 500]

        #space['max_features'] = [2, 4, 6]
        # space['min_samples_leaf'] = [1, 2, 3, 4, 5] More thorough
        space['min_samples_leaf'] = [1, 2, 4]

        # define search


        #GridSearchCV is for finding the best hyperparameters
        search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
        print("passed search")
        # execute search
        result = search.fit(X_vectorized, y_train)
        print("passed result")

        # get the best performing model fit on the whole training set
        best_model = search.best_estimator_
        print("passed best_model")
        # evaluate model on the hold out dataset
        #We did fit_transform, it should have just been transform!
        X_test_vectorized = vectorizer.transform(X_test)
        print(search.cv_results_['params'][search.best_index_])
        yhat = best_model.predict(X_test_vectorized)
        
        # evaluate the model
        acc = accuracy_score(y_test, yhat)
        if (acc > max_accuracy):
            max_accuracy = acc
            # max_train_index = train_ix
            # max_test_index = test_ix
            best_params = search.cv_results_['params'][search.best_index_]
            true_best_model = best_model 
        # print("acc" + str(acc))
        # store the result
        outer_results.append(acc)
        # report progress
        print('>acc=%.7f, est=%.3f, cfg=%s' % (acc, search.best_score_, search.best_params_))
    # summarize the estimated performance of the model
    print('Final Average Accuracy Across K Folds: %.7f (%.3f)' % (mean(outer_results), std(outer_results)))
    # The mean tells us how well our current model perfoms on average
    # The std tells us how much the skill is expected to vary in practice
    
    print("best model performance on test set")
    if true_best_model:
        # X_test_final_vectorized = vectorizer.transform(X_test_final)
        # yhat = true_best_model.predict(X_test_final_vectorized)
        # acc = accuracy_score(y_test_final, yhat)
        # print(str(acc))
        #"ValueError: Number of features of the model must match the input. Model n_features is 7576 and input n_features is 7551"
        print("best params")
        print(best_params)


    #{'min_samples_leaf': 1, 'n_estimators': 300, 'max_features' = 6}                        
    
    # STILL TO DO:
    # Try higher K's for both the repeated kfold and kfold. I couldn't do higher k because
    # it took forever to run and takes up SO MUCH CPU


    # kf = KFold(n_splits = 2)
    # for train, test in kf.split(X): 
    #     beginning_train = train[0]
    #     end_train = train[len(train)-1]
    #     beginning_test = test[0]
    #     end_test = test[len(test)-1]
    #     X_train, X_test, y_train, y_test = X[beginning_train:end_train+1], X[beginning_test:end_test+1], y[beginning_train:end_train+1], y[beginning_test:end_test+1]
    #     print("Train index: " + str(train) + "\n Test index: " + str(test))
    #     train_clf(X_train, y_train, X_test, y_test, train, test)

    # print("max train index" + str(max_train_index))
    # print("max test index" + str (max_test_index))
    # beginning_train = max_train_index[0]
    # end_train = max_train_index[len(max_train_index)-1]
    # beginning_test = max_test_index[0]
    # end_test = max_test_index[len(max_test_index)-1]
    # X_train, X_test, y_train, y_test = X[beginning_train:end_train+1], X[beginning_test:end_test+1], y[beginning_train:end_train+1], y[beginning_test:end_test+1]
    # train_clf(X_train, y_train, X_test, y_test, [], [])
    # print("ends")

    



    


    # train_clf(X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:])