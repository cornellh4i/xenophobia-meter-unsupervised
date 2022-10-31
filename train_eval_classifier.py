from typing import List


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


def train_clf(train_tweets: List[str], train_labels: List[int], 
              test_tweets: List[str], test_labels: List[int], 
              classifier_filename: str = None,
              vectorizer_filename: str = None, 
              error_log_filename: str = None) -> None:
    """
    Trains a classifier on `train_tweets` and 
      saves the vectorizer and model to disk if respective filenames are passed in.
    Prints out the classification report/scores of the trained classifier on `test_tweets`.
    Also plots the confusion matrix and, if a filename is provided, misprediction information 
      is written to another file.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, plot_confusion_matrix
    import numpy as np
    from joblib import dump
    from matplotlib import pyplot as plt
    
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=preprocess_tweet,
        ngram_range=(1, 3),
        min_df=2,  # only consider terms that occur in at least 5 tweets
        max_df=0.65  # only consider terms that occur in less than 75% of all tweets
    )

    # train
    X_vectorized = vectorizer.fit_transform(train_tweets)
    clf = RandomForestClassifier(n_estimators=100, min_samples_split=1.0,
                                 min_samples_leaf=1, max_features='sqrt', 
                                 max_depth=None, bootstrap=False, 
                                 class_weight='balanced')
    clf.fit(X_vectorized, train_labels)

    # test and print classification report
    X_test_vectorized = vectorizer.transform(test_tweets)
    y_pred = clf.predict(X_test_vectorized)
    clf_report = classification_report(test_labels, y_pred)
    print(clf_report)
    
    if error_log_filename:
        with open(error_log_filename, 'w') as f:
            for i in range(len(y_pred)):
                if y_pred[i] != test_labels[i]:
                    f.write("***\n")
                    f.write(f"EXPECTED: {test_labels[i]}, PREDICTED: {y_pred[i]}\n")
                    f.write(f"TWEET: {test_tweets[i]}\n\n")
        print(f"Wrote incorrectly predicted tweets to {error_log_filename}.")

    
    # confusion matrix
    plot_confusion_matrix(clf, X_test_vectorized, test_labels)
    plt.show()
    
    
    # persist classifier and vectorizer
    MODEL_ROOT_DIR = os.path.join('..', 'models')

    if classifier_filename:
        full_clf_filename = os.path.join(MODEL_ROOT_DIR, f"{classifier_filename}.joblib")
        print(f"Saving classifier to {full_clf_filename}")
        dump(clf, full_clf_filename)
        
    if vectorizer_filename:
        full_vect_filename = os.path.join(MODEL_ROOT_DIR, f"{vectorizer_filename}.joblib")
        print(f"Saving vectorizer to {full_vect_filename}")
        dump(vectorizer, full_vect_filename)


if __name__ == '__main__':
    import os
    import pandas as pd

    # read tweets
    filename = os.path.join('..', 'data', 'out',
                            'test_clean_updated.csv')
    tweets_df = pd.read_csv(
        filename, usecols=['Content', 'Average Rating(0.5)', 'User'])
    
    # format tweets to pass into model
    X = tweets_df['Content'].fillna(' ').values
    y = tweets_df['Average Rating(0.5)'].values
    
    num_tweets = len(y)
    split_idx = int(num_tweets * 0.7)
    print("Total number of tweets:", num_tweets)
    print("Split index:", split_idx)
    
    train_clf(X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:])
