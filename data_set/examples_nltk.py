from nltk.corpus import movie_reviews
import random
from data_set import utils
import nltk


def document_classification():
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    random.shuffle(documents)
    all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
    word_features = list(all_words)[:2000]
    feature_sets = [(utils.document_features(d, word_features), c) for (d, c) in documents]
    train_set, test_set = feature_sets[50:], feature_sets[:50]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)