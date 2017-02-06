import numpy


def split_train_test_data(x_data, y_data, test_percent=None, test_set_size=None):
    indices = numpy.random.permutation(len(x_data))
    test_set_size = int(len(x_data) * test_percent) if test_percent is not None else test_set_size
    return (x_data[indices[test_set_size:]], x_data[indices[:test_set_size]],
            y_data[indices[test_set_size:]], y_data[indices[:test_set_size]])


def lexical_diversity(text):
    return len(set(text)) / len(text)


def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features