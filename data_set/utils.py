import numpy
import matplotlib.pyplot as plt
from sklearn import linear_model


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


def get_regression_performance(data, features, test_set_size, model=None):
    model = model if model is not None else linear_model.LinearRegression()
    feature_data = data.data[:, features]
    x_train, x_test, y_train, y_test = split_train_test_data(
        feature_data, data.target, test_set_size=test_set_size)
    # Train the model using the training sets
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)


def plot_data_feature_regression(data, feature, test_set_size, model=None):
    model = model if model is not None else linear_model.LinearRegression()
    feature_data = data.data[:, numpy.newaxis, feature]
    x_train, x_test, y_train, y_test = split_train_test_data(
        feature_data, data.target, test_set_size=test_set_size)
    # Train the model using the training sets
    model.fit(x_train, y_train)
    plt.scatter(x_test, y_test, color='black')
    plt.plot(x_test, model.predict(x_test), color='blue',
             linewidth=3)

    plt.xticks(())
    plt.yticks(())
    text = '{0}\n{1}'.format(
        'Coefficients of feature {0}: {1}'.format(feature, model.coef_),
        'Variance Score {0}'.format(model.score(x_test, y_test))
    )
    plt.text(0, -10, text)
    plt.show()
    return model.score(x_test, y_test)
