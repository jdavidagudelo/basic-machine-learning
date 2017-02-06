from sklearn.model_selection import KFold, cross_val_score
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from data_set import utils
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import cluster
import numpy
import inspect


# The iris data an example of CLASSIFICATION
def classify_irises(test_set_size=10):
    iris = datasets.load_iris()
    x_iris = iris.data
    y_iris = iris.target
    x_iris_train, x_iris_test, y_iris_train, y_iris_test = utils.split_train_test_data(
        x_iris, y_iris, test_set_size=test_set_size)
    knn = KNeighborsClassifier()
    knn.fit(x_iris_train, y_iris_train)
    print(knn.score(x_iris_test, y_iris_test))
    # using regression for classification
    logistic = linear_model.LogisticRegression(C=1e5)
    logistic.fit(x_iris_train, y_iris_train)
    print(logistic.score(x_iris_test, y_iris_test))
    # using svm
    svc = svm.SVC(kernel='linear')
    svc.fit(x_iris_train, y_iris_train)
    print(svc.score(x_iris_test, y_iris_test))
    svc = svm.SVC(kernel='poly', degree=3)
    svc.fit(x_iris_train, y_iris_train)
    print(svc.score(x_iris_test, y_iris_test))
    svc = svm.SVC(kernel='rbf')
    svc.fit(x_iris_train, y_iris_train)
    print(svc.score(x_iris_test, y_iris_test))


def cluster_k_means():
    iris = datasets.load_iris()
    x_iris = iris.data
    y_iris = iris.target
    k_means = cluster.KMeans(n_clusters=3)
    k_means.fit(x_iris)
    print(k_means.labels_[::10])
    print(y_iris[::10])


def cross_validation_tests():
    models = [KNeighborsClassifier(),
              linear_model.LogisticRegression(C=1e5),
              svm.SVC(kernel='linear'), svm.SVC(kernel='rbf'),
              svm.SVC(kernel='poly', degree=3)]
    for model in models:
        cross_validation(model)


def grid_search_svm_tests():
    models = [svm.SVC(kernel='linear'), svm.SVC(kernel='rbf'),
              svm.SVC(kernel='poly', degree=3)]
    for model in models:
        grid_search_svm(model)


def cross_validated_estimators_tests():
    models = [linear_model.ElasticNetCV(), linear_model.LarsCV(),
              linear_model.LassoCV(), linear_model.LassoLarsCV(),
              linear_model.LogisticRegressionCV(),
              linear_model.OrthogonalMatchingPursuitCV(),
              linear_model.RidgeClassifierCV(), linear_model.RidgeCV()]
    for model in models:
        cross_validated_estimators(model)


def cross_validated_estimators(estimator):
    """
    Way to estimate parameters of linear estimator.
    :param estimator:
    :return:
    """
    diabetes = datasets.load_digits()
    x_diabetes = diabetes.data
    y_diabetes = diabetes.target
    fit = estimator.fit(x_diabetes, y_diabetes)
    for member in inspect.getmembers(fit):
        if member[0].endswith('_'):
            value = getattr(fit, member[0], '')
            if isinstance(value, (float, int)):
                print('{0} = {1}'.format(member[0], value))
    print()


def grid_search_svm(svm_classifier):
    iris = datasets.load_digits()
    x_digits = iris.data
    y_digits = iris.target
    cs = numpy.logspace(-6, -1, 10)
    grid = GridSearchCV(estimator=svm_classifier, param_grid=dict(C=cs),
                        n_jobs=-1)
    validation = cross_val_score(grid, x_digits, y_digits)
    print(validation)
    grid.fit(x_digits, y_digits)
    print(grid.best_score_)
    print(grid.best_estimator_.C)


def cross_validation(classifier):
    k_fold = KFold(n_splits=3)
    iris = datasets.load_digits()
    x_digits = iris.data
    y_digits = iris.target
    validation = [classifier.fit(x_digits[train], y_digits[train]).score(x_digits[test], y_digits[test])
                  for train, test in k_fold.split(x_digits)]
    print(validation)
    # The same
    validation = cross_val_score(classifier, x_digits, y_digits, cv=k_fold, n_jobs=-1)
    print(validation)


def linear_regression_diabetes(test_set_size=-20):
    alphas = numpy.logspace(-4, -1, 6)
    diabetes = datasets.load_diabetes()
    x_diabetes = diabetes.data
    y_diabetes = diabetes.target
    x_diabetes_train, x_diabetes_test, y_diabetes_train, y_diabetes_test = utils.split_train_test_data(
        x_diabetes, y_diabetes, test_set_size=test_set_size)
    regression = linear_model.LinearRegression()
    print([regression.fit(x_diabetes_train,
                          y_diabetes_train).score(x_diabetes_test, y_diabetes_test) for alpha in alphas])
    regression = linear_model.Ridge()
    print([regression.set_params(
        alpha=alpha).fit(x_diabetes_train,
                         y_diabetes_train).score(x_diabetes_test, y_diabetes_test) for alpha in alphas])
    regression = linear_model.Lasso()
    print([regression.set_params(
        alpha=alpha).fit(x_diabetes_train,
                         y_diabetes_train).score(x_diabetes_test, y_diabetes_test) for alpha in alphas])
    regression = linear_model.LassoLars()
    print([regression.set_params(
        alpha=alpha).fit(x_diabetes_train,
                         y_diabetes_train).score(x_diabetes_test, y_diabetes_test) for alpha in alphas])
