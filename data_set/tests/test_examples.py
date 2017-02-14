from unittest import TestCase
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from data_set import utils


class ExamplesTest(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass


    def test_performace(self):
        diabetes = datasets.load_diabetes()
        self.assertEqual(0, utils.get_regression_performance(diabetes, [1], 70))

    def test_linear_regression_single_feature(self):
        return
        diabetes = datasets.load_diabetes()
        performances = {}
        for feature in range(len(diabetes.data[0])):
            performances[feature] = utils.plot_data_feature_regression(diabetes, feature, 70)
        texts = ['{0} = {1}'.format(feature, performace) for feature, performace in performances.items()]
        text = ','.join(texts)
        plt.legend((0, 0), text)
        plt.show()
