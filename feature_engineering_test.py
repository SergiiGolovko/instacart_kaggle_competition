from pandas import DataFrame
from unittest import TestCase

from feature_engineering import features_by_product


class FeaturesByProductTest(TestCase):

    def test_single_order(self):
        cols = ['order_number', 'reordered']
        data = DataFrame([[2, 0]], columns=cols)
        # label, total_orders, num_orders, last_time_ordered, is_prev_order,
        # is_before_prev_order]
        exp_res = [0, 1, 0, -1, 0, 0]
        res = features_by_product(data)
        self.assertEqual(exp_res, res)

    def test_two_consequitive_orders(self):
        cols = ['order_number', 'reordered']
        data = DataFrame([[2, 0], [3, 1]], columns=cols)
        # label, total_orders, num_orders, last_time_ordered, is_prev_order,
        # is_before_prev_order]
        exp_res = [1, 2, 1, 2, 1, 0]
        res = features_by_product(data)
        self.assertEqual(exp_res, res)

    def test_two_nonconsequitive_orders(self):
        cols = ['order_number', 'reordered']
        data = DataFrame([[1, 0], [3, 0]], columns=cols)
        # label, total_orders, num_orders, last_time_ordered, is_prev_order,
        # is_before_prev_order]
        exp_res = [0, 2, 1, 1, 0, 1]
        res = features_by_product(data)
        self.assertEqual(exp_res, res)

    def test_three_consequitive_orders(self):
        cols = ['order_number', 'reordered']
        data = DataFrame([[1, 0], [2, 1], [3, 1]], columns=cols)
        # label, total_orders, num_orders, last_time_ordered, is_prev_order,
        # is_before_prev_order
        exp_res = [1, 2, 2, 2, 1, 1]
        res = features_by_product(data)
        self.assertEqual(exp_res, res)
