from numpy import array
from unittest import TestCase

from generate_predictions import mean_F1_score


class MeanF1ScoreTest(TestCase):

    def test_single_user_precision1(self):
        y_true = array([1, 1, 1, 1])
        y_pred_proba = array([0.51, 0.49, 0.51, 0.49])
        user_id = array([1, 1, 1, 1])
        self.assertEqual(mean_F1_score(y_true, y_pred_proba, user_id, 0.5),
                         2/3)

    def test_single_user_F1_0(self):
        y_true = array([1, 1, 0, 0])
        y_pred_proba = array([0.49, 0.49, 0.51, 0.51])
        user_id = array([1, 1, 1, 1])
        self.assertEqual(mean_F1_score(y_true, y_pred_proba, user_id, 0.5), 0)

    def test_nones(self):
        y_true = array([0, 0, 0, 0])
        y_pred_proba = array([0.49, 0.49, 0.49, 0.49])
        user_id = array([1, 1, 1, 1])
        self.assertEqual(mean_F1_score(y_true, y_pred_proba, user_id, 0.5), 1)

    def test_y_true_none(self):
        y_true = array([0, 0, 0, 0])
        y_pred_proba = array([0.49, 0.49, 0.49, 0.51])
        user_id = array([1, 1, 1, 1])
        self.assertEqual(mean_F1_score(y_true, y_pred_proba, user_id, 0.5), 0)

    def test_precision_recall(self):
        y_true = array([0, 0, 1, 1])
        y_pred_proba = array([0.51, 0.49, 0.49, 0.51])
        user_id = array([1, 1, 1, 1])
        self.assertEqual(mean_F1_score(y_true, y_pred_proba, user_id, 0.5),
                         0.5)
