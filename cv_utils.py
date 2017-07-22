"""
__file__

    cv_utils.py

__description__

    This file contains cross validation utils.

__author__

    Sergii Golovko < sergii.golovko@gmail.com >
"""

from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.cross_validation import StratifiedShuffleSplit
import random

# Global cv parameters.
N_FOLDS = 2
VAL_SIZE = 0.2
RANDOM_SEED = 8888


def get_cv(y, n_folds=N_FOLDS, type='split'):
    '''
    :param
    :return:
    '''

    random.seed(RANDOM_SEED)

    if type == 'split':
        cv = StratifiedShuffleSplit(y, n_iter=n_folds, test_size=VAL_SIZE,
                                    train_size=None,
                                    random_state=RANDOM_SEED)
    elif type == 'kfold':
        cv = StratifiedKFold(y, n_folds=n_folds, shuffle=True,
                             random_state=RANDOM_SEED)
    else:
        raise ValueError('Unknown cv type!')

    return cv
