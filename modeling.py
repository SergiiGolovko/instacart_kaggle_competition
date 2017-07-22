from logging import info, debug, basicConfig, INFO
from os.path import realpath, join, split
from os import makedirs

from cv_utils import get_cv
from globals import CONFIG
from pickle_utils import dump_data, try_load
from model_utils import get_classifiers, get_param_grids
from model_utils import cross_validation, tune_parameters, fit_and_predict

# Directory paths.
BASE_DIR = split(realpath(__file__))[0]
DATA_DIR = join(BASE_DIR, 'data')
PICKLE_DIR = join(BASE_DIR, 'pickles')
METAFEATURES_DIR = join(PICKLE_DIR, 'metafeatures')
RAW_PREDICTIONS_DIR = join(PICKLE_DIR, 'raw_predictions')

# Other global parameters.
MODELLING_MODE = 'single_run'
N_FOLDS = 5


def single_run(clf, X_train, X_test, y_train, name, cv):
    cross_validation(
        clf, X_train, y_train, cv, filename=join(METAFEATURES_DIR, name))
    y_test_pred_proba = fit_and_predict(clf, X_train, y_train, X_test)
    dump_data(y_test_pred_proba, join(RAW_PREDICTIONS_DIR, name))


def tune_params_run(clf, X_train, X_test, y_train, name, param_grid, cv):
    tune_parameters(clf, name, param_grid, X_train, y_train, cv)
    single_run(clf, X_train, X_test, y_train, cv)


# TODO: implement this function
def multiple_run(clf, X_train, X_test, y_train, name, param_grid, cv):
    return


def modelling(features_file, mode=MODELLING_MODE):

    # 1. Load data from the file and create train/test sets.
    features = try_load(features_file, raise_error=True)
    train_inds = features['train_set'] == 1
    features[features['user_id'] == 1].to_csv('features.csv')
    drop_cols = ['train_set', 'user_id', 'label']
    feat_cols = [c for c in features.columns if c not in drop_cols]
    X_train = features[feat_cols][train_inds].values
    X_test = features[feat_cols][~train_inds].values
    y_train = features['label'][train_inds].values

    # 2. Print some useful information.
    info('Shape of train set: ' + str(X_train.shape))
    info('Shape of test set: ' + str(X_test.shape))
    info('Number of pos cases in train set: %d' % sum(y_train == 1))
    info('Number of neg cases in train set: %d' % sum(y_train == 0))
    info('Accuracy of naive model: %.3f'
         % max((y_train == 0).mean(), (y_train == 1).mean()))

    # 2. Get classifiers and parameters.
    names = ['XGBClassifier']
    estimators = get_classifiers(names)
    par_grids = get_param_grids(names)
    cv = get_cv(y_train, N_FOLDS, type='kfold')

    # 3. Depending on the mode estimate models.
    for (name, clf) in zip(names, estimators):
        if mode == 'single_run':
            single_run(clf, X_train, X_test, y_train, name, cv)
        if mode == 'tune_params':
            tune_params_run(
                    clf, X_train, X_test, y_train, name, param_grid, cv)
        if mode == 'multiple_run':
            tune_params_run(
                    clf, X_train, X_test, y_train, name, param_grid, cv)


if __name__ == '__main__':
    format = '%(asctime)s %(levelname)s %(filename)s %(funcName)s %(message)s'
    basicConfig(level=INFO, format=format, datefmt='%m/%d/%Y %I:%M:%S')
    makedirs(METAFEATURES_DIR, exist_ok=True)
    makedirs(RAW_PREDICTIONS_DIR, exist_ok=True)
    if CONFIG['CONFIG'] == 'config_test':
        modelling('pickles/features_1_3223.pckl')
