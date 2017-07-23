from logging import basicConfig, info, debug, DEBUG
from numpy import array, concatenate, zeros
from pandas import DataFrame

from pickle_utils import try_load


def mean_F1_score(y_true, y_pred_proba, user_id, threshold=0.5):

    info('Calculating mean F1 score.')

    y_true = array(y_true).reshape((len(y_true), 1))
    y_pred_proba = array(y_pred_proba).reshape((len(y_pred_proba), 1))
    user_id = array(user_id).reshape((len(user_id), 1))

    y_pred = (y_pred_proba > threshold).astype(int)
    cols = ['user_id', 'y_true', 'y_pred']
    df = DataFrame(concatenate([user_id, y_true, y_pred], axis=1),
                   columns=cols)

    gdf = df.groupby('user_id')
    cols = ['n_pos', 'n_true_pos', 'n_pos_pred', 'precision', 'recall', 'F1']
    res_df = DataFrame(zeros((len(gdf), 6)), columns=cols,
                       index=gdf['user_id'].last().index)
    res_df['n_pos'] = gdf['y_true'].sum()
    res_df['n_pos_pred'] = gdf['y_pred'].sum()
    res_df['n_true_pos'] = gdf.apply(lambda x: sum(x.y_true * x.y_pred))
    res_df['precision'] = res_df.apply(
        lambda x: x.n_true_pos / max(x.n_pos_pred, 1)
        if not ((x.n_pos == 0) and (x.n_pos_pred == 0)) else 1,
        axis=1)
    res_df['recall'] = res_df.apply(
        lambda x: x.n_true_pos / max(x.n_pos, 1)
        if not ((x.n_pos == 0) and (x.n_pos_pred == 0)) else 1, axis=1)
    res_df['F1'] = res_df.apply(
        lambda x: 2 * (x.precision * x.recall) / (x.precision + x.recall)
        if x.precision + x.recall > 0 else 0,
        axis=1)
    info('Mean F1 score is calculated.')

    return res_df['F1'].mean()


def evaluate_predictions(pred_file, features_file):

    info('Evaluating predictions.')

    info('Loading data.')
    y_pred_proba = try_load(pred_file, raise_error=True)
    wanted_cols = ['user_id', 'label', 'train_set']
    df = try_load(features_file, raise_error=True)[wanted_cols]
    df = df[df['train_set'] == 1]
    debug('Data size: %d, unique users_id: %d'
          % (len(df), df['user_id'].nunique()))
    info('Finished loading data.')
    debug('Scoring predictions')
    score = mean_F1_score(df['label'], y_pred_proba, df['user_id'], 0.05)
    debug('Score is: %.5f' % score)
    info('Finished evaluating predictions.')


if __name__ == '__main__':

    format = '%(asctime)s %(levelname)s %(filename)s %(funcName)s %(message)s'
    basicConfig(level=DEBUG, format=format, datefmt='%m/%d/%Y %I:%M:%S')
    features_file = './pickles/features_1_3223.pckl'
    pred_file = './pickles/metafeatures/XGBClassifier'
    evaluate_predictions(pred_file, features_file)
