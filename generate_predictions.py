from logging import basicConfig, info, debug, DEBUG
from numpy import array, concatenate, zeros, argmax
from pandas import DataFrame, merge, read_csv

from globals import CONFIG
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
    debug('Mean F1 score: %.3f' % res_df['F1'].mean())
    info('Mean F1 score is calculated.')

    return res_df['F1'].mean()


def evaluate_predictions(pred_file, features_file, thresholds, output_file):

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
    scores = [mean_F1_score(df['label'], y_pred_proba, df['user_id'], th/100)
              for th in thresholds]
    res = DataFrame({'threshold': thresholds, 'score': scores})
    res.to_csv(output_file, index=False)
    debug('Optimal threshold is: %d' % thresholds[argmax(array(scores))])
    debug('Optimal score is: %.3f' % scores[argmax(array(scores))])
    info('Finished evaluating predictions.')


def generate_predictions(
        pred_file, features_file, orders_file, output_file, threshold):

    info('Generating predictions.')

    # Load data.
    y_pred_proba = try_load(pred_file, raise_error=True)
    wanted_cols = ['user_id', 'train_set', 'product_id']
    df = try_load(features_file, raise_error=True)[wanted_cols]
    df = df[df['train_set'] == 0][['user_id', 'train_set', 'product_id']]
    df['y'] = y_pred_proba > threshold
    orders = read_csv(orders_file, usecols=['user_id', 'order_id', 'eval_set'])
    orders = orders[orders['eval_set'] == 'test']
    df = merge(df, orders[['order_id', 'user_id']], on='user_id')
    df.drop(['user_id', 'train_set'], 1, inplace=True)

    # Geerate predictions.
    df['y'] = df['y'] > threshold
    res = df.groupby('order_id').apply(lambda x: set(x.loc[x.y, 'product_id']))
    res = res.reset_index()
    res.rename(columns={0: 'product_id'}, inplace=True)
    res['product_id'] = res['product_id'].apply(
            lambda s: ' '.join({str(int(el)) for el in s}) if len(s) > 0
            else 'None')
    res[['order_id', 'product_id']].to_csv(output_file, index=False)
    info('Finished generate_predictions')


if __name__ == '__main__':

    format = '%(asctime)s %(levelname)s %(filename)s %(funcName)s %(message)s'
    basicConfig(level=DEBUG, format=format, datefmt='%m/%d/%Y %I:%M:%S')
    if CONFIG['CONFIG'] == 'config_test':
        features_file = './pickles/features_1_3223.pckl'
    else:
        features_file = './pickles/basic_features.pckl'
    train_pred_file = './pickles/metafeatures/XGBClassifier'
    test_pred_file = './pickles/raw_predictions/XGBClassifier'
    orders_file = './data/orders.csv'
    output_file = './output/predictions.csv'
    statistics_file = './statistics/thresholds.csv'
    evaluate_predictions(
            train_pred_file, features_file, range(15, 50), statistics_file)
    generate_predictions(
            test_pred_file, features_file, orders_file, output_file, 0.23)
