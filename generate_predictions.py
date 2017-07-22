from logging import info
from numpy import array, concatenate, zeros
from pandas import DataFrame


def mean_F1_score(y_true, y_pred_proba, user_id, threshold=0.5):

    info('Calculating mean F1 score.')

    y_true = array(y_true).reshape((len(y_true), 1))
    y_pred_proba = array(y_pred_proba).reshape((len(y_pred_proba), 1))
    user_id = array(user_id).reshape((len(y_pred_proba), 1))

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
    info('Mean F1 score is created.')

    return res_df['F1'].mean()
