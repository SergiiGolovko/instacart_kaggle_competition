from logging import basicConfig, info, INFO
from numpy import arange, array_split
from os import makedirs
from os.path import realpath, join, split
from pandas import concat, merge, read_csv
from pandas import DataFrame

from pickle_utils import dump_data

# Directory paths.
BASE_DIR = split(realpath(__file__))[0]
DATA_DIR = join(BASE_DIR, 'data')
PICKLE_DIR = join(BASE_DIR, 'pickles')
ORDERS_DIR = join(PICKLE_DIR, 'orders')

# File paths.
ORDERS_FILE = join(DATA_DIR, 'orders.csv')
ORDER_PRODUCTS_PRIOR_FILE = join(DATA_DIR, 'order_products__prior.csv')
ORDER_PRODUCTS_TRAIN_FILE = join(DATA_DIR, 'order_products__train.csv')

# Global parameters.
NCHUNKS = 64


def split_data(nchunks=NCHUNKS):

    info('Splitting data into %d chunks.' % nchunks)

    # Read data.
    info('Reading data.')
    orders = read_csv(ORDERS_FILE)
    order_products_prior = read_csv(ORDER_PRODUCTS_PRIOR_FILE)
    order_products_train = read_csv(ORDER_PRODUCTS_TRAIN_FILE)

    # Merge / concatenate all orders together.
    info('Concatenating data.')
    orders_prior = orders[orders['eval_set'] == 'prior']
    orders_prior = merge(orders_prior, order_products_prior, on='order_id')
    orders_train = orders[orders['eval_set'] == 'train']
    orders_train = merge(orders, order_products_train, on='order_id')
    orders_test = orders[orders['eval_set'] == 'test'].copy()
    orders = concat([orders_prior, orders_train, orders_test])

    # User ids are sequential here, so we can split by chunks.
    info('Final split.')
    user_ids = arange(start=1, stop=orders.user_id.max() + 1)
    chunks = array_split(user_ids, nchunks)

    # Finally make splits.
    makedirs(ORDERS_DIR, exist_ok=True)
    for i, chunk in enumerate(chunks):
        info('Saving chunk #%d.' % i)
        chunk_df = DataFrame(chunk, columns=['user_id'])
        chunk_orders = merge(chunk_df, orders, on='user_id')
        sort_cols = ['user_id', 'order_number', 'add_to_cart_order']
        chunk_orders.sort_values(by=sort_cols, inplace=True)
        file = join(ORDERS_DIR, 'users_%s_%s.pckl' % (chunk[0], chunk[-1]))
        dump_data(chunk_orders, file)

    info('Data is split.')


if __name__ == '__main__':
    format = '%(asctime)s %(levelname)s %(filename)s %(funcName)s %(message)s'
    basicConfig(level=INFO, format=format, datefmt='%m/%d/%Y %I:%M:%S')
    split_data()
