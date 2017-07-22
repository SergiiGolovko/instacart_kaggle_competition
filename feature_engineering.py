from logging import basicConfig, info, INFO
from numpy import array
from os.path import realpath, join, split
from pandas import concat, read_csv, merge, DataFrame

from pickle_utils import dump_data, try_load
from multiprocess_utils import multiprocess_from_folder
from globals import CONFIG

# Directory paths.
BASE_DIR = split(realpath(__file__))[0]
DATA_DIR = join(BASE_DIR, 'data')
PICKLE_DIR = join(BASE_DIR, 'pickles')
ORDERS_DIR = join(PICKLE_DIR, 'orders')

# File paths.
DEPARTMENTS_FILE = join(DATA_DIR, 'departments.csv')
AISLES_FILE = join(DATA_DIR, 'aisles.csv')
PRODUCTS_FILE = join(DATA_DIR, 'products.csv')
PRODUCTS_INFO_FILE = join(PICKLE_DIR, 'product_info.pckl')
BASIC_FEATURES_FILE = join(PICKLE_DIR, 'basic_features.pckl')


def features_by_product(data):

    # Target variable, whether product was reordered at the last order.
    label = data['reordered'].iloc[-1]

    # Number of orders and total orders.
    total_orders = data['order_number'].iloc[-1] - 1
    num_orders = len(data) - 1

    # last_time_ordered, is_prev_order, is_before_prev_order
    last_time_ordered = data['order_number'].iloc[-2] if num_orders > 0 else -1
    is_prev_order = last_time_ordered == total_orders
    if (last_time_ordered == total_orders - 1) and (total_orders > 0):
        is_before_prev_order = True
    elif num_orders > 1:
        is_before_prev_order = data['order_number'].iloc[-3] == total_orders - 1
    else:
        is_before_prev_order = False

    # Number of times the order was reordered from previous time.
    num_reordered = (
        sum(data['order_number'].iloc[:-1] + 1 == data['order_number'].iloc[1:]))
    num_reordered = num_reordered - is_prev_order

    # Eval set.
    train_set = data['eval_set'].iloc[-1] == 'train'

    return [int(label), total_orders, num_orders, num_reordered, last_time_ordered,
            int(is_prev_order), int(is_before_prev_order), train_set]


def feature_by_product_names():
    return ['label', 'total_orders', 'num_orders', 'num_reordered',
            'last_time_ordered', 'is_prev_order', 'is_before_prev_order',
            'train_set']


def basic_features(orders_file):

    info('Creating basic features.')

    orders = try_load(orders_file, raise_error=True)
    train_inds = (orders['eval_set'] == 'train')

    # Drop not reordered items for train set.
    drop_ind = (orders['eval_set'] == 'train') & (orders['reordered'] == 0)
    orders = orders[~drop_ind]

    # Add to train set all products that were previously ordered and label
    # them as being not reordered.
    cols = ['user_id', 'product_id', 'order_number', 'eval_set']
    user_prod_order = orders[cols].drop_duplicates(
            subset=['user_id', 'product_id'], keep='last')
    user_prod_order['max_order_number'] = (
            user_prod_order.groupby('user_id')['order_number'].transform(max))
    user_prod_order['last_eval_set'] = (
            user_prod_order.groupby('user_id')['eval_set'].transform('last'))
    drop_ind = (user_prod_order.max_order_number == user_prod_order.order_number)
    user_prod_order = user_prod_order[~drop_ind]
    user_prod_order['reordered'] = 0
    user_prod_order['order_number'] = user_prod_order['max_order_number']
    user_prod_order['eval_set'] = user_prod_order['last_eval_set']
    user_prod_order.drop(
        ['max_order_number', 'last_eval_set'], axis=1, inplace=True)
    orders = concat([orders, user_prod_order])

    # Drop test set with reordered null.
    drop_ind = (orders['eval_set'] == 'test') & (orders['reordered'].isnull())
    orders = orders[~drop_ind]

    # Create features.
    gr_orders = orders.groupby(['user_id', 'product_id'])
    gr_orders = gr_orders.apply(lambda x: features_by_product(x)).reset_index()

    # Generate result
    res = DataFrame(array(list(gr_orders.loc[:, 0].values)),
                    columns=feature_by_product_names())
    res['user_id'] = gr_orders['user_id']
    res['product_id'] = gr_orders['product_id']
    # res['eval_set'] = gr_orders['eval_set']

    # Add department and aisle id.
    product_info = get_products_info()
    cols = ['product_id', 'aisle_id', 'department_id']
    res = merge(res, product_info[cols], on='product_id')

    info('Basic features are created.')

    return res


def get_products_info():

    data = try_load(PRODUCTS_INFO_FILE)
    if data is not None:
        info('Using previously created product info.')
        return data

    products = read_csv(PRODUCTS_FILE)
    departments = read_csv(DEPARTMENTS_FILE)
    aisles = read_csv(AISLES_FILE)

    data = merge(products, departments, on='department_id')
    data = merge(data, aisles, on='aisle_id')

    dump_data(data, PRODUCTS_INFO_FILE)
    return data


if __name__ == '__main__':
    format = '%(asctime)s %(levelname)s %(filename)s %(funcName)s %(message)s'
    basicConfig(level=INFO, format=format, datefmt='%m/%d/%Y %I:%M:%S')
    if CONFIG['CONFIG'] == 'config_normal':
        data = multiprocess_from_folder(basic_features, 'orders_file', ORDERS_DIR)
        dump_data(data, BASIC_FEATURES_FILE)
    else:   # 'config_test'
        data = basic_features('pickles/orders/users_1_3223.pckl')
        dump_data(data, 'pickles/features_1_3223.pckl')
