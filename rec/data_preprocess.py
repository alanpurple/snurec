""" Wemakeprice Recommendation Project.

Authors:
- Hyunsik Jeon (jeon185@snu.ac.kr)
- Jaemin Yoo (jaeminyoo@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.

File: rec/data_preprocess.py
- Preprocess aggregated data to generate instances.

Version: 1.0.0

"""
import fnmatch
import time
from datetime import datetime, timedelta, timezone

import click
import pandas as pd
import tqdm

from rec.utils import *

# Global variables for efficient programming.
MIN_ORDERS = None  # determined at main()
MAX_ORDERS = None  # determined at main()
MAX_CLICKS = None  # determined at main()


class LookupTable:
    """
    Class that stores items and user information.
    """

    def __init__(self, emb_path, all_items, all_users):
        """
        Class initializer.

        :param emb_path: a path to an embedding directory.
        :param all_items: a list of all items to use.
        :param all_users: a lost of all users to use.
        """
        id_path = os.path.join(emb_path, 'ids.tsv')
        vec_path = os.path.join(emb_path, 'vectors.tsv')
        title_path = os.path.join(emb_path, 'titles.tsv')

        item_ids = np.loadtxt(id_path, dtype=int)
        item_dict = {e: i for i, e in enumerate(item_ids)}
        index = np.array([item_dict[x] for x in all_items])
        item_titles = [e.strip() for e in open(title_path).readlines()]

        self.item_ids = all_items
        self.item_dict = {e: i for i, e in enumerate(all_items)}
        self.item_titles = np.array(item_titles)[index]
        self.item_vectors = np.loadtxt(vec_path)[index]
        self.item_categories = None

        self.user_ids = all_users
        self.user_dict = {e: i for i, e in enumerate(all_users)}

    def set_categories(self, path):
        """
        Set category information.

        :param path: a path of a product-category CSV file.
        :return: None.
        """
        items = set(self.get_items())
        df = pd.read_csv(path)
        df = df[df[PROD_NO].apply(lambda x: x in items)]

        category_list = df[CATEGORY_ID].unique()
        category_list.sort()
        category_dict = {e: i for i, e in enumerate(category_list)}
        df[CATEGORY_IDX] = df[CATEGORY_ID].map(category_dict)
        df.drop(columns=[CATEGORY_ID], inplace=True)

        category_shape = len(items), len(category_list)
        category_vectors = np.zeros(category_shape, dtype=np.uint8)
        for item_id, df_ in df.groupby(by=PROD_NO):
            item_idx = self.item_to_index(item_id)
            category_vectors[item_idx, df_[CATEGORY_IDX]] = 1
        self.item_categories = category_vectors

    def get_items(self):
        """
        Return the list of all items.

        :return: a NumPy array.
        """
        return self.item_ids

    def item_to_index(self, item_id):
        """
        Convert a unique item ID to an integer index.

        :param item_id: an item ID.
        :return: the corresponding index.
        """
        if item_id in self.item_dict:
            return self.item_dict[item_id]
        else:
            return -1

    def user_to_index(self, user_id):
        """
        Convert a unique user ID to an integer index.

        :param user_id: a user ID.
        :return: the corresponding index.
        """
        if user_id in self.user_dict:
            return self.user_dict[user_id]
        else:
            return -1

    def save_users(self, path):
        """
        Save the user information as a file.

        :param path: the path to a directory to save a file in.
        :return: None.
        """
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, 'ids'), self.user_ids)

    def save_items(self, path):
        """
        Save the item information as a file.

        :param path: the path to a directory to save files in.
        :return: None.
        """
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, 'ids'), self.item_ids)
        np.save(os.path.join(path, 'titles'), self.item_titles)
        np.save(os.path.join(path, 'embeddings'), self.item_vectors)
        np.save(os.path.join(path, 'categories'), self.item_categories)


def get_timestamps(version):
    """
    Get timestamps to split data into training, validation and test.

    :param version: a version of timestamps to return.
    :return: a pair of timestamps.
    """
    if version == '1w':
        return 1554476400, 1554562800
    elif version == '2w':
        return 1554908400, 1555081200
    elif version == '6w':
        return 1554044400, 1554649200
    else:
        raise ValueError(version)


def split_orders(df, ts_val, ts_test, column=TIMESTAMP):
    """
    Split a DataFrame of orders based on given timestamps.

    :param df: the DataFrame of orders.
    :param ts_val: a starting timestamp for validation data.
    :param ts_test: a starting timestamp for test data.
    :param column: a target column to split rows.
    :return: a tuple of training, validation, and test DataFrames.
    """
    df_trn = df.loc[df[column] < ts_val, :]
    df_val = df.loc[df[column] < ts_test, :]
    df_test = df.copy()

    df_trn.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    return df_trn, df_val, df_test


def read_behaviors(path, cols=None):
    """
    Read a DataFrame of behaviors (clicks or orders).

    :param path: the path to the DataFrame.
    :param cols: a list of columns to read.
    :return: the loaded DataFrame.
    """
    dtype = {COMP_MID: str, PROD_NO: int, TIMESTAMP: int, PROD_NM: str}
    return pd.read_csv(path, delimiter=',', dtype=dtype, usecols=cols)


def set_parameters(min_orders, max_orders, max_clicks):
    """
    Set global parameters of this script.

    :param min_orders: the minimum number of orders in each instance.
    :param max_orders: the maximum number of orders in each instance.
    :param max_clicks: the maximum number of clicks in each instance.
    :return: None.
    """
    assert min_orders <= max_orders
    global MIN_ORDERS, MAX_ORDERS, MAX_CLICKS
    MIN_ORDERS = min_orders
    MAX_ORDERS = max_orders
    MAX_CLICKS = max_clicks


def filter_blacklist(df, path):
    """
    Exclude some items from a DataFrame.

    :param df: a DataFrame of clicks or orders.
    :param path: the path to a file containing items to exclude.
    :return: None.
    """
    f = open(path)
    patterns = [e.strip() for e in f.readlines()]
    f.close()

    def run_filter(x):
        for pat in patterns:
            if fnmatch.fnmatch(x, pat):
                return True
        return False

    index = df[df[PROD_NM].apply(run_filter)].index
    df.drop(columns=[PROD_NM], index=index, inplace=True)


def filter_by_occurrences(df, column, cut):
    """
    Exclude some entries (items or users) based on their occurrences.

    :param df: a DataFrame of clicks or orders.
    :param column: a column to exclude entries from.
    :param cut: the minimum threshold of occurrences.
    :return: a list of entries to use.
    """
    counts = df[column].value_counts(sort=False)
    all_entries = counts[counts >= cut].index.values
    all_entries.sort()
    return all_entries


def convert_to_index(df, table):
    """
    Convert entries (items or users) into integer indices.

    :param df: a DataFrame of clicks or orders.
    :param table: a LookupTable for the conversions.
    :return: None.
    """
    df[USER_IDX] = df[COMP_MID].apply(table.user_to_index)
    df[PROD_IDX] = df[PROD_NO].apply(table.item_to_index)
    index = df[(df[USER_IDX] < 0) | (df[PROD_IDX] < 0)].index
    df.drop(index=index, columns=[COMP_MID, PROD_NO], inplace=True)


def to_instances(df_orders, df_clicks, min_timestamp=0):
    """
    Convert preprocessed DataFrames of orders and clicks into instances.

    :param df_orders: a DataFrame of orders.
    :param df_clicks: a DataFrame of clicks.
    :param min_timestamp: the minimum timestamp to generate instances.
    :return: the tuple of generated instances.
    """
    x_users = []
    x_orders = []
    x_clicks = []
    y_orders = []

    for user_idx, group in tqdm.tqdm(df_orders.groupby(USER_IDX)):
        if group.shape[0] <= MIN_ORDERS:
            continue

        click_group = None
        if df_clicks is not None:
            click_group = df_clicks[(df_clicks[USER_IDX] == user_idx)]

        order_indices = group[PROD_IDX]
        for i, (_, row) in enumerate(group.iterrows()):
            if i < MIN_ORDERS or row[TIMESTAMP] < min_timestamp:
                continue

            o_length = min(max(MIN_ORDERS, i), MAX_ORDERS)
            orders = np.full(MAX_ORDERS, -1, dtype=np.int32)
            orders[-o_length:] = order_indices[i - o_length:i]

            clicks = np.full(MAX_CLICKS, -1, dtype=np.int32)
            if df_clicks is not None:
                tz = timezone(timedelta(hours=5))  # 4am (KST) splits the day.
                y_time = datetime.fromtimestamp(row[TIMESTAMP], tz=tz)
                begin, end = get_yesterday(y_time)
                yesterday_clicks = click_group[(click_group[TIMESTAMP] >= begin)
                                               & (click_group[TIMESTAMP] <= end)]
                click_indices = yesterday_clicks.nlargest(
                    n=MAX_CLICKS, columns=[TIMESTAMP])[PROD_IDX].to_numpy()

                if click_indices.shape[0] > 0:
                    click_indices = np.flip(click_indices, axis=0)
                    clicks[-click_indices.shape[0]:] = click_indices[:]

            x_users.append(row[USER_IDX])
            x_orders.append(orders)
            x_clicks.append(clicks)
            y_orders.append(row[PROD_IDX])

    x_users = np.array(x_users, dtype=np.int32)
    x_orders = np.array(x_orders, dtype=np.int32)
    x_clicks = np.array(x_clicks, dtype=np.int32)
    y_orders = np.array(y_orders, dtype=np.int32)
    return x_users, x_orders, x_clicks, y_orders


def get_yesterday(timestamp):
    """
    Get the information of yesterday of a timestamp.

    :param timestamp: the current timestamp.
    :return: the beginning and ending timestamps of the yesterday.
    """
    ago = 1
    if timestamp.hour == 4:
        ago = 2
    yesterday = timestamp - timedelta(days=ago)
    yesterday_begin = datetime(
        yesterday.year, yesterday.month, yesterday.day, 0, 0, 0, 0)
    yesterday_begin_ts = datetime.timestamp(yesterday_begin)
    yesterday_end = datetime(
        yesterday.year, yesterday.month, yesterday.day, 23, 59, 59, 999)
    yesterday_end_ts = datetime.timestamp(yesterday_end)

    return yesterday_begin_ts, yesterday_end_ts


def save_instances(x_users, x_orders, x_clicks, y_orders, path):
    """
    Save instances as files.

    :param x_users: the features of users.
    :param x_orders: the features of orders.
    :param x_clicks: the features of clicks.
    :param y_orders: the labels of orders.
    :param path: the path to a directory to save files in.
    :return: None.
    """
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, 'users'), x_users)
    np.save(os.path.join(path, 'orders'), x_orders)
    np.save(os.path.join(path, 'clicks'), x_clicks)
    np.save(os.path.join(path, 'labels'), y_orders)


@click.command()
@click.option('--orders', type=click.Path(), default='../data/orders_6weeks.csv')
@click.option('--clicks', type=click.Path(), default=None)
@click.option('--categories', type=click.Path(), default='../data/prod_category.csv')
@click.option('--embedding', type=click.Path(), default='../out/title')
@click.option('--blacklist', type=click.Path(), default='../data/blacklist.txt')
@click.option('--out', type=click.Path(), default='../out/instances')
@click.option('--version', type=str, default='6w')
@click.option('--item-cut', type=int, default=1)
@click.option('--min-orders', type=int, default=4)
@click.option('--max-orders', type=int, default=8)
@click.option('--max-clicks', type=int, default=16)
def main(orders='../data/orders_6weeks.csv', clicks=None, categories='../data/prod_category.csv',
         embedding='../out/title', blacklist='../data/blacklist.txt', out='../out/instances',
         version='6w',item_cut=1, min_orders=4, max_orders=8, max_clicks=16):
    """
    Preprocess data and create instances for training recommendation models.

    :param orders: the path to orders.
    :param clicks: the path to clicks (optional).
    :param categories: the path to categories.
    :param embedding: the path to embedding vectors.
    :param blacklist: the path to a black list.
    :param out: the path to save outputs.
    :param version: the version of training data (2w, 6w, ...).
    :param item_cut: a threshold for using valid items.
    :param min_orders: the minimum number of orders in instances.
    :param max_orders: the maximum number of orders in instances.
    :param max_clicks: the maximum number of clicks in instances.
    :return: None.
    """
    start_time = time.time()

    set_parameters(min_orders, max_orders, max_clicks)
    columns = [COMP_MID, TIMESTAMP, PROD_NO, PROD_NM]
    df_orders = read_behaviors(orders, cols=columns)
    filter_blacklist(df_orders, blacklist)

    all_items = filter_by_occurrences(df_orders, PROD_NO, item_cut)
    all_users = filter_by_occurrences(df_orders, COMP_MID, min_orders + 1)
    table = LookupTable(embedding, all_items, all_users)
    table.set_categories(categories)
    convert_to_index(df_orders, table)

    df_clicks = None
    if clicks is not None:
        df_clicks = read_behaviors(clicks, cols=columns)
        filter_blacklist(df_clicks, blacklist)
        convert_to_index(df_clicks, table)

    ts_val, ts_test = get_timestamps(version)
    df_trn, df_val, df_test = split_orders(df_orders, ts_val, ts_test)
    trn_data = to_instances(df_trn, df_clicks)
    val_data = to_instances(df_val, df_clicks, ts_val)
    test_data = to_instances(df_test, df_clicks, ts_test)

    table.save_users(os.path.join(out, 'users'))
    table.save_items(os.path.join(out, 'items'))
    save_instances(*trn_data, os.path.join(out, 'training'))
    save_instances(*val_data, os.path.join(out, 'validation'))
    save_instances(*test_data, os.path.join(out, 'test'))

    with open(os.path.join(out, 'stats.txt'), 'w') as f:
        f.write(f'# of training instances: {trn_data[0].shape[0]}\n')
        f.write(f'# of validation instances: {val_data[0].shape[0]}\n')
        f.write(f'# of test instances: {test_data[0].shape[0]}\n')
        f.write(f'# of valid users: {table.user_ids.shape[0]}\n')
        f.write(f'# of target items: {table.item_ids.shape[0]}\n')
        f.write(f'Duration: {time.time() - start_time}\n')


if __name__ == '__main__':
    main()
