""" Wemakeprice Recommendation Project.

Authors:
- Hyunsik Jeon (jeon185@snu.ac.kr)
- Jaemin Yoo (jaeminyoo@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.

refactored by Alan Anderson (alan@wemakeprice.com)

File: rec/utils.py
- Useful functions and constants for recommendation scripts.

Version: 1.0.0

"""
import os

import numpy as np
import tensorflow as tf

from rec import models

# Information to access a HDFS.
HOST = 'matisse.snu.ac.kr'
PORT = 22
USERNAME = 'wemap'
PASSWORD = 'wemakerec'

# Column names to access the DataFrames.
COMP_MID = 'comp_mid'
TIMESTAMP = 'timestamp'
PROD_NO = 'prod_no'
USER_IDX = 'user_idx'
PROD_NM = 'prod_nm'
PROD_IDX = 'prod_idx'
CATEGORY_ID = 'gnb_category_id'
CATEGORY_IDX = 'gnb_category_idx'


def initialize_model(name,num_items,emb_len, embeddings, categories,
                    num_layers=1,num_units=128,decay=0,emb_way='mlp'):
    """
    Initialize a recommendation model based on its name.

    :param name: the name of a model to initialize.
    :param embeddings: embedding vectors of all items.
    :param categories: multi-hot categories of all items.
    :param kwargs: a dictionary of other arguments.
    :return: the initialized model.
    """

    if name in {'last', 'average'}:
        return models.BaselineModel(num_items,emb_len,embeddings, mode=name)
    elif name == 'rnn-v1':
        return models.RNN1(num_items,emb_len,embeddings,
                           num_layers=num_layers,
                           num_units=num_units,
                           decay=decay)
    elif name == 'rnn-v2':
        return models.RNN2(num_items,emb_len,embeddings, categories,
                           num_layers=num_layers,
                           num_units=num_units,
                           decay=decay,
                           emb_way=emb_way,
                           )
    elif name == 'rnn-v3':
        return models.RNN3(num_items,emb_len,embeddings, categories,
                           num_layers=num_layers,
                           num_units=num_units,
                           decay=decay,
                           emb_way=emb_way
                           )
    elif name == 'rnn-v4':
        return models.RNN4(num_items,emb_len,embeddings, categories,
                           num_layers,num_units, decay,
                           emb_way=emb_way)
    else:
        raise ValueError(name)


def read_users(path):
    """
    Read user information (for training).

    :param path: the path to user information.
    :return: the loaded NumPy array.
    """
    return np.load(os.path.join(path, 'ids.npy'), allow_pickle=True)


def read_titles(path):
    """
    Read title information (for evaluation).

    :param path: the path to title information.
    :return: the loaded NumPy array.
    """
    return np.load(os.path.join(path, 'titles.npy'))

def read_instances(path, batch_size, buffer_size=10000):
    """
    Read instances (for both training and evaluation).

    :param path: the path to a directory containing instance files.
    :param batch_size: a batch size.
    :param buffer_size: a buffer size for shuffling.
    :return: the dataset consisting of multiple batches.
    """
    users = np.load(os.path.join(path, 'users.npy'))
    orders = np.load(os.path.join(path, 'orders.npy'))
    clicks = np.load(os.path.join(path, 'clicks.npy'))
    labels = np.load(os.path.join(path, 'labels.npy'))

    arrays = (users, orders, clicks), labels
    dataset = tf.data.Dataset.from_tensor_slices(arrays)
    return dataset.shuffle(buffer_size).batch(batch_size)
