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

rewritten by Alan Anderson (alan@wemakeprice.com)

"""
import os

import numpy as np
import tensorflow as tf

import models

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


def initialize_model(name=None, embeddings=None,
                    num_layers=1,num_units=128,decay=0,category_emb=None):
    """
    Initialize a recommendation model based on its name.

    :param name: the name of a model to initialize.
    :param embeddings: embedding vectors of all items.
    :param categories: multi-hot categories of all items.
    :param kwargs: a dictionary of other arguments.
    :return: the initialized model.
    """

    assert name[:3]=='rnn'

    if name == 'rnn-v1':
        return models.RNN1(embeddings,num_layers,
                           num_units,
                           decay)
    elif name == 'rnn-v2':
        return models.RNN2(embeddings,num_layers,
                           num_units,
                           decay,category_emb)
    elif name == 'rnn-v3':
        return models.RNN3(embeddings,num_layers,
                           num_units,
                           decay,category_emb)
    elif name == 'rnn-v4':
        return models.RNN4(embeddings,num_layers,
                           num_units,
                           decay,category_emb)
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

def read_instances(path, batch_size, buffer_size=10000,is_exp=False):
    """
    Read instances (for both training and evaluation).

    :param path: the path to a directory containing instance files.
    :param batch_size: a batch size.
    :param buffer_size: a buffer size for shuffling.
    :return: the dataset consisting of multiple batches.
    """
    
    orders = np.load(os.path.join(path, 'orders.npy'))
    labels = np.load(os.path.join(path, 'labels.npy'))
    if is_exp:
        #users = np.load(os.path.join(path, 'users.npy'))
        clicks = np.load(os.path.join(path, 'clicks.npy'))
        arrays = (orders, clicks), labels
    else:
        arrays=orders,labels
    dataset = tf.data.Dataset.from_tensor_slices(arrays)
    return dataset.shuffle(buffer_size).batch(batch_size)
