""" Wemakeprice Recommendation Project.

Authors:
- Hyunsik Jeon (jeon185@snu.ac.kr)
- Jaemin Yoo (jaeminyoo@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.

File: rec/data_aggregate.py
- Aggregate preprocessed data from a remote HDFS.

Version: 1.0.0

"""
import fnmatch
import time

import click
import pandas as pd
import paramiko
import tqdm

from utils import *


def read_data_from_hdfs(in_path):
    """
    Read data stored in a HDFS.

    :param in_path: the path to the HDFS.
    :return: a DataFrame of loaded data.
    """
    transport = paramiko.Transport((HOST, PORT))
    transport.connect(username=USERNAME, password=PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)

    dfs = []
    for path in tqdm.tqdm(sftp.listdir(in_path)):
        if fnmatch.fnmatch(path, '*.csv'):
            with sftp.open(in_path + '/' + path) as f:
                dfs.append(pd.read_csv(f))
    return pd.concat(dfs)


def remove_duplicate_items(df):
    """
    Remove duplicate records with the same IDs and timestamps.

    :param df: a DataFrame to remove duplicates.
    :return: the new DataFrame.
    """
    df_left = df[[COMP_MID, TIMESTAMP]]
    df_right = df.shift(fill_value=0)[[COMP_MID, TIMESTAMP]]
    return df.loc[(df_left != df_right).any(axis=1), :]


def remove_continuous_items(df):
    """
    Remove continuous records whose items are the same.

    :param df: a DataFrame to remove items.
    :return: the new DataFrame.
    """
    df_left = df[[COMP_MID, PROD_NO]]
    df_right = df.shift(fill_value=0)[[COMP_MID, PROD_NO]]
    return df.loc[(df_left != df_right).any(axis=1), :]


@click.command()
@click.option('--hdfs', type=click.Path(), default=None)
@click.option('--out', type=click.Path(), default=None)
def main(hdfs=None, out=None):
    """
    Aggregate data from a HDFS and store them as a single CSV.

    :param hdfs: the path to the HDFS.
    :param out: the path to store the CSV.
    :return: None.
    """
    assert hdfs is not None

    start_time = time.time()
    out = '../data/aggregated.csv' if out is None else out

    df = read_data_from_hdfs(hdfs)
    df.sort_values([COMP_MID, TIMESTAMP], ascending=[True, True], inplace=True)
    df = remove_duplicate_items(df)
    df = remove_continuous_items(df)
    df.to_csv(out, header=True, index=False)

    print(f'duration: {time.time() - start_time:.4f}')


if __name__ == '__main__':
    main()
