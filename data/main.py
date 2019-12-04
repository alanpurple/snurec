""" Wemakeprice Recommendation Project.

Authors:
- Hyunsik Jeon (jeon185@snu.ac.kr)
- Jaemin Yoo (jaeminyoo@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.

File: data/main.py
- Script for generating CSV files from raw data.

Version: 1.0.0

"""
import os

import click
import pandas as pd

from data import preprocess as pre
from data.columns import MEMBER_ID, COMP_MEMBER_ID


def to_paths(test):
    """
    Return paths to input files based on whether it is test or not.

    :param test: whether it is for test or not.
    :return: the paths to files.
    """
    if test:
        org_path = os.path.join(
            os.path.split(os.getcwd())[0], 'res/metadata_src')
        pre_path = os.path.join(
            os.path.split(os.getcwd())[0], 'res/metadata')
    else:
        org_path = '/home/wemap/data/metadata_src/'
        pre_path = '/home/wemap/data/metadata/'

    org_data = {
        'User': os.path.join(
            org_path, 'indedu_suniv_member_20190325_201905021021.csv'),
        'Product': os.path.join(
            org_path, 'indedu_suniv_prod_deal_20190325_201904301118.csv'),
        'Category': os.path.join(
            org_path, 'indedu_suniv_gnbcate_20190325_201905021059.csv'),
        'DealItem': os.path.join(
            org_path, 'indedu_suniv_deal_prod_20190430_201905021021.csv'),
        'ProdCategory': os.path.join(
            org_path, 'indedu_suniv_prod_deal_gnbcate_mapping_'
                      '20190325_201905021023.csv'),
        'ValidProd': os.path.join(
            org_path, 'indedu_suniv_valid_deal_prod_20190521.csv')}

    pre_data = {
        'User': os.path.join(pre_path, 'user.csv'),
        'CompMid': os.path.join(pre_path, 'comp_mid.csv'),
        'Product': os.path.join(pre_path, 'product.csv'),
        'Category': os.path.join(pre_path, 'category.csv'),
        'DealItem': os.path.join(pre_path, 'deal_item.csv'),
        'ProdCategory': os.path.join(pre_path, 'prod_category.csv')}

    search_words = os.path.join(pre_path, 'search_words.txt')
    return org_data, pre_data, search_words


@click.command()
@click.option('--which', '-w', multiple=True)
@click.option('--to-sql', is_flag=True)
@click.option('--test', is_flag=True)
def main(which=None, to_sql=False, test=False):
    """
    Preprocess data and store CSV files.

    :param which: the type of data to preprocess.
    :param to_sql: whether to store the outputs in a DB or not.
    :param test: whether to run a test or not.
    :return: None.
    """
    org_data, pre_data, search_words = to_paths(test)
    valid_prods = None
    product_df = None

    which = tuple(map(lambda x: x.lower(), which))
    # No 'which' option means preprocessing everything.
    if not which:
        which = ('user', 'product', 'category', 'deal_item', 'prod_category')

    # If it is test, to_sql will be ignored.
    if test and to_sql:
        print("Test option is on, so to-sql will be ignored.")
        to_sql = False

    # To preprocess product data, valid product data must be ready.
    if 'product' in which:
        valid_prods = pd.read_csv(org_data['ValidProd']).deal_prod_no
    # To preprocess deal_item data, product data must be ready.
    elif 'deal_item' in which or 'prod_category' in which:
        product_df = pd.read_csv(pre_data['Product'])

    if 'user' in which:
        print("Preprocessing user data...")
        if os.path.exists(pre_data['CompMid']):
            comp_mid_df = pd.read_csv(pre_data['CompMid'])
            user = pre.User(org_data['User'], comp_mid_df)
        else:
            user = pre.User(org_data['User'])
        user.preprocess()
        user.to_csv(pre_data['User'])
        if user.comp_mid_df is None:
            comp_mid_df = user.df[[MEMBER_ID, COMP_MEMBER_ID]]
            comp_mid_df.to_csv(pre_data['CompMid'], index=False)
        if to_sql:
            user.to_sql()

    if 'product' in which:
        print("Preprocessing product data...")
        with open(search_words, 'r') as f:
            search_words = f.read().splitlines()
        product = pre.Product(org_data['Product'], valid_prods, search_words)
        product.preprocess()
        product.to_csv(pre_data['Product'])
        product_df = product.df
        if to_sql:
            product.to_sql()

    if 'category' in which:
        print("Preprocessing category data...")
        category = pre.Category(org_data['Category'])
        category.preprocess()
        category.to_csv(pre_data['Category'])
        if to_sql:
            category.to_sql()

    if 'deal_item' in which:
        print("Preprocessing deal_item data...")
        deal_item = pre.DealItem(org_data['DealItem'], product_df)
        deal_item.preprocess()
        deal_item.to_csv(pre_data['DealItem'])
        if to_sql:
            deal_item.to_sql()

    if 'prod_category' in which:
        print("Preprocessing prod_category data...")
        prod_category = pre.ProdCategory(
            org_data['ProdCategory'], product_df)
        prod_category.preprocess()
        prod_category.to_csv(pre_data['ProdCategory'])
        if to_sql:
            prod_category.to_sql()


if __name__ == '__main__':
    main()
