""" Wemakeprice Recommendation Project.

Authors:
- Hyunsik Jeon (jeon185@snu.ac.kr)
- Jaemin Yoo (jaeminyoo@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.

File: data/db.py
- Variables and functions to access the database.

Version: 1.0.0

"""
import sqlalchemy
from sqlalchemy.dialects.mysql import TINYINT, SMALLINT, MEDIUMINT, INTEGER

from .columns import *

_USER = 'wemap'
_PASSWORD = 'wemap2019'
_ADDRESS = 'warhol6.snu.ac.kr'
_PORT = '3306'
_DB = 'wemap'

_URL = 'mysql+pymysql://{0}:{1}@{2}:{3}/{4}?charset=utf8'.format(
    _USER, _PASSWORD, _ADDRESS, _PORT, _DB)

# The sqlalchemy engine that connects to wemap database.
ENGINE = sqlalchemy.create_engine(_URL, echo=False, encoding='utf-8')

# Data preprocessing class names and their corresponding table names.
TABLES = {
    'User': 'user',
    'Product': 'product',
    'Category': 'category',
    'DealItem': 'deal_item',
    'ProdCategory': 'prod_category',
}


def create_tables():
    """
    Create tables for all metadata.
    :return: None.
    """
    for table_name in TABLES.values():
        create_table(table_name)


def create_table(table_name):
    """
    Create table for given metadata.
    :param table_name: a name of the table to create, must be one of
                       'user', 'product', 'category', 'deal_item', or
                       'prod_category'.
    :return: None.
    """
    meta = sqlalchemy.MetaData()
    meta.reflect(bind=ENGINE)

    if table_name == TABLES['User']:
        columns = (
            sqlalchemy.Column(COMP_MEMBER_ID, sqlalchemy.CHAR(5), primary_key=True),
            sqlalchemy.Column(SEX, TINYINT(1, unsigned=True)),
            sqlalchemy.Column(AGE, SMALLINT(4, unsigned=True)),
            sqlalchemy.Column(MEMBER_ID, sqlalchemy.CHAR(64), nullable=False),
        )

    elif table_name == TABLES['Product']:
        columns = (
            sqlalchemy.Column(PRODUCT_NO, INTEGER(9, unsigned=True),
                              primary_key=True, autoincrement=False),
            sqlalchemy.Column(PRICE, INTEGER(9, unsigned=True)),
            sqlalchemy.Column(SALE_START_DATE, sqlalchemy.DATE),
            sqlalchemy.Column(SALE_END_DATE, sqlalchemy.DATE),
            sqlalchemy.Column(PRODUCT_NAME, sqlalchemy.VARCHAR(200)),
            sqlalchemy.Column(DEAL_CONSTRUCTION, sqlalchemy.VARCHAR(700)),
        )

    elif table_name == TABLES['Category']:
        columns = (
            sqlalchemy.Column(CATEGORY_NO, MEDIUMINT(6, unsigned=True),
                              primary_key=True),
            sqlalchemy.Column(DEPTH0_NAME, sqlalchemy.VARCHAR(30)),
            sqlalchemy.Column(DEPTH1_NO, MEDIUMINT(6, unsigned=True)),
            sqlalchemy.Column(DEPTH1_NAME, sqlalchemy.VARCHAR(30)),
            sqlalchemy.Column(DEPTH2_NO, MEDIUMINT(6, unsigned=True)),
            sqlalchemy.Column(DEPTH2_NAME, sqlalchemy.VARCHAR(30)),
            sqlalchemy.Column(DEPTH0_DISPLAY_ORDER, TINYINT(2, unsigned=True)),
            sqlalchemy.Column(DEPTH1_DISPLAY_ORDER, TINYINT(2, unsigned=True)),
            sqlalchemy.Column(DEPTH2_DISPLAY_ORDER, TINYINT(2, unsigned=True)),
        )

    elif table_name == TABLES['DealItem']:
        columns = (
            sqlalchemy.Column(DEAL_NO, INTEGER(9, unsigned=True),
                              sqlalchemy.ForeignKey(
                                  meta.tables[TABLES['Product']].c[PRODUCT_NO]
                              )),
            sqlalchemy.Column(ITEM_NO, INTEGER(9, unsigned=True),
                              sqlalchemy.ForeignKey(
                                  meta.tables[TABLES['Product']].c[PRODUCT_NO]
                              )),
        )

    elif table_name == TABLES['ProdCategory']:
        columns = (
            sqlalchemy.Column(PRODUCT_NO, INTEGER(9, unsigned=True),
                              sqlalchemy.ForeignKey(
                                  meta.tables[TABLES['Product']].c[PRODUCT_NO]
                              )),
            sqlalchemy.Column(CATEGORY_NO, MEDIUMINT(6, unsigned=True),
                              sqlalchemy.ForeignKey(
                                  meta.tables[TABLES['Category']].c[CATEGORY_NO]
                              )),
        )

    else:
        print("Not a valid table name.")
        return

    table = sqlalchemy.Table(table_name, meta, *columns)
    meta.create_all(ENGINE)
