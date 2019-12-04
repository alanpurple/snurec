""" Wemakeprice Recommendation Project.

Authors:
- Hyunsik Jeon (jeon185@snu.ac.kr)
- Jaemin Yoo (jaeminyoo@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.

File: data/preprocess.py
- Classes for preprocessing metadata.

Version: 1.0.0

"""
from abc import abstractmethod, ABC
import pandas as pd
import itertools
import random
from .columns import *
from .db import ENGINE, TABLES


# Constants for filtering deal construction
REGION_NAMES = '마포,홍대,신촌,강남,논현,청담,역삼,노원,삼성,서초,중구,상계,목동,명동,일산,장항동,고양,광명,철산,' \
               '구리,김포,남양주,부천,중동,상동,중동,평촌,안양,오산,용인,처인구,동백,의정부,화성,수원,영통,신림역,' \
               '신도림역,영등포역,강남역,양재역,교대,남부터미널,인천,부평,구월동,경남,창원,마산,대전,둔산동,대구,' \
               '동천동,부산,사하구,대연동,동아대,잠실,성수,왕십리,건대입구,서울,경기,인천,경남,대구,부산,노원구,' \
               '동대문구,이태원,시흥시,성동구,화성시,강동,장안구,강북구,서대문구,강서,충주,종로,판교,강서구,경기도,' \
               '구리시,포천시,송파구,마포구,은평구,충북,금 천구,중랑구,충남,청주,분당,세종,광진구,하남시,고양시,이대,' \
               '평택,순천,광주'
GARBAGE_WORDS = 'AK플라자,AK백화점,AKMALL,AKPLAZA,AK몰,lch91,bigdeal'
FILTER_WORD_SET = set(REGION_NAMES.split(',')) | set(GARBAGE_WORDS.split(','))


class Preprocess(ABC):
    """
    Base class of all preprocessing classes.
    """
    def __init__(self, fp):
        """
        Class initializer.
        :param fp: file path of original data.
        """
        self.df = pd.read_csv(fp)
        self.preprocessed = False

    def preprocess(self):
        """
        A wrapper method of preprocessing method.
        :return: None.
        """
        if self.preprocessed:
            raise ValueError("The data has already been preprocessed.")
        else:
            self._preprocess()
            self.preprocessed = True

    @abstractmethod
    def _preprocess(self):
        """
        A preprocessing method.
        :return: None.
        """
        pass

    def to_csv(self, fp):
        """
        Write preprocessed data as a csv file.
        :param fp: file path to write data.
        :return: None.
        """
        if self.preprocessed:
            self.df.to_csv(fp, index=False)
        else:
            raise ValueError("The data has not yet been preprocessed.")

    def to_sql(self):
        """
        Insert preprocessed data into database.
        :return: None.
        """
        if self.preprocessed:
            self.df.to_sql(TABLES[self.__class__.__name__], ENGINE,
                           if_exists='append', index=False)
        else:
            raise ValueError("The data has not yet been preprocessed.")


class User(Preprocess):
    """
    A class preprocessing user data.
    """
    def __init__(self, fp, comp_mid_df=None):
        """
        Class initializer.
        :param fp: file path of original data.
        :param comp_mid_df: a dataframe that contains matching table of
                            member id and compressed member id. If not given,
                            compressed member id will be newly generated.
        """
        super().__init__(fp)
        self.comp_mid_df = comp_mid_df

    def _preprocess(self):
        if self.comp_mid_df is None:
            print("No comp_mid dataframe has been provided, thus assigning"
                  " a new one.")
            self._assign_compmid()
        else:
            self._join_compmid()

    def _assign_compmid(self):
        """
        Newly assign compressed member id to users.
        :return: None.
        """
        comp_mids = self.create_compmid()
        sampled_compmids = random.sample(comp_mids, k=self.df.shape[0])
        assign_kwargs = {COMP_MEMBER_ID: sampled_compmids}
        self.df = self.df.assign(**assign_kwargs)

    def create_compmid(self):
        """
        Generate compressed member id.
        :return: a list of generated compressed member id.
        """
        letters = '0123457689abcdefghijklmnopqrstuvwxyz'
        product_results = list(itertools.product(letters, repeat=5))
        comp_mids = [''.join(result) for result in product_results]
        return comp_mids

    def _join_compmid(self):
        """
        Join user data with pre-existing matching table.
        :return: None.
        """
        self.df = self.df.merge(self.comp_mid_df, on=MEMBER_ID)


class Product(Preprocess):
    """
    A class preprocessing product data.
    """
    def __init__(self, fp, valid_prods, search_words):
        """
        Class initializer.
        :param fp: file path of original data.
        :param valid_prods: a list of valid product number.
        :param search_words: a list of actual search keywords.
        """
        super().__init__(fp)
        self.valid_prods = valid_prods
        self.search_words = set(search_words)

    def _preprocess(self):
        self._filter_columns()
        self._filter_invalid_prods()
        self._filter_deal_construction()

    def _filter_columns(self):
        """
        Filter redundant columns from the data.
        :return: None.
        """
        self.df = self.df[[PRODUCT_NO, PRICE, SALE_START_DATE, SALE_END_DATE,
                           PRODUCT_NAME, DEAL_CONSTRUCTION]]

    def _filter_invalid_prods(self):
        """
        Filter rows of invalid products from the data.
        :return: None.
        """
        self.df = self.df[self.df[PRODUCT_NO].isin(self.valid_prods)]

    def _filter_deal_construction(self):
        """
        Filter deal construction column.
        :return: None.
        """
        self.df[DEAL_CONSTRUCTION] = self.df[DEAL_CONSTRUCTION].apply(
            lambda dc: self._filter_each_deal_construction(dc)
        )

    def _filter_each_deal_construction(self, dc):
        """
        Filter each deal construction value with garbage words and
        search keywords.
        :param dc: deal construction value.
        :return: filtered deal construction value.
        """
        if pd.isna(dc):
            return dc
        else:
            dc_set = set(dc.split(','))
            filtered_dc_set = (dc_set & self.search_words) - FILTER_WORD_SET
            filtered_dc = ','.join(list(filtered_dc_set))
            return filtered_dc


class Category(Preprocess):
    """
    A class preprocessing category data.
    """
    def __init__(self, fp):
        """
        Class initializer.
        :param fp: file path of original data.
        """
        super().__init__(fp)

    def _preprocess(self):
        self._change_dtype()
        self._assign_category_no()

    def _change_dtype(self):
        """
        The dtype of DEPTH2_NO column is float because of NaN values, so
        change the dtype to int.
        :return: None.
        """
        self.df[DEPTH2_NO] = pd.Series(self.df[DEPTH2_NO],
                                       dtype=pd.Int64Dtype())

    def _assign_category_no(self):
        """
        Assign CATEGORY_NO column to the data.
        :return: None.
        """
        category_no = self.df.apply(lambda x: self.get_leaf_category_no(x),
                                    axis=1)
        kwargs = {CATEGORY_NO: category_no}
        self.df = self.df.assign(**kwargs)

    def get_leaf_category_no(self, row):
        """
        Get leaf category number of each row.
        :param row: a row to find leaf category.
        :return: leaf category number.
        """
        if pd.isna(row[DEPTH2_NO]):
            return row[DEPTH1_NO]
        else:
            return row[DEPTH2_NO]


class DealItem(Preprocess):
    """
    A class preprocessing deal-item matching data.
    """
    def __init__(self, fp, product_df):
        """
        Class initializer.
        :param fp: file path of original data.
        :param product_df: a preprocessed product dataframe.
        """
        super().__init__(fp)
        self.df = self.df.rename(columns={PRODUCT_NO: ITEM_NO})
        self.product_df = product_df

    def _preprocess(self):
        self._filter_by_product()

    def _filter_by_product(self):
        """
        Filter rows of which deal or item does not belong to product data.
        :return: None.
        """
        self.df = self.df[self.df[ITEM_NO].isin(self.product_df[PRODUCT_NO])]
        self.df = self.df[self.df[DEAL_NO].isin(self.product_df[PRODUCT_NO])]


class ProdCategory(Preprocess):
    """
    A class preprocessing product-category matching data.
    """
    def __init__(self, fp, product_df):
        """
        Class initializer.
        :param fp: file path of original data.
        :param product_df: a preprocessed product dataframe.
        """
        super().__init__(fp)
        self.product_df = product_df

    def _preprocess(self):
        self._change_dtype()
        self._filter_by_product()

    def _change_dtype(self):
        """
        The dtype of PRODUCT_NO column is float because of NaN values, so
        change the dtype to int.
        :return: None.
        """
        self.df[PRODUCT_NO] = pd.Series(self.df[PRODUCT_NO],
                                        dtype=pd.Int64Dtype())

    def _filter_by_product(self):
        """
        Filter rows of which product does not belong to product data.
        :return: None.
        """
        self.df = self.df[(self.df[PRODUCT_NO].isin(self.product_df[PRODUCT_NO]))]
