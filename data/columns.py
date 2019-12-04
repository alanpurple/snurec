""" Wemakeprice Recommendation Project.

Authors:
- Hyunsik Jeon (jeon185@snu.ac.kr)
- Jaemin Yoo (jaeminyoo@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.

File: data/columns.py
- Constants of DataFrame column names.

Version: 1.0.0

"""
TIMESTAMP = 'timestamp'
ACTION_TYPE = 'action_type'
SEARCH_KEYWORD = 'search_keyword'

MEMBER_ID = 'm_id'
COMP_MEMBER_ID = 'comp_mid'
AGE = 'age'
SEX = 'sex'

PRODUCT_NO = 'prod_no'
PRICE = 'sale_price'
SALE_START_DATE = 'sale_start_dt'
SALE_END_DATE = 'sale_end_dt'
PRODUCT_NAME = 'prod_nm'
DEAL_CONSTRUCTION = 'deal_construction'

DEPTH0_NAME = 'gnb_depth0_nm'
DEPTH1_NO = 'gnb_depth1_cate_id'
DEPTH1_NAME = 'gnb_depth1_nm'
DEPTH2_NO = 'gnb_depth2_cate_id'
DEPTH2_NAME = 'gnb_depth2_nm'
DEPTH0_DISPLAY_ORDER = 'display_order_depth0'
DEPTH1_DISPLAY_ORDER = 'display_order_depth1'
DEPTH2_DISPLAY_ORDER = 'display_order_depth2'
CATEGORY_NO = 'gnb_category_id'

DEAL_NO = 'deal_no'
ITEM_NO = 'item_no'
