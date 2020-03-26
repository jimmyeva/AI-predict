# -*- coding: utf-8 -*-
# @Time    : 2019/11/20 10:20
# @Author  : Ye Jinyu__jimmy
# @File    : selected_sku_evaluated
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import *
import itertools
import datetime
import os
import pymysql
import copy
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
import math
import warnings
import cx_Oracle

import importlib,sys
importlib.reload(sys)
LANG="en_US.UTF-8"
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

'''
产品中心选择的测试SKU：

65550：樱桃小番茄
07540：优选泰国桂团
06290：金边香芒
01310：克伦森无子提
05020：鲜丰玉麒麟
06850：百香果
07600：菲律宾凤梨
07300：蜜宝火龙果
01020：辽宁黑珍珠
12240：珍珠牛奶枣

'''

#--------------------------------先将这些数据换成标准的dataframe格格式,ssd == selected_sku_data
ssd = pd.DataFrame({'Code':['65550','07540','06290','01310','05020','06850','07600','07300','01020','12240']})

#----------------------------------设置函数用来获取每日的评估的对比数据
def start_end_date_finance(start, end,ssd):
    data_record = pd.DataFrame()
    for days in tqdm(pd.date_range(start, end)):
        today = days.strftime('%Y%m%d')
        final_data = pd.read_csv('D:/AI/xianfengsg/TEST/evaluate_AI_HUMAN/final' + str(today) + '.csv',
                      encoding='utf_8_sig',index_col=False)#,dtype = {'Code' : str})   #,converters={u'Code': str})
        final_data['Code'] = final_data['Code'].astype(int)
        final_data['Code'] = final_data['Code'].astype(str)
        def polishing(x):
            code = x['Code'].rjust(5, '0')
            return code

        final_data['Code'] = final_data.apply(lambda x: polishing(x), axis=1)
        print(final_data)
        merge_data = pd.merge(final_data,ssd,on='Code',how='inner')
        print(merge_data)
        data_record = data_record.append(merge_data)
    return data_record



data_record = start_end_date_finance('20191021','20191107',ssd)
data_record.to_csv('D:/AI/xianfengsg/TEST/evaluate_AI_HUMAN/selected_sku_evaluate/final.csv',
                      encoding='utf_8_sig',index=False)














