# -*- coding: utf-8 -*-
# @Time    : 2019/11/18 10:48
# @Author  : Ye Jinyu__jimmy
# @File    : get_each_sku]

import pandas as pd
import cx_Oracle
import os
import numpy as np
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
# 注：设置环境编码方式，可解决读取数据库乱码问题
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
from matplotlib import pyplot as plt
import math
import time
import multiprocessing
import re
import tqdm
import pymysql
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdate
#parser是根据字符串解析成datetime,字符串可以很随意，可以用时间日期的英文单词，可以用横线、逗号、空格等做分隔符。没指定时间默认是0点，没指定日期默认是今天，没指定年份默认是今年。
from dateutil.parser import parse
# from pylab import *
plt.switch_backend('agg')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
#如下是支持中文数字
# mpl.rcParams['font.sans-serif'] = ['SimHei']
from statsmodels.graphics.mosaicplot import mosaic
'''读取销售数据，查看每个sku的基本统计学信息，并汇总记录'''


#----------------------------------------------------设置函数记录所有的SKU
#-------------最新的逻辑是从叫货目录进行选择
def get_order_code():
    print('正在读取叫货目录的数据')
    dbconn = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com", database="purchare_sys",
                             user="purchare_sys", password="purchare_sys@123", port=3306,
                             charset='utf8')
    get_orders = """SELECT pcr.goods_id GOODS_GID,pcr.goods_code GOODS_CODE,pcr.goods_name FROM p_call_record pcr 
    WHERE pcr.warehouse_id ='1000255' GROUP BY pcr.goods_id"""
    orders = pd.read_sql(get_orders, dbconn)
    orders['GOODS_GID'] = orders['GOODS_GID'].astype(str)
    print('叫货目录读取完成')
    dbconn.close()
    return orders

#------------------------依据每个SKU的7th_code进行销售数据的读取
def get_sales_his(code_7th):
    print('正在读取销售数据:'+str(code_7th))
    dbconn = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com", database="purchare_sys",
                             user="purchare_sys", password="purchare_sys@123", port=3306,
                             charset='utf8')
    sales_sql = """ SELECT sh.Qty QTY
    FROM sales_his sh WHERE sh.GDGID='%s'"""%(code_7th)
    sales = pd.read_sql(sales_sql, dbconn)
    print('销售数据读取完成')
    dbconn.close()
    return sales

#-------------------------用于设置函数用来更改dataframe的列名，使数据能够在一维展示
def rename_columns(data,code_7th):
    new_name = str(code_7th)
    data.rename(columns={'QTY': new_name}, inplace=True)
    return data

#-----------------------设置循环读销售数据的函数
def get_cycle_sku():
    orders = get_order_code()
    sku_gid_list = list(set(orders['GOODS_GID']))
    print(sku_gid_list)
    final = pd.DataFrame()
    # for i in range(10):
    for code_7th in sku_gid_list:
    # code_7th = '3004924'
    #     code_7th = sku_gid_list[i]
        print(code_7th)
        sales = get_sales_his(code_7th)
        sales = rename_columns(sales,code_7th)
        data = sales.describe()
        print(data)
        final= pd.concat([data, final], axis=1)
    return final

# final = get_cycle_sku()
#
# final.to_csv('D:/AI/xianfengsg/TEST/statistics_analysis/final_statistics.csv',encoding='utf_8_sig')

#--------------------------设置函数用来记录获取转置后的数据
def final_data():
    final = pd.read_csv('D:/AI/xianfengsg/TEST/statistics_analysis/final_statistics.csv',encoding='utf_8_sig')
    data = pd.DataFrame(final.values.T, index=final.columns, columns=final.index)
    series = data.iloc[0].to_list()
    data.columns = series
    data = data.drop(index = ['Unnamed: 0'])
    data = data.drop(['unique'],axis=1)
    data = data.fillna(0)
    index_list = data.index.values.tolist()
    data['Sku_id'] = index_list
    data.reset_index(drop=True,inplace=True)
    return data


#-------------------------获取每个sku的标签数据，效果大于等于人工和不如人工，作为标签项
def sku_label():
    get_good_sku = pd.read_csv('D:/AI/xianfengsg/TEST/statistics_analysis/good_sku.csv',encoding='utf_8_sig')
    get_all_sku  = pd.read_csv('D:/AI/xianfengsg/TEST/statistics_analysis/result.csv',encoding='utf_8_sig')

    get_good_sku.rename(columns={'name': 'label'}, inplace = True)
    get_good_sku['label'] = 1
    all_sku = get_all_sku[['Sku_id']]
    #求差集
    all_sku = all_sku.append(get_good_sku)
    all_sku = all_sku.append(get_good_sku)
    bad_sku = all_sku.drop_duplicates(subset=['Sku_id'],keep=False)
    bad_sku['label'] = 0
    data_total = pd.concat([bad_sku,get_good_sku])
    data_total.reset_index(drop=True, inplace=True)
    return data_total

all_statistics = final_data()
data_label = sku_label()
all_statistics['Sku_id'] = all_statistics['Sku_id'].astype(str)
data_label['Sku_id'] = data_label['Sku_id'].astype(str)
data_merge = pd.merge(all_statistics,data_label,on='Sku_id',how='inner')
data_merge['indicator_1'] = data_merge['std']/data_merge['mean']
data_merge['indicator_2'] = data_merge['std'] * data_merge['count']
data_merge['indicator_3'] = (data_merge['std']/data_merge['mean']) * data_merge['count']
def indicator_4(x):
    y = x['std'] *(x['count'] ** 0.5)
    return y
data_merge['indicator_4'] = data_merge.apply(lambda x: indicator_4(x),axis =1)
data_merge['indicator_5'] = (data_merge['std']/data_merge['mean']) * (data_merge['count'] ** 0.5)
print(data_merge)
# cross1 = pd.crosstab(pd.qcut(data_merge['mean'],6),data_merge['label'])
# print(cross1)
# mosaic(cross1.stack())
#
#
# cross2 = pd.crosstab(pd.qcut(data_merge['count'],5),data_merge['label'])
# print(cross2)
# mosaic(cross2.stack())
#
# cross3 = pd.crosstab(pd.qcut(data_merge['std'],6),data_merge['label'])
# print(cross3)
# mosaic(cross3.stack())
#
# cross4 = pd.crosstab(pd.qcut(data_merge['75%'],6),data_merge['label'])
# print(cross4)
# mosaic(cross4.stack())


#---------------
cross5 = pd.crosstab(pd.qcut(data_merge['indicator_1'],7),data_merge['label'])
print(cross5)
mosaic(cross5.stack())


cross6 = pd.crosstab(pd.qcut(data_merge['indicator_2'],7),data_merge['label'])
print(cross6)
mosaic(cross6.stack())

cross7 = pd.crosstab(pd.qcut(data_merge['indicator_3'],7),data_merge['label'])
print(cross7)
mosaic(cross7.stack())

cross8 = pd.crosstab(pd.qcut(data_merge['indicator_4'],7),data_merge['label'])
print(cross8)
mosaic(cross8.stack())

cross9 = pd.crosstab(pd.qcut(data_merge['indicator_5'],7),data_merge['label'])
print(cross9)
mosaic(cross9.stack())
#



