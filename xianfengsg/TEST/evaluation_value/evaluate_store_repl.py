# -*- coding: utf-8 -*-
# @Time    : 2019/11/8 9:16
# @Author  : Ye Jinyu__jimmy
# @File    : evaluate_store_repl
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
# import XGBOOST_foreacst
# import decision_repl

import importlib,sys
importlib.reload(sys)
LANG="en_US.UTF-8"
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'


# data = pd.read_excel('D:/jimmy-ye/AI/AI_PRODUCT_PLAN/value_repl.xlsx',
#                          encoding='utf_8_sig')

#定义函数读取商品信息
def get_goods(name):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    # 读取的数据包括销量的时间序列，天气和活动信息
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", parameters)
    # 查看详细的出库数据，进行了日期的筛选，查看销量签50名的SKU
    get_code_sql = """SELECT rdp.B1GDGID FROM RPT_DZV_PSCPS rdp WHERE rdp.G1NAME like'%s' AND ROWNUM=1""" % (name)
    goods = pd.read_sql(get_code_sql, conn)
    if goods.empty ==True:
        sku_gid = '无数据'
        avg_price = '无数据'
        pass
    else:
        sku_gid = goods['B1GDGID'].iloc[0]
        #在读取其他详细信息
        get_price = """SELECT avg(r.RTLPRC) price  FROM RPGGD r WHERE
         r.GDGID='%s' AND r.LSTUPDTIME > TO_DATE('2019-01-01','yyyy-mm-dd') GROUP BY r.GDGID"""%(sku_gid)
        price = pd.read_sql(get_price, conn)
        avg_price = price['PRICE'].iloc[0]
    conn.close
    return sku_gid,avg_price


# data['sku_gid']= str(0)
# data['avg_price'] = str(0)
# for i in tqdm(range(len(data))):
#     print(i)
#     name = data['name'].iloc[i]
#     print(name)
#     sku_gid, avg_price= get_goods(name)
#     data['sku_gid'].iloc[i] = sku_gid
#     data['avg_price'].iloc[i] = avg_price
#
# print(data)
# data.to_csv('D:/jimmy-ye/AI/AI_PRODUCT_PLAN/data.csv',encoding='utf_8_sig')



#先读取所有浙北门店的门店code
def get_stcode():
    host = "192.168.1.205"  # 数据库ip
    port = "21521"  # 端口
    sid = "HDBI"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    # 读取的数据包括销量的时间序列，天气和活动信息
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("HDBI", "HDBI54325000", parameters)
    # 查看详细的出库数据，进行了日期的筛选，查看销量签50名的SKU
    store_sql = """SELECT os.CODE FROM ODS_STORE os  where os.area like '%%3301%%'"""
    store_code = pd.read_sql(store_sql, conn)
    stcode_list = store_code['CODE'].to_list()
    stcode = tuple(stcode_list)
    conn.close
    return stcode




#获取每家门店的2019年上半年的所有销售信息
def get_sales(i):
    host = "192.168.1.205"  # 数据库ip
    port = "21521"  # 端口
    sid = "HDBI"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    # 读取的数据包括销量的时间序列，天气和活动信息
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("HDBI", "HDBI54325000", parameters)
    # 查看详细的出库数据，进行了日期的筛选，查看销量签50名的SKU
    sales_sql = """SELECT ossd.GDCODE,TRUNC(ossd.FILDATE),sum(ossd.GDQTY),sum(ossd.GDSALE) FROM ods_sale_store_dtl ossd
             WHERE ossd.STCODE = '%s' AND ossd.FILDATE>=  to_date('2019-01-01','yyyy-mm-dd') 
            and ossd.FILDATE <  to_date('2019-07-01','yyyy-mm-dd') GROUP BY ossd.GDCODE,TRUNC(ossd.FILDATE)"""%(i)
    sales = pd.read_sql(sales_sql, conn)
    conn.close
    return sales

stcode = get_stcode()
data=pd.DataFrame()
for i in tqdm(stcode):
    print('正在进行门店的数据读取',i)
    sales = get_sales(i)
    sales.to_csv('D:/jimmy-ye/AI/AI_PRODUCT_PLAN/data'+str(i)+'.csv',encoding='utf_8_sig')
    data = data.append(sales)
data.to_csv('D:/jimmy-ye/AI/AI_PRODUCT_PLAN/data_final.csv',encoding='utf_8_sig')






