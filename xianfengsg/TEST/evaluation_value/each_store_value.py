# -*- coding: utf-8 -*-
# @Time    : 2019/11/11 9:59
# @Author  : Ye Jinyu__jimmy
# @File    : each_store_value


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
    sales_sql = """SELECT ossd.GDCODE,TRUNC(ossd.FILDATE) DATE_TIME ,sum(ossd.GDQTY) TOTAL_GDQTY
                ,sum(ossd.GDSALE) TOTAL_GDSALE FROM ods_sale_store_dtl ossd
             WHERE ossd.STCODE = '%s' AND ossd.FILDATE>=  to_date('2019-01-01','yyyy-mm-dd') 
            and ossd.FILDATE <  to_date('2019-07-01','yyyy-mm-dd') GROUP BY ossd.GDCODE,TRUNC(ossd.FILDATE)"""%(i)

    sales = pd.read_sql(sales_sql, conn)
    sales['GDCODE'] = sales['GDCODE'].astype(str)
    conn.close
    return sales



#需要对每一家门店都进行毛利的计算查看门店的价格情况,这里分为两步操作，先要通过五位code码获取到对应的7位子码，然后通过7位子码，找到对应的销售数据
def get_goods_price(code):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    # 读取的数据包括销量的时间序列，天气和活动信息
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", parameters)
    get_7th_code_sql = """SELECT rdp.B2GDGID  FROM RPT_DZV_PSCPS rdp WHERE rdp.G2CODE='%s'"""%(code)
    get_7th_code = pd.read_sql(get_7th_code_sql,conn)
    code_7th =  get_7th_code['B2GDGID'].iloc[0]
    #读取到7位子码的信息后，继续读取该商品的会员价的信息
    get_price_sql = """SELECT r.MBRPRC,trunc(r.LSTUPDTIME) FROM RPGGD r WHERE r.gdgid  ='%s'
                    AND r.LSTUPDTIME >= TO_DATE('2019-01-01','yyyy-mm-dd')
                    AND r.LSTUPDTIME <= TO_DATE('2019-07-01','yyyy-mm-dd') """%(code_7th)
    price = pd.read_sql(get_price_sql,conn)
    price['GDCODE'] = price['GDCODE'].astype(str)
    return price



#获取所有的商品的7位码和5位码和会员价，日期四个维度的信息，然后再和每次门店的销售数据进行合并
def get_goods():
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  #端口
    sid = "hdapp"  #数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    # 读取的数据包括销量的时间序列，天气和活动信息
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", parameters)
    get_goods = """SELECT r.MBRPRC,trunc(r.LSTUPDTIME) DATE_TIME,r.gdgid ,b.G2CODE GDCODE FROM RPGGD r 
                    INNER JOIN(SELECT *  FROM RPT_DZV_PSCPS rdp)b 
                    ON b.B2GDGID = r.GDGID
                    WHERE r.LSTUPDTIME >= TO_DATE('2019-01-01','yyyy-mm-dd')
                    AND r.LSTUPDTIME <= TO_DATE('2019-07-01','yyyy-mm-dd')"""
    goods_price = pd.read_sql(get_goods,conn)
    return goods_price

#由于设定价格的特性不是连续的日期，因此需要对日期进行在处理的操作
def data_fill(data):
    data_pro = data.drop_duplicates()
    date_range_sku = pd.date_range(start='20190101', end='20190701')
    data_sku = pd.DataFrame({'DATE_TIME': date_range_sku})
    code = set(data_pro['GDCODE'])
    final = pd.DataFrame()
    for each_code in tqdm(code):
        data_each = data_pro[data_pro['GDCODE'] == each_code]
        result = pd.merge(data_each, data_sku, on=['DATE_TIME'], how='right')
        result = result.fillna(method='ffill')
        result = result.fillna(method='bfill')
        final = final.append(result)
    return final

stcode = get_stcode()
data=pd.DataFrame()
# goods = get_goods()
# goods_price = data_fill(goods)
# goods_price.to_csv('D:/jimmy-ye/AI/AI_PRODUCT_PLAN/goods_price.csv',encoding='utf_8_sig')


#
# sales = get_sales('33010061')
# sales = sales.sort_values(by=['GDCODE','DATE_TIME'])
# sales_final = pd.merge(sales, goods_price, on=['GDCODE', 'DATE_TIME'], how='left')
# # sales_final = sales_final.fillna(method='ffill')
# # sales_final = sales_final.fillna(method='bfill')
# sales_final.to_csv('D:/jimmy-ye/AI/AI_PRODUCT_PLAN/data_process.csv', encoding='utf_8_sig')
for i in tqdm(stcode):
    print('正在进行门店的数据读取',i)
    if os.path.exists('D:/jimmy-ye/AI/AI_PRODUCT_PLAN/data_process' + str(i) + '.csv') == True:
        sales_final = pd.read_csv('D:/jimmy-ye/AI/AI_PRODUCT_PLAN/data_process' + str(i) + '.csv', encoding='utf_8_sig')
        print('读取数据完成')
        # sales_final = get_sales(i)
        if sales_final.empty == False:
            # sales = sales.sort_values(by=['GDCODE','DATE_TIME'])
            # sales_final = pd.merge(sales,goods_price,on=['GDCODE','DATE_TIME'],how='left')
        #     sales_final= sales_final.fillna(method='ffill')
        #     sales_final = sales_final.fillna(method='bfill')
        #     sales_final.to_csv('D:/jimmy-ye/AI/AI_PRODUCT_PLAN/data_process'+str(i)+'.csv',encoding='utf_8_sig')
            sales_final['STCODE'] = i
            data = data.append(sales_final)
        else:
            print('该门店没有数据')
            pass
    else:
        pass
data = data.dropna(axis=0, how='any')
data.to_csv('D:/jimmy-ye/AI/AI_PRODUCT_PLAN/data_final_processed_stcode.csv',encoding='utf_8_sig')






