# -*- coding: utf-8 -*-
# @Time    : 2019/6/24 9:15
# @Author  : Ye Jinyu__jimmy
# @File    : test_01

import pandas as pd
import cx_Oracle
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import re
import os
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
# 注：设置环境编码方式，可解决读取数据库乱码问题
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
import math
import time
import multiprocessing
from dateutil.parser import parse



def read_recent():
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    #读取的数据包括销量的时间序列，天气和活动信息
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", parameters)
    orders_history = """SELECT oep.STOREANAME,oep.GOODSNAME,oep.OPR_DT,oep.NEWQTY,oep.QTY,
                        oep.CHECKEDQTY,oep.PRICE 
                        FROM OA_EXAMINE_PREALCPOOL oep where oep.OPR_DT > to_date('2019-03-01','yyyy-mm-dd')"""
    orders_history_sql_read = pd.read_sql(orders_history, conn)
    conn.close
    return orders_history_sql_read

def read_history():
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    #读取的数据包括销量的时间序列，天气和活动信息
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", parameters)
    orders_recent_sql = """SELECT oepb.STOREANAME,oepb.GOODSNAME,oepb.OPR_DT,oepb.NEWQTY,oepb.QTY,
                            oepb.CHECKEDQTY,oepb.PRICE 
                            FROM OA_EXAMINE_PREALCPOOL_BAK oepb"""
    orders_recent_sql_read = pd.read_sql(orders_recent_sql, conn)
    conn.close
    return orders_recent_sql_read



def date_normalize(data_frame):
    data_frame_sort = data_frame.sort_values(by = ['OPR_DT'],ascending=False )
    data_frame_sort['OPR_DT'] = pd.to_datetime(data_frame_sort['OPR_DT']).dt.normalize()
    return data_frame_sort

def old_algorithm():
    order_recent = read_recent()
    print(order_recent,type(order_recent))
    order_history =read_history()
    print(order_history,type(order_history))
    orders_total = pd.concat([order_history,order_recent],ignore_index=True)
    orders_total['total_amount_price'] = orders_total['NEWQTY'] *orders_total['PRICE']
    sales = date_normalize(orders_total)
    final_data = pd.DataFrame()
    final_data["income_qty"]=sales.groupby(["STOREANAME",'OPR_DT'],as_index = False).sum()["total_amount_price"]
    final_data['variety'] = sales.groupby(["STOREANAME",'OPR_DT'],as_index = False).count()["GOODSNAME"]
    final_data['STOREANAME'] = sales.groupby(["STOREANAME",'OPR_DT'],as_index = False).mean()["STOREANAME"]
    final_data['OPR_DT'] = sales.groupby(["STOREANAME",'OPR_DT'],as_index = False).mean()["OPR_DT"]
    final_data['QTY'] = sales.groupby(["STOREANAME",'OPR_DT'],as_index = False).count()["QTY"]
    final_data['CHECKEDQTY'] = sales.groupby(["STOREANAME",'OPR_DT'],as_index = False).mean()["CHECKEDQTY"]
    print(final_data)

    final_data.to_csv('C:/Users/dell/Desktop/final_data.csv', encoding='utf_8_sig')

# Sale_Data_group = Sale_Data.groupby(['BJ_BARCODE', 'BJ_DATE'], as_index=False)
#             Sale_Data_group_sum = Sale_Data_group.agg(sum)




def read_history_sum():
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    #读取的数据包括销量的时间序列，天气和活动信息
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", parameters)
    orders_history = """SELECT oepb.STOREANAME,SUBSTR(oepb.OPR_DT,1,10)AS time_date,SUM(oepb.NEWQTY *oepb.PRICE)AS total_price,
    SUM(oepb.QTY)AS old_qty,
                        SUM(oepb.CHECKEDQTY)AS mid_qty,sum(oepb.NEWQTY)as last_qty, sum(oepb.PRICE)AS price
                        FROM OA_EXAMINE_PREALCPOOL_BAK oepb 
                         GROUP BY oepb.STOREANAME,SUBSTR(oepb.OPR_DT,1,10)"""
    orders_history_sql_read = pd.read_sql(orders_history, conn)
    conn.close
    return orders_history_sql_read

def read_rencent_sum():
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    #读取的数据包括销量的时间序列，天气和活动信息
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", parameters)
    orders_recent_sql = """SELECT oepb.STOREANAME,SUBSTR(oepb.OPR_DT,1,10)AS time,SUM(oepb.NEWQTY *oepb.PRICE) AS total_price,
                        SUM(oepb.QTY) AS old_qty,
                        SUM(oepb.CHECKEDQTY) AS mid_qty,sum(oepb.NEWQTY) as last_qty, sum(oepb.PRICE) AS price
                        FROM OA_EXAMINE_PREALCPOOL_BAK oepb  
                         GROUP BY oepb.STOREANAME,SUBSTR(oepb.OPR_DT,1,10)"""
    orders_recent_sql_read = pd.read_sql(orders_recent_sql, conn)
    conn.close
    return orders_recent_sql_read
#
# orders_recent_sql_read = read_history_sum()
# orders_recent_sql_read.to_csv('C:/Users/dell/Desktop/orders_recent_sql_read.csv', encoding='utf_8_sig')
# orders_history_sql_read = read_history_sum()
# orders_history_sql_read.to_csv('C:/Users/dell/Desktop/orders_history_sql_read.csv', encoding='utf_8_sig')
# final_data=pd.concat([orders_recent_sql_read,orders_history_sql_read])
# final_data.to_csv('C:/Users/dell/Desktop/orders_store.csv', encoding='utf_8_sig')

# store_sales = pd.read_excel('C:/Users/dell/Desktop/sales_store_new.xlsx',encoding='utf_8_sig')
#
# store_orders = pd.read_csv('C:/Users/dell/Desktop/final_data.csv', encoding='utf_8_sig')
# print(store_sales.head(10))
# print(store_orders.head(10))
# store_orders =  store_orders.rename(index=str,columns={'SUBSTR(OEPB.OPR_DT,1,10)':'time',
#                                                        'STORENAME':'名称',
#                                                        'SUM(OEPB.NEWQTY*OEPB.PRICE)':'total_price',
#                                                        'SUM(OEPB.QTY)':'old_qty',
#                                                        'SUM(OEPB.CHECKEDQTY)':'mid_qty',
#                                                        'SUM(OEPB.NEWQTY)':'last_qty',
#                                                        'SUM(OEPB.PRICE)':'price'})
# store_sales =  store_sales.rename(index=str,columns={'SUBSTR(BS.FILDATE,1,10)':'time'})
# store_sales['time'] = pd.to_datetime(store_sales['time']).dt.normalize()
# store_orders['time'] = pd.to_datetime(store_orders['time']).dt.normalize()
#
# total_data = pd.merge(store_sales,store_orders,on=['time','名称'],how='outer')
#
# total_data.to_csv('C:/Users/dell/Desktop/total_data.csv', encoding='utf_8_sig')
def time_convert():
    orders_store = pd.read_csv('C:/Users/dell/Desktop/orders_store.csv',encoding='utf_8_sig')
    def date_convert(x):
        num = re.findall('\d+', x['TIME_DATE'])
        w=num[0]
        y= num[1]
        z=num[2]
        date = str(20) + str(z) + '-' + str(y) + '-' + str(w)
        return date
    orders_store['TIME_DATE'] = orders_store.apply(lambda x: date_convert(x), axis=1)
    orders_store['TIME_DATE'] = pd.to_datetime(orders_store['TIME_DATE']).dt.normalize()
    orders_store.to_csv('C:/Users/dell/Desktop/orders_store_new.csv', encoding='utf_8_sig')

def merge_data():
    store_sales = pd.read_excel('C:/Users/dell/Desktop/sales_store_new.xlsx',encoding='utf_8_sig')
    store_orders = pd.read_csv('C:/Users/dell/Desktop/orders_store_new.csv',encoding='utf_8_sig')
    store_sales['SUBSTR(BS.FILDATE,1,10)'] = pd.to_datetime(store_sales['SUBSTR(BS.FILDATE,1,10)']).dt.normalize()
    store_orders['TIME_DATE'] = pd.to_datetime(store_orders['TIME_DATE']).dt.normalize()

    store_sales.rename(index=str,columns = {'SUBSTR(BS.FILDATE,1,10)':'TIME_DATE','名称':'STOREANAME'}, inplace=True)
    print(store_sales)
    total_data = pd.merge(store_sales,store_orders,on=['STOREANAME','TIME_DATE'],how='inner')

    print(store_sales)
    total_data.to_csv('C:/Users/dell/Desktop/total_data.csv', encoding='utf_8_sig')

total_data= pd.read_csv('C:/Users/dell/Desktop/total_data_new.csv',encoding='utf_8_sig')
total_data = total_data[['STOREANAME', 'ADDRESS', 'CITY', 'PROVINCE', 'COUNTY', 'SUM(BS.SCRTOTAL)', 'SUM(BS.REALAMT)', 'SUM(BS2.IAMT)', 'SUM(BS2.ITAX)', 'COUNT(BS.FLOWNO)', 'TIME_DATE', 'TOTAL_PRICE', 'OLD_QTY', 'MID_QTY', 'LAST_QTY', 'PRICE']]
print(total_data)
total_data = total_data.drop_duplicates(subset=['TIME_DATE','TOTAL_PRICE'], keep='first', inplace=False)
total_data.to_csv('C:/Users/dell/Desktop/total_data_last.csv', encoding='utf_8_sig')
