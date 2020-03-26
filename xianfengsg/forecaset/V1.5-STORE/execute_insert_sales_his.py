# -*- coding: utf-8 -*-
# @Time    : 2019/12/9 13:55
# @Author  : Ye Jinyu__jimmy
# @File    : execute_insert_sales_his.py

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

'''这个操作是进行门店的销售数据的获取，简单清洗的操作'''




#---------------------------------------读取数据找对杭州配送中心对应的订货目录数据
def get_order_code():
    dbconn = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com", database="purchare_sys",
                             user="purchare_sys", password="purchare_sys@123", port=3306,
                             charset='utf8')
    get_orders = """SELECT pcr.goods_id GOODS_GID,pcr.goods_code GOODS_CODE,pcr.goods_name FROM p_call_record pcr 
    WHERE pcr.warehouse_id ='1000255' GROUP BY pcr.goods_id"""
    orders = pd.read_sql(get_orders, dbconn)
    orders['GOODS_GID'] = orders['GOODS_GID'].astype(str)
    orders['GOODS_CODE'] = orders['GOODS_CODE'].astype(str)
    dbconn.close()
    return orders



#由于涉及到子母码的问题，对于门店会使用母码进行采购，但是用子码进行销售
def mother_code(mother):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    dsn = cx_Oracle.makedsn(host, port, sid)
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码)
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", dsn)
    # 查看售价当前表
    get_kids_code_sql = """SELECT rdp.G2CODE FROM RPT_DZV_PSCPS rdp WHERE rdp.G1CODE= '%s'""" % (mother)
    get_kids_code = pd.read_sql(get_kids_code_sql, conn)
    print('商品子母码读取完成:' + str(len(get_kids_code)))
    sku_id_list = get_kids_code['G2CODE'].to_list()
    code_tuple = tuple(sku_id_list)
    conn.close
    return code_tuple




#--------------------------------------------先读取基本的销售数据
def get_original_sales(sku_code,start_date,end_date):
    host = "172.16.253.250"  # 数据库ip
    port = "1521"  # 端口
    sid = "HDBI"  # 数据库名称
    dsn = cx_Oracle.makedsn(host, port, sid)
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码)
    conn = cx_Oracle.connect("HDBI", "HDBI54325000", dsn)
    print('正在读取销售数据...')
    # 查看出货详细单的数据
    stock_sales_detail_sql = """ SELECT ossd.STCODE,ossd.GDCODE,ossd.FILDATE as Ocrdate,ossd.GDQTY,ossd.GDSALE FROM ods_sale_store_dtl ossd
          WHERE  ossd.STCODE IN (SELECT os.CODE FROM ODS_STORE os where os.area like '3301%%')
          AND ossd.FILDATE >=  to_date('%s','yyyy-mm-dd')
          and ossd.FILDATE <  to_date('%s','yyyy-mm-dd')
          and ossd.GDCODE  in %s """ % \
                        (start_date, end_date,sku_code)
    store_sales_detail = pd.read_sql(stock_sales_detail_sql, conn)

    store_sales_detail['OCRDATE'] = pd.to_datetime(store_sales_detail['OCRDATE'], unit='s').dt.strftime('%Y-%m-%d')

    stkout_detail = store_sales_detail.groupby(['STCODE','OCRDATE','GDCODE'], as_index=False).agg(sum)
    print('销售数据读取完成，总长度:' + str(len(stkout_detail)))
    conn.close
    return stkout_detail


#对获取门店销售的数据进行清洗和格式的转换
def cleaning_sales(data):
    data = data.rename(index=str, columns={'STCODE': 'SENDER','GDQTY': 'Qty','GDSALE': 'Total','GDCODE':'Price'})
    data['Dc_name'] = '杭州配送中心'
    data['Num'] = '1000255'
    data['Crtotal'] = '1000255'
    data['Munit'] = '箱'
    data['Qtystr'] = '1000255'
    data['Qpc'] = '1000255'
    data['Rtotal'] = '1000255'
    return data

#将读取数据和简单的数据清洗进行封装处理
def process_sales_data(sku_gid,sku_code,sku_name,start_date,end_date):
    stkout_detail = get_original_sales(sku_code,start_date,end_date)
    sales_data = cleaning_sales(stkout_detail)
    sales_data['GDGID'] = sku_gid
    sales_data['Sku_name'] = sku_name
    # sales_data = sales_data.drop(['GDCODE'],axis=1)
    return sales_data


def get_his_sales(start_date,end_date):
    data_sku = get_order_code()
    '''删除任何有空置的情况'''
    sku_list = data_sku.values.tolist()
    #data为获取的所有需要预测的sku的历史销售数据
    data = pd.DataFrame()
    for i in tqdm(sku_list):
        sku_gid = i[0]
        sku_code = i[1]
        sku_name = i[2]
    # sku_gid= '3000004'
    # sku_code ='06630'
    # sku_name ='台湾红心芭乐(叫货)'

        code_tuple = mother_code(sku_code)
        print('sku_id'+ str(sku_code))
        print('code_tuple',code_tuple)
        if len(code_tuple)>0:
            sales_data = process_sales_data(sku_gid,code_tuple,sku_name,start_date,end_date)
            if sales_data.empty == True:
                print('sku_id：'+str(sku_gid) +'未获取到销售数据')
                pass
            else:
                data = data.append(sales_data)
        else:
            pass
    return data

#-------------------------依旧商品的code判断
def get_store_name(store_code):
    host = "172.16.253.250"  # 数据库ip
    port = "1521"  # 端口
    sid = "HDBI"  # 数据库名称
    dsn = cx_Oracle.makedsn(host, port, sid)
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码)
    conn = cx_Oracle.connect("HDBI", "HDBI54325000", dsn)
    print('正在读取销售数据...')
    # 查看出货详细单的数据
    store_name_sql = """SELECT os.GID,os.NAME FROM ODS_STORE os  where os.CODE = '%s' """ % \
                             (store_code)
    store_name = pd.read_sql(store_name_sql, conn)
    if store_name.empty ==True:
        stname, stgid = '无数据','无数据'
    else:
        stname = store_name['NAME'].iloc[0]
        stgid = store_name['GID'].iloc[0]
    print('销售数据读取完成，总长度:' + str(len(store_name)))
    conn.close
    return stname,stgid


#设置函数用于读取和保存
def store_name_final(data):
    Stcode_list = data['SENDER'].values.tolist()
    final_data = pd.DataFrame()
    for store_code in tqdm(Stcode_list):
        stname,stgid = get_store_name(store_code)
        mid_data = data[data['SENDER'] == store_code]
        mid_data['Warehouse_name'] = stname
        mid_data['Gid'] = stgid
        final_data = final_data.append(mid_data)
    return final_data



def connectdb():
    print('连接到mysql服务器...')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123",port=3306, charset='utf8')
    print('连接成功')
    return db


def drop_data(db,STCODE):
    cursor = db.cursor()
    print('STCODE',STCODE)
    sql = """delete from sales_his where SENDER != ('1000255') """
    # sql = """delete from sales_his """
    print('历史重复数据删除成功')
    cursor.execute(sql)


def insert_sales_db(db,data):
    print(data)
    cursor = db.cursor()
    data_list = data.values.tolist()

    print(data_list)
    print(str(len(data_list)))
    sql = """INSERT INTO sales_his (SENDER,OCRDATE,Price,
    Qty,Total,Dc_name,Num,Crtotal,Munit,
    Qtystr,Qpc,Rtotal,GDGID,Sku_name,Warehouse_name,Gid)
     VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    try:
        cursor.executemany(sql, data_list)
        print("所有历史数据的sku数据插入数据库成功")
        db.commit()
    except OSError as reason:
        print('出错原因是%s' % str(reason))
        db.rollback()


#<=============================================================================关闭
def closedb(db):
    db.close()


#<=============================================================================
def main():
    start_date = '2018-01-01'
    end_date = '2018-06-01'

    data = get_his_sales(start_date, end_date)
    final_data = store_name_final(data)

    final_data.to_csv('./final_data.csv',encoding='utf_8_sig')
    db = connectdb()
    print('历史销售数据的总长度'+str(len(final_data)))

    STODE_LIST = final_data['SENDER'].to_list()
    code_tuple = tuple(STODE_LIST)
    drop_data(db,code_tuple)
    if final_data.empty:
        print("The data frame is empty")
        print("result:1")
        closedb(db)
    else:
        insert_sales_db(db,final_data)
        closedb(db)
        print("result:1")
        print("result:所有数据插入mysql数据库成功")


#《============================================================================主函数入口
if __name__ == '__main__':
    try:
        main()
    except OSError as reason:
        print('出错原因是%s'%str(reason))
        print ("result:0")


