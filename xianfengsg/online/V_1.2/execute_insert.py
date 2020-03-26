# -*- coding: utf-8 -*-
# @Time    : 2019/10/15 11:26
# @Author  : Ye Jinyu__jimmy
# @File    : execute_insert

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

'''此脚本是用于将历史的天气信息存入mysql数据库中'''

#生成日志文件
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print(
            "----生成新的文件目录----")
    else:
        print(
            "当前文件夹已经存在")


def print_in_log(string):
    print(string)
    date_1 = datetime.datetime.now()
    str_10 = datetime.datetime.strftime(date_1, '%Y%m%d')
    file = open('./' + str(str_10) + '/' + 'log' + str(str_10) + '.txt', 'a')
    file.write(str(string) + '\n')


# def mkdir(path):
#     folder = os.path.exists('/root/AI_SC/AI_SC_V_1.0/')
#     if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
#         os.makedirs('/root/AI_SC/AI_SC_V_1.0/')  # makedirs 创建文件时如果路径不存在会创建这个路径
#         print(
#             "----生成新的文件目录----")
#     else:
#         print(
#             "当前文件夹已经存在")
#
#
# def print_in_log(string):
#     print(string)
#     date_1 = datetime.datetime.now()
#     str_10 = datetime.datetime.strftime(date_1, '%Y%m%d')
#     file = open('/root/AI_SC/AI_SC_V_1.0/test.txt', 'a')
#     file.write(str(string) + '\n')



#读取天气数据
def read_weather():
    read_data = pd.read_csv('D:/jimmy-ye/AI/AI_supply_chain/data/CN101210101-小时.csv', encoding='utf_8_sig')
    read_data= read_data.drop(['风力'], axis=1)
    read_data = read_data.rename(index=str, columns={'日期': 'Date','小时':'Hour','温度':'Temperature',
                                                     '体感温度': 'Feel_tem','天气状况代码':'Wather_code','天气状况名称':'Weather_description',
                                                     '湿度': 'Humidity','降雨量':'Rain','大气压':'Pressure',
                                                     '能见度': 'Visability','风向':'Wind_direction','风向角度':'Wind_angle'
                                                     ,'风速':'Wind_speed'})
    read_data['Update_time'] = datetime.date.today().strftime('%Y-%m-%d')
    print_in_log('天气数据读取完成，总长度:'+ str(len(read_data)))
    return read_data


'''读取销售数据，存入对应mysql数据库'''
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

'''如下是根据实际仓库发货的情况进行数据获取'''
#------------------------------------------------------------------>根据SKU 的id来获取每个SKU的具体的销售明细数据
# def get_detail_sales_data(sku_id,start_date,end_date,DC_CODE):
#     host = "192.168.1.11"  # 数据库ip
#     port = "1521"  # 端口
#     sid = "hdapp"  # 数据库名称
#     dsn = cx_Oracle.makedsn(host, port, sid)
#     # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码)
#     conn = cx_Oracle.connect("hd40", "xfsg0515pos", dsn)
#     # 查看出货详细单的数据
#     stkout_detail_sql = """SELECT ss.sender,s.name Dc_name,wrh.GID,wrh.NAME warehouse_name
#         ,max(ss.num) AS NUM,b.gdgid,G.NAME sku_name,TRUNC(c.time) OCRDATE,SUM(b.CRTOTAL) CRTOTAL,
#         MAX(b.munit) munit,SUM(b.qty) qty,max(b.QTYSTR) QTYSTR
#         ,SUM(b.TOTAL) TOTAL,SUM(b.price) price,max(b.qpc) qpc,SUM(b.RTOTAL) RTOTAL
#         FROM stkout ss ,stkoutdtl b, stkoutlog c,store s ,goods g,warehouseh wrh
#         where ss.num = b.num  and ss.num =c.num and b.gdgid=g.gid and ss.sender =s.gid
#         and ss.cls='统配出' and ss.cls=c.cls and ss.cls=b.cls and ss.wrh = wrh.gid
#         and c.stat IN ('700','720','740','320','340')
#        and c.time  >=  to_date ('%s','yyyy-mm-dd')
#         and c.time <=  to_date('%s','yyyy-mm-dd')
#         and b.GDGID = %s AND wrh.NAME LIKE'%%商品仓%%' AND ss.SENDER= %s GROUP BY ss.sender,TRUNC(c.time),s.name
#                                         ,wrh.GID,wrh.NAME,b.gdgid,G.NAME """ % \
#                         (start_date,end_date,sku_id,DC_CODE)
#     stkout_detail = pd.read_sql(stkout_detail_sql, conn)
#     print_in_log('销售数据读取完成，总长度:' + str(len(stkout_detail)))
#     conn.close
#     return stkout_detail


'''以下是根据门店的实际销售作为主要数据源进行数据的获取'''
def get_detail_sales_data(sku_code,start_date,end_date):
    host = "172.16.253.250"  # 数据库ip
    port = "1521"  # 端口
    sid = "HDBI"  # 数据库名称
    dsn = cx_Oracle.makedsn(host, port, sid)
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码)
    conn = cx_Oracle.connect("HDBI", "HDBI54325000", dsn)
    # 查看出货详细单的数据
    stkout_detail_sql = """SELECT ossd.GDCODE,TRUNC(ossd.FILDATE) Ocrdate,sum(ossd.GDSALE) Rtotal,SUM(ossd.GDCOST) Price,
  sum(ossd.GDQTY) Qty FROM ods_sale_store_dtl ossd INNER JOIN(SELECT * FROM ODS_STORE os  where os.area like '3301%%')b 
            ON b.CODE=ossd.STCODE WHERE ossd.GDCODE  in %s 
            AND  ossd.FILDATE>=  to_date('%s','yyyy-mm-dd') 
            and ossd.FILDATE <  to_date('%s','yyyy-mm-dd')
            GROUP BY ossd.GDCODE,TRUNC(ossd.FILDATE) ORDER BY TRUNC(ossd.FILDATE) """ % \
                        (sku_code,start_date,end_date)
    stkout_detail = pd.read_sql(stkout_detail_sql, conn)
    stkout_detail = stkout_detail.groupby(['OCRDATE'], as_index=False).agg(sum)
    print(stkout_detail)
    print_in_log('销售数据读取完成，总长度:' + str(len(stkout_detail)))
    conn.close
    return stkout_detail


#对获取门店销售的数据进行清洗和格式的转换
def cleaning_sales(data):
    data['SENDER'] = '1000255'
    data['Dc_name'] = '杭州配送中心'
    data['Gid'] = '1000008'
    data['Warehouse_name'] = '杭州配送中心商品仓'
    data['Num'] = '1000255'
    data['Crtotal'] = '1000255'
    data['Munit'] = '箱'
    data['Qtystr'] = '1000255'
    data['Qpc'] = '1000255'
    data['Total'] = '1000255'
    return data

#将读取数据和简单的数据清洗进行封装处理
def process_sales_data(sku_gid,sku_code,sku_name,start_date,end_date):
    stkout_detail = get_detail_sales_data(sku_code,start_date,end_date)
    sales_data = cleaning_sales(stkout_detail)
    sales_data['GDGID'] = sku_gid
    sales_data['Sku_name'] = sku_name
    # sales_data = sales_data.drop(['GDCODE'],axis=1)
    return sales_data

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
    print_in_log('商品子母码读取完成:' + str(len(get_kids_code)))
    sku_id_list = get_kids_code['G2CODE'].to_list()
    code_tuple = tuple(sku_id_list)
    conn.close
    return code_tuple


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
        print_in_log('sku_id'+ str(i))
        code_tuple = mother_code(sku_code)
        if len(code_tuple)>0:
            sales_data = process_sales_data(sku_gid,code_tuple,sku_name,start_date,end_date)
            if sales_data.empty == True:
                print_in_log('sku_id：'+str(sku_gid) +'未获取到销售数据')
                pass
            else:
                data = data.append(sales_data)
        else:
            pass
    data['OCRDATE'] = pd.to_datetime(data['OCRDATE'], unit='s').dt.strftime('%Y-%m-%d')
    return data


def connectdb():
    print_in_log('连接到mysql服务器...')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123",port=3306, charset='utf8')
    print_in_log('连接成功')
    return db


#《-------------------------------------------------------------------------------------删除重复日期数据
def drop_data(db,yes_date):
    cursor = db.cursor()
    sql = """delete from sales_his where Ocrdate = DATE ('%s')"""%(yes_date)
    # sql = """delete from sales_his """
    print('历史重复数据删除成功')
    cursor.execute(sql)


#<======================================================================================================================
def insertdb(db,data):
    cursor = db.cursor()
    # param = list(map(tuple, np.array(data).tolist()))
    data_list = data.values.tolist()
    print_in_log(str(len(data_list)))
    sql = """INSERT INTO weather_history (Date,Hour,Temperature,
    Feel_tem,Wather_code,Weather_description,Humidity,Rain,Pressure,Visability,
    Wind_direction,Wind_angle,Wind_speed,Update_time)
     VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    try:
        cursor.executemany(sql, data_list)
        print_in_log("所有品牌的sku数据插入数据库成功")
        db.commit()
    except OSError as reason:
        print_in_log('出错原因是%s' % str(reason))
        db.rollback()


def insert_sales_db(db,data):
    cursor = db.cursor()
    data_list = data.values.tolist()
    print(data_list)
    print_in_log(str(len(data_list)))
    sql = """INSERT INTO sales_his (OCRDATE,RTOTAL,PRICE,
    QTY,SENDER,DC_NAME,GID,WAREHOUSE_NAME,NUM,
    CRTOTAL,MUNIT,QTYSTR,QPC,TOTAL,GDGID,SKU_NAME)
     VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    try:
        cursor.executemany(sql, data_list)
        print_in_log("所有历史数据的sku数据插入数据库成功")
        db.commit()
    except OSError as reason:
        print_in_log('出错原因是%s' % str(reason))
        db.rollback()


#<=============================================================================关闭
def closedb(db):
    db.close()


#<=============================================================================
def main():
    yes_date = (datetime.date.today() - datetime.timedelta(1)).strftime('%Y%m%d')
    today_date = datetime.date.today().strftime('%Y%m%d')
    mkdir(today_date)
    sales_his = get_his_sales(yes_date, today_date)
    print(sales_his)
    db = connectdb()
    print_in_log('历史销售数据的总长度'+str(len(sales_his)))
    drop_data(db,yes_date)
    if sales_his.empty:
        print_in_log("The data frame is empty")
        print_in_log("result:1")
        closedb(db)
    else:
        insert_sales_db(db,sales_his)
        closedb(db)
        print_in_log("result:1")
        print_in_log("result:所有数据插入mysql数据库成功")


#《============================================================================主函数入口
if __name__ == '__main__':
    try:
        main()
    except OSError as reason:
        print_in_log('出错原因是%s'%str(reason))
        print_in_log ("result:0")