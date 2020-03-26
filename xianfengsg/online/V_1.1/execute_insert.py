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
    folder = os.path.exists('/root/AI_SC/AI_SC_V_1.0/'+path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs('/root/AI_SC/AI_SC_V_1.0/'+path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print(
            "----生成新的文件目录----")
    else:
        print(
            "当前文件夹已经存在")


def print_in_log(string):
    print(string)
    date_1 = datetime.datetime.now()
    str_10 = datetime.datetime.strftime(date_1, '%Y%m%d')
    file = open('/root/AI_SC/AI_SC_V_1.0/' + str(str_10) + '/' + 'log' + str(str_10) + '.txt', 'a')
    file.write(str(string) + '\n')

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
    get_orders = """SELECT pcr.goods_id GOODS_CODE FROM p_call_record pcr WHERE pcr.warehouse_id ='1000255'GROUP BY pcr.goods_id """
    orders = pd.read_sql(get_orders, dbconn)
    orders['GOODS_CODE'] = orders['GOODS_CODE'].astype(str)
    dbconn.close()
    return orders


#------------------------------------------------------------------>根据SKU 的id来获取每个SKU的具体的销售明细数据
def get_detail_sales_data(sku_id,start_date,end_date,DC_CODE):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    dsn = cx_Oracle.makedsn(host, port, sid)
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码)
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", dsn)
    # 查看出货详细单的数据
    stkout_detail_sql = """SELECT ss.sender,s.name Dc_name,wrh.GID,wrh.NAME warehouse_name
        ,max(ss.num) AS NUM,b.gdgid,G.NAME sku_name,TRUNC(c.time) OCRDATE,SUM(b.CRTOTAL) CRTOTAL,
        MAX(b.munit) munit,SUM(b.qty) qty,max(b.QTYSTR) QTYSTR
        ,SUM(b.TOTAL) TOTAL,SUM(b.price) price,max(b.qpc) qpc,SUM(b.RTOTAL) RTOTAL
        FROM stkout ss ,stkoutdtl b, stkoutlog c,store s ,goods g,warehouseh wrh
        where ss.num = b.num  and ss.num =c.num and b.gdgid=g.gid and ss.sender =s.gid
        and ss.cls='统配出' and ss.cls=c.cls and ss.cls=b.cls and ss.wrh = wrh.gid
        and c.stat IN ('700','720','740','320','340')
       and c.time  >=  to_date ('%s','yyyy-mm-dd')
        and c.time <=  to_date('%s','yyyy-mm-dd')
        and b.GDGID = %s AND wrh.NAME LIKE'%%商品仓%%' AND ss.SENDER= %s GROUP BY ss.sender,TRUNC(c.time),s.name      
                                        ,wrh.GID,wrh.NAME,b.gdgid,G.NAME """ % \
                        (start_date,end_date,sku_id,DC_CODE)
    stkout_detail = pd.read_sql(stkout_detail_sql, conn)
    print_in_log('销售数据读取完成，总长度:' + str(len(stkout_detail)))
    conn.close
    return stkout_detail

def get_his_sales(start_date,end_date):
    df = get_order_code()
    sku_code = df[['GOODS_CODE']]
    '''只用于测试使用'''
    # sku_code = sku_code.loc[0:2]
    sku_code = sku_code.dropna(axis=0,how='any')
    sku_id_list = sku_code['GOODS_CODE'].to_list()
    gid_tuple = tuple(sku_id_list)
    print_in_log('gid_tuple'+ str(len(gid_tuple)))
    #data为获取的所有需要预测的sku的历史销售数据
    data = pd.DataFrame()
    for i in tqdm(gid_tuple):
        print_in_log('sku_id'+ str(i))
        stkout_detail = get_detail_sales_data(i,start_date,end_date,'1000255')
        if stkout_detail.empty == True:
            print_in_log('sku_id：'+str(i) +'未获取到销售数据')
            pass
        else:
            data = data.append(stkout_detail)
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
    sql = """delete from sales_his where Ocrdate = DATE ('%s') """%(yes_date)
    print_in_log('所有数据库ocrdate是昨日的数据已经删除完毕')
    print_in_log(str(sql))
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
    # param = list(map(tuple, np.array(data).tolist()))
    data_list = data.values.tolist()
    print_in_log(str(len(data_list)))
    sql = """INSERT INTO sales_his (sender,DC_NAME,GID,
    WAREHOUSE_NAME,NUM,GDGID,SKU_NAME,OCRDATE,CRTOTAL,MUNIT,
    QTY,QTYSTR,TOTAL,PRICE,QPC,RTOTAL)
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
