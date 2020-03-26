# -*- coding: utf-8 -*-
# @Time    : 2019/11/7 9:40
# @Author  : Ye Jinyu__jimmy
# @File    : evaluation_human_ai

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
import XGBOOST_forecast
import decision_repl

import importlib,sys
importlib.reload(sys)
LANG="en_US.UTF-8"
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

'''该脚本是为了对比人工下单和AI下单的实际对比,时间开始的时间是2019年10月21日'''

#设置函数用来生成和保存每个的对应的日志信息
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print(
            "----生成新的文件目录----")
    else:
        print("当前文件夹已经存在")


def print_in_log(string):
    print(string)
    date_1 = datetime.datetime.now()
    str_10 = datetime.datetime.strftime(date_1, '%Y%m%d')
    file = open('./' + str(str_10) + '/' + 'log' + str(str_10) + '.txt', 'a')
    file.write(str(string) + '\n')


#--------------------------获取oracle主库里面实际下单的情况
def get_real_orders(yes_date,today):
    dbconn = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com", database="purchare_sys",
                                 user="purchare_sys",password="purchare_sys@123",port = 3306,
                                 charset='utf8')
    print_in_log('采购2.0，MYSQL数据库连接成功,正在获取人工的补货数据')
    get_orders = """SELECT DATE_FORMAT(e.create_time,'%%Y-%%m-%%d') Account_date,e.received_warehouse_name Warehouse_name ,
                    a.group_name ,a.goods_code Code,
                    case when e.is_urgent=0 then '不是' when e.is_urgent=1 then '是' end urgent,sum(a.amount) real_orders
                    from p_warehouse_order_dtl a 
                    left join p_warehouse_order e on e.id=a.warehouse_order_id
                    LEFT JOIN (
                    select b.warehouse_order_id,b.warehouse_order_dtl_id,a.plan_order_id id,b.id dtlid from p_purchase_plan_order a,p_purchase_plan_order_dtl b
                    where a.plan_order_id=b.p_purchase_plan_order_id
                    ) b on a.warehouse_order_id=b.warehouse_order_id and a.id=b.warehouse_order_dtl_id
                    where a.amount<>0 AND e.create_time > date ('%s') AND e.create_time < date ('%s') AND 
                    e.received_warehouse_name='杭州配送商品仓'
                    group by DATE_FORMAT(e.create_time,'%%Y-%%m-%%d'),e.received_warehouse_name,a.group_name,
                    a.goods_code,a.goods_name,e.is_urgent""" \
                 %(yes_date,today)
    orders= pd.read_sql(get_orders,dbconn)
    orders['Code'] = orders['Code'].astype(str)
    def polishing(x):
        x['Code'].rjust(5, '0')
    orders['Code'] = orders.apply(lambda x: polishing(x), axis=1)
    dbconn.close()
    print_in_log('人工的补货数据读取完成，并关闭了服务器的连接')
    return orders


#-----------------------------------------该函数是用来获取AI的补货数据
def get_AI_replenishment(today_date):
    print_in_log('连接到mysql服务器...')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123", port=3306, charset='utf8')
    print_in_log('连接成功,正在获取AI的补货建议')
    replenishment_sql = """SELECT Sku_id,Code,Warehouse_name,Sku_name,rate,Forecast_box,SS,Stock,Suggestion_qty,Account_date
     FROM dc_replenishment where Account_date = DATE('%s')""" % (today_date)
    db.cursor()
    read_replenishment_sql = pd.read_sql(replenishment_sql, db)
    read_replenishment_sql['Code'] = read_replenishment_sql['Code'].astype(str)
    def polishing(x):
        return x['Code'].rjust(5, '0')
    read_replenishment_sql['Code'] = read_replenishment_sql.apply(lambda x: polishing(x), axis=1)
    db.close()
    print_in_log('AI的补货建议读取完成，并关闭了服务器的连接')
    return read_replenishment_sql




#设置一个类用来按照约定的时间去run预测和决策算法

#===============================用来运行每次的forecast和decision程序，需要预测和决策输入写入数据库中
def run_forecast_decision(today,tomorrow):
    print(today)
    XGBOOST_forecast.main_function(today)
    print_in_log(str(today)+'的forecast已经完成')

    decision_repl.main(today)
    print_in_log(str(today) + '的decision已经完成')


#===============================用于循环2019年10月21日2019年11月7日的预测和决策的数据
def start_end_date(start, end):
    for days in tqdm(pd.date_range(start, end)):
        today = days.strftime('%Y%m%d')
        tomorrow = (days + datetime.timedelta(1)).strftime('%Y%m%d')
        print_in_log('正在进行forecast和decision的日期是：'+str(today))
        run_forecast_decision(today,tomorrow)

start_end_date('20191021','20191107')











#用于简单的测试，先将历史的仓库的出库数据从本地读取并存入到mysql的数据库中

def connectdb():
    print_in_log('连接到mysql服务器...')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123",port=3306, charset='utf8')
    print_in_log('连接成功')
    return db

#《-------------------------------------------------------------------------------------删除重复日期数据
def drop_data(db):
    cursor = db.cursor()
    # sql = """delete from sales_his where Ocrdate = DATE ('%s')"""%(yes_date)
    sql = """delete from dc_forecast"""
    sql_replenishment = """delete from dc_replenishment"""
    cursor.execute(sql)
    cursor.execute(sql_replenishment)
    print('历史重复数据删除成功')

# db= connectdb()
# drop_data(db)
# db.close()

def insert_sales_db(db,data):
    cursor = db.cursor()
    data_list = data.values.tolist()
    print(data_list)
    print_in_log(str(len(data_list)))
    sql = """INSERT INTO sales_his (SENDER,DC_NAME,GID,
    WAREHOUSE_NAME,NUM,GDGID,SKU_NAME,OCRDATE,CRTOTAL,MUNIT
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

def main():
    data = pd.read_excel('D:/jimmy-ye/AI/AI_supply_chain/V1.0/result_start_up_10.21/sales_DC.xlsx',
                         encoding='utf_8_sig',converters={u'Ocrdate': str})
    print(data)
    db = connectdb()
    # print_in_log('历史销售数据的总长度'+str(len(sales_his)))
    drop_data(db)
    if data.empty:
        print_in_log("The data frame is empty")
        print_in_log("result:1")
        closedb(db)
    else:
        insert_sales_db(db,data)
        closedb(db)
        print_in_log("result:1")
        print_in_log("result:所有数据插入mysql数据库成功")


