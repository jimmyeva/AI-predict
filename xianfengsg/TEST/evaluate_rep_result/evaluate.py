# -*- coding: utf-8 -*-
# @Time    : 2019/11/28 10:32
# @Author  : Ye Jinyu__jimmy
# @File    : evaluate

import cx_Oracle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pymysql
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
import warnings
warnings.simplefilter("ignore")
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
'''这个脚本是看AI建议的数量和实际补货的数量，同时还需要把库存信息进行合并'''

#读取AI建议的订货量
def get_AI_repl():
    print('连接到mysql服务器...，正在读取AI建议数据')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123", port=3306, charset='utf8')
    # 查看出货详细单的数据
    AI_REL_sql = """SELECT * FROM dc_replenishment dr WHERE dr.Account_date > DATE('20191120')"""
    db.cursor()
    AI_repl = pd.read_sql(AI_REL_sql, db)
    db.close()
    return AI_repl

#获取人工的实际订货量
def get_human_repl():
    print('连接到mysql服务器...，正在读取人工实际订货数据')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123", port=3306, charset='utf8')
    # 查看出货详细单的数据
    human_REL_sql = """SELECT e.create_time Account_date,e.received_warehouse_name 仓位,a.group_name 采购组,a.goods_code Code,
                        sum(a.amount) 订单箱数 from p_warehouse_order_dtl a
                        left join p_warehouse_order e on e.id=a.warehouse_order_id
                        LEFT JOIN (select b.warehouse_order_id,b.warehouse_order_dtl_id,a.plan_order_id id,b.id dtlid from 
                        p_purchase_plan_order a,p_purchase_plan_order_dtl b 
                        where a.plan_order_id=b.p_purchase_plan_order_id) b on a.warehouse_order_id=b.warehouse_order_id 
                        and a.id=b.warehouse_order_dtl_id
                        where a.amount<>0 AND e.create_time > DATE('2019-11-21') AND e.received_warehouse_name='杭州配送商品仓' 
                        group by DATE_FORMAT(e.create_time,'%Y-%m-%d %H-%M-%M'),e.received_warehouse_name,a.group_name,
                        a.goods_code,a.goods_name,e.is_urgent"""
    db.cursor()
    human_repl = pd.read_sql(human_REL_sql,db)
    db.close()
    return human_repl


#对人工补货的数据进行合并和数据整理,并进行和并操作并返回对应的dataframe
def process_01():
    human_repl = get_human_repl()
    human_repl['Account_date'] = human_repl['Account_date'].dt.date
    human_repl = human_repl.groupby(["Account_date",'仓位','采购组','Code'],as_index = False).agg(sum)
    print(human_repl)

    AI_repl = get_AI_repl()
    AI_repl['Account_date'] = AI_repl['Account_date'].dt.date

    #进行数据合并的操作,将AI的补货建议和人工的实际进行合并，注意一点是只进行测试了十款sku
    reslut = pd.merge(AI_repl,human_repl,on=['Account_date','Code'],how='left')
    return reslut


#再读取一下库存的数据信息
def get_stock():
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    #读取的数据包括销量的时间序列，天气和活动信息
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", parameters)
    #查看详细的出库数据，进行了日期的筛选，查看销量签50名的SKU
    goods_sql = """SELECT GOODS_CODE Code,DIFFERENCE_QTY, 
    FILDATE Account_date, WAREHOUSE,INVENTORY FROM DC_hangzhou_inv
    WHERE FILDATE >=  to_date('2019-11-20','yyyy-mm-dd')"""
    goods = pd.read_sql(goods_sql, conn)
    goods.dropna(axis=0, how='any', inplace=True)
    conn.close
    return goods

#需要查看库存的信息，每日的门店订货数据
stock = get_stock()
stock['store_orders'] = stock['INVENTORY'] - stock['DIFFERENCE_QTY']
stock.rename(columns={'CODE':'Code','ACCOUNT_DATE':'Account_date'},inplace=True)
print(stock)
stock['Account_date'] = stock['Account_date'].dt.date
result = process_01()
final = pd.merge(result, stock, on=['Account_date', 'Code'], how='left')
final.to_csv('./final.csv', encoding='utf_8_sig')

