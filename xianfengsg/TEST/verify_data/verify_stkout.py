# -*- coding: utf-8 -*-
# @Time    : 2019/11/5 19:25
# @Author  : Ye Jinyu__jimmy
# @File    : verify_stkout

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

'''该脚本是用于检测hd主库的仓库销售数据与产品中心每日销售数据的对比情况，产品中心的采购更新模块作为初始化的选择源头
通过选择mysql数据库内的最小规格的出库数量'''


#首先获取对应的满足要求的数据格式，和对应的日期时间
def read_excel(path):
    data_xls = pd.ExcelFile(path)
    df=data_xls.parse(sheet_name='采购更新模块',header=1,usecols=[0,1,2,3,5,6,8,9],
                      converters={u'订货编码': str})
    df = df.dropna(axis=0, how='any')
    df['最小规格对应的门店订货数量'] = df['订货箱数'] * df['规格']
    date = data_xls.parse(sheet_name='采购更新模块',header=0,usecols=[4])
    label = date.columns.values[0]
    print(label)
    date_str = label[5:]
    today = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    yesterday = (today - datetime.timedelta(1)).strftime('%Y%m%d')
    dt = today.strftime('%Y%m%d')
    return df,yesterday,dt




#依据gdgid获取每个sku实际的销售数量
def get_detail_sales_data(sku_id,start_date,end_date,DC_CODE):
    print('连接到mysql服务器...，正在读取销售数据')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123", port=3306, charset='utf8')
    # 查看出货详细单的数据
    stkout_detail_sql = """SELECT QTY,GDGID FROM sales_his sh  WHERE sh.Ocrdate >= DATE('%s') 
AND sh.Ocrdate <= DATE('%s') AND sh.GDGID = %s AND sh.SENDER= %s""" % \
                        (start_date,end_date,sku_id,DC_CODE)
    db.cursor()
    read_orignal_forecast = pd.read_sql(stkout_detail_sql, db)
    db.close()
    print(str(sku_id)+'销售数据读取完成')
    return read_orignal_forecast


#拿到对应的商品的五位code码，根据五位吗找到对应的七位码
#-----------------------------------------------------------将五位码转成7位码
def get_7th_code(i):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    #读取的数据包括销量的时间序列，天气和活动信息
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", parameters)
    #查看详细的出库数据，进行了日期的筛选，查看销量签50名的SKU
    goods_sql = """SELECT * FROM GOODSH g WHERE g.CODE = '%s'""" %(i,)
    goods = pd.read_sql(goods_sql, conn)
    conn.close
    sku_id = goods['GID'].iloc[0]
    return sku_id

#获取到每个sku实际的订货数量
def get_his_sales(df,start_date,end_date):
    sku_code = df[['订货编码']]
    sku_code = sku_code.dropna(axis=0,how='any')
    sku_id_list = sku_code['订货编码'].to_list()
    print(sku_id_list)
    #data为获取的所有需要预测的sku的历史销售数据
    data = pd.DataFrame()
    for sku_gid in tqdm(sku_id_list):
        print(sku_gid)
        sku_id = get_7th_code(sku_gid)
        stkout_detail = get_detail_sales_data(sku_id,start_date,end_date,1000255)
        stkout_detail['订货编码'] = pd.Series(sku_gid)
        if stkout_detail.empty == True:
            print('sku_id：'+str(sku_gid)+'未获取到销售数据')
            pass
        else:
            data = data.append(stkout_detail)
    return data


data,yesterday,dt = read_excel('D:/jimmy-ye/AI/AI_supply_chain/product_design/'
                  'DATA_SOURCE/product_center_rel/浙北仓库存模板10.24(1).xlsx')
sales_sql = get_his_sales(data,yesterday,dt)
result = pd.merge(sales_sql,data,on=['订货编码'])
result.to_csv('D:/jimmy-ye/AI/AI_supply_chain/V1.0/result_start_up_10.21/compare_real_sql_merge_data.csv',
                  encoding='utf_8_sig')