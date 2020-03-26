# -*- coding: utf-8 -*-
# @Time    : 2019/12/19 9:40
# @Author  : Ye Jinyu__jimmy
# @File    : main_function

import pymysql
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from matplotlib import pyplot
from math import sqrt
from random import randint
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
import os
# 注：设置环境编码方式，可解决读取数据库乱码问题
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
import get_holiday_inf_2019
import get_holiday_inf_2018
import get_holiday_inf_2020
import psycopg2
import datetime
from tqdm import tqdm
import cx_Oracle
import process_data
'''设置一个主函数用于分别选择门店，然后针对每个门店进行学习'''

#-----------------------------*获取所有门店代码*---------------------------------

def read_oracle_data(code): #传入的参数是目前16个城市公司的代码
    host = "172.16.253.250"  # 数据库ip
    port = "1521"  # 端口
    sid = "HDBI"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    #读取的数据包括销量的时间序列，天气和活动信息
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("HDBI", "HDBI54325000", parameters)
    #查看详细的出库数据，进行了日期的筛选，查看销量签50名的SKU
    store_code_sql = """ SELECT os.CODE,os.NAME,os.CITY FROM ODS_STORE os where os.area like '%s%%' """ %(code)
    print(store_code_sql)
    store_code = pd.read_sql(store_code_sql, conn)
    #将SKU的的iD转成list，并保存前80个，再返回值
    conn.close
    return store_code




code = 3301
region='杭州'
store_code= read_oracle_data(code)
print(store_code)

#--------------------------------------*单个门店单独预测*--------------------------，
def predict_each_store(store_code,end_date,region):
    code_list = set(store_code['CODE'])
    for code in code_list:
        name = store_code[store_code['CODE']==code]['NAME'].iloc[0]
        city_01 = store_code[store_code['CODE']==code]['CITY'].iloc[0]
        city_01 = city_01.split('市')
        city = city_01[0]
        print(city,name)
        training_data, predict_data = process_data.main(end_date,city,region)

        print(training_data)
        training_data.to_csv('./training_data.csv',encoding='utf_8_sig')



today = datetime.date.today()
end_date = today.strftime('%Y%m%d')
date_7days = (today - datetime.timedelta(7)).strftime('%Y%m%d')
predict_each_store(store_code,end_date,date_7days,region)
