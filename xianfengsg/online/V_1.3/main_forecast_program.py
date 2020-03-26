# -*- coding: utf-8 -*-
# @Time    : 2019/12/26 10:29
# @Author  : Ye Jinyu__jimmy
# @File    : main_forecast_program

import pandas as pd
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
from sklearn import preprocessing
import numpy as np
import time
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,train_test_split
import features_engineering
import cx_Oracle
import datetime
import pymysql
import tqdm
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
import psycopg2
import XGBOOST_forecast_engineer

'''该脚本函数是用于将分别对每个城市公司的销售情况进行总预测是主调度逻辑'''
#--------------------------------------设置函数获取
def get_wh_list():
    conn = psycopg2.connect(database="dc_rpt", user="ads", password="ads@xfsg2019", host="192.168.1.205",
                            port="3433")
    print("Opened database successfully,connected with PG DB")
    wh_code_sql = """SELECT wh_code,wh_name FROM ads_aig_supply_chain.ads_rpt_ai_wh_d GROUP BY wh_code,wh_name """
    try:
        wh_df = pd.read_sql(wh_code_sql, conn)
    except:
        print("load data from postgres failure !")
        wh_df = pd.DataFrame()
        exit()
    conn.close()
    print(wh_df)
    return wh_df



#设置函数用于循环进行每个配送中心的单独运算
def main_function(start_date,end_date,date_7days,date_7days_after):
    wh_list = get_wh_list()
    num = len(wh_list)
    print('一共有%d个城市公司进行计算'%num)
    # for i in range(num):
    i = 0
    wh_code = wh_list['wh_code'].iloc[i]
    print(start_date, end_date, date_7days, wh_code)
    print('正在进行的城市是和代码是：', wh_list['wh_name'].iloc[i],wh_code)
    XGBOOST_forecast_engineer.main(start_date,end_date,date_7days,wh_code,date_7days_after)
    print( wh_list['wh_name'].iloc[i],'计算完成')
    print('所有城市计算完成')


if __name__ == '__main__':
    # today = datetime.date.today()
    # end_date = today.strftime('%Y%m%d')
    start = '20191201'
    end= '20200115'
    for days in pd.date_range(start, end):
        date_7days = (days - datetime.timedelta(7)).strftime('%Y%m%d')
        date_7days_after = (days + datetime.timedelta(7)).strftime('%Y%m%d')
        day = days.strftime('%Y%m%d')
        start_date = '20180101'
        try:
            main_function(start_date,day,date_7days,date_7days_after)
        except OSError as reason:
            print('出错原因是%s' % str(reason))
            print ("result:0")