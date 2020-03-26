# -*- coding: utf-8 -*-
# @Time    : 2019/11/27 15:01
# @Author  : Ye Jinyu__jimmy
# @File    : evaluate_model_result

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
'''该程序包是用来评估预测模型的效果好坏，采用的指标是MAE和所有样本的MAE的均值进行衡量；设计思路，分别使用两种预测'''
#首先选择需要测试的日期,开始预测的日期
#
import XGBOOST_forecast_old
import XGBOOST_forecast_11_26


test_date = '2019-11-27'
def get_forecast(test_date):
    result = XGBOOST_forecast_old.main(test_date)
    result = result.rename(columns={'Forecast_qty':'Forecast_qty_old'})
    result = result[['Account_date', 'Sku_id', 'Forecast_qty_old', 'Sku_name']]
    result['Account_date'] = pd.to_datetime(result['Account_date'])
    result.to_csv('./result.csv',encoding='utf_8_sig')
    new_result = XGBOOST_forecast_11_26.main(test_date)
    new_result = new_result[['Account_date','Sku_id','Forecast_qty','Sku_name']]
    new_result['Account_date'] = pd.to_datetime(new_result['Account_date'])
    new_result.to_csv('./new_result.csv', encoding='utf_8_sig')
    final_data = pd.merge(new_result,result,on=['Account_date','Sku_id','Sku_name'],how='inner')
    return final_data


#以下需要根据设定的日期读取真实的销售数据
def get_real_sales(date):
    print('连接到mysql服务器...，正在读取销售数据')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123", port=3306, charset='utf8')
    # 查看出货详细单的数据
    stkout_detail_sql = """SELECT sh.GDGID as Sku_id,sh.Ocrdate as Account_date,sh.Qty,sh.Sku_name as Sku_name
                    FROM sales_his sh  WHERE sh.Ocrdate >= DATE('%s')""" % (date)
    print(stkout_detail_sql)
    db.cursor()
    read_orignal_forecast = pd.read_sql(stkout_detail_sql, db)
    read_orignal_forecast['Account_date'] = pd.to_datetime(read_orignal_forecast['Account_date'])
    db.close()
    print('销售数据读取完成')
    return read_orignal_forecast



read_orignal_forecast = get_real_sales(test_date)
final_data= get_forecast(test_date)
final = pd.merge(final_data,read_orignal_forecast,on=['Account_date','Sku_id'],how='inner')

final.to_csv('./new.csv',encoding='utf_8_sig')