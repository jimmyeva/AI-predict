# -*- coding: utf-8 -*-
# @Time    : 2019/12/10 10:16
# @Author  : Ye Jinyu__jimmy
# @File    : test.py
import pandas as pd
from datetime import datetime
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import concatenate
from math import sqrt
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
import os
# 注：设置环境编码方式，可解决读取数据库乱码问题
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
import numpy as np
import psycopg2
#
# conn = psycopg2.connect(database="dc_rpt", user="ads", password="ads@xfsg2019", host="192.168.1.205", port="3433")
# print("Opened database successfully")
#
# cur = conn.cursor()
# cur.execute("SELECT store_code,goods_code,goods_name,start_dt from "
#             "ads_trd_str.ads_newstore_purchase_amount_more_d LIMIT 100 ")
# rows = cur.fetchall()
# print(rows)
# print("Operation done successfully")
# conn.close()

df   = pd.read_excel('store_num_test.xlsx',encoding='utf_8_sig')     #读入股票数据
data = df.iloc[:,1:6].values  #取第2-5列


# def get_test_data(time_step=5,test_begin=660):
#     data_test=data[test_begin:]
#     print(len(data_test))
#     size = (len(data_test) + time_step-1)//time_step  #有size个sample
#     print(size)
#     test_x = []
#     for i in range(size-1):
#         #x是获取前3-9列的特征，y取的是是10列的label
#         x = data_test[i*time_step:(i+1)*time_step,1:]
#         test_x.append(x.tolist())
#         test_x.append((data_test[(i+1)*time_step:,1:]).tolist())
#     return test_x

def get_test_data(time_step=20,test_begin=660):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data = (data_test-mean)/std  #标准化
    print(normalized_test_data)
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample
    test_x= []
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,1:]
       test_x.append(x.tolist())
       test_x.append((normalized_test_data[(i+1)*time_step:,1:]).tolist())
    return mean,std,test_x

mean,std,test_x = get_test_data(20)

# print('mean','\n',mean)
# print('std','\n',std)
# print('test_x','\n',test_x)
# print(len(test_x))
# print(len(test_x[0]))
# print(len(test_x[1]))
# print('test_y','\n',test_y)
# print(len(test_y))

