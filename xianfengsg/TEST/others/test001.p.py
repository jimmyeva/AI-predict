# -*- coding: utf-8 -*-
# @Time    : 2019/8/31 22:10
# @Author  : Ye Jinyu__jimmy
# @File    : test001.p

import pandas as pd
import cx_Oracle
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

import os
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
# 注：设置环境编码方式，可解决读取数据库乱码问题
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
import math
import time
import multiprocessing
from dateutil.parser import parse


# data = pd.read_csv('C:/Users/dell/Desktop/total_data.csv', encoding='utf_8_sig')
# data=data[['名称','ADDRESS','CITY','PROVINCE','COUNTY','SUM(BS.SCRTOTAL)','SUM(BS.REALAMT)','SUM(BS2.IAMT)',
#            'SUM(BS2.ITAX)','COUNT(BS.FLOWNO)','time']]
# print(data)
# data.drop_duplicates(inplace=True)
# print(data)
data=pd.read_csv('C:/Users/dell/Desktop/sales_store.csv',encoding='utf_8_sig')
data_01 = data[data['PROVINCE'] == '浙江省']
data_01 = data_01.groupby(['名称','ADDRESS','CITY','PROVINCE','COUNTY','time'], as_index=False).agg(sum)
print(data_01)
