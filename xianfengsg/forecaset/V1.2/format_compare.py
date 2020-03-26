# -*- coding: utf-8 -*-
# @Time    : 2019/7/30 9:14
# @Author  : Ye Jinyu__jimmy
# @File    : format_compare.py
import pandas as pd
import cx_Oracle
import os
import numpy as np
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 50)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
# 注：设置环境编码方式，可解决读取数据库乱码问题
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
from matplotlib import pyplot as plt
import multiprocessing
import re
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdate
#parser是根据字符串解析成datetime,字符串可以很随意，可以用时间日期的英文单词，
# 可以用横线、逗号、空格等做分隔符。没指定时间默认是0点，没指定日期默认是今天，没指定年份默认是今年。
from dateutil.parser import parse
# from pylab import *
plt.switch_backend('agg')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
#如下是支持中文数字
# mpl.rcParams['font.sans-serif'] = ['SimHei']
#读取得到数据
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from tqdm import *
import itertools
import datetime
import os
import copy
import sys
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import math
import warnings
from chinese_calendar import is_workday, is_holiday
# import chinese_calendar as calendar  #
import time
warnings.filterwarnings("ignore")
#该脚本是为了防止在画图程序画图的过程中有的只有年份但是没有具体的日期而做的操作




#读取预测数据这里是csv的文件
def read_forecast_data(old_path):
    data_old = pd.read_csv(old_path,encoding='utf_8_sig',low_memory=False)
    return data_old


#--------------------------------------------------------------------------------->日的标准化转化
def date_normalize(data_frame):
    data_frame_sort = data_frame.sort_values(by = ['Account_date'],ascending=False )
    data_frame_sort['Account_date'] = pd.to_datetime(data_frame_sort['Account_date']).dt.normalize()
    return data_frame_sort


#------------------------------------------------------------------------->设置函数按照预测的数据,并完成画图
def get_real_data(forecast_data):

    DC_list = set(forecast_data['Dc_code'])
    start_date = forecast_data['Account_date'].min()
    end_date = forecast_data['Account_date'].max()
    print('读取的截止日期和开始日期是：%s,%s，日期格式是%s'%(start_date,end_date,type(end_date)))
    for DC in tqdm(DC_list):
        sku_data = forecast_data[forecast_data['Dc_code'] == DC]
        sku_id_list = set(sku_data['Sku_id'])
        for x in tqdm(sku_id_list):
            mid_forecast = forecast_data[forecast_data['Dc_code'] == DC]
            mid_forecast = mid_forecast[mid_forecast['Sku_id'] == x]
            mid_forecast = date_normalize(mid_forecast)
            plot_compare(mid_forecast,DC,x)




#----------------------------------------->查看和比较真实和预测数据的真实误差，并查看误差的统计学指标，保存到csv
def description_error(data):
    data['abs_error'] = data['Forecast_qty'] -data['QTY']
    data['relative_error'] = data['abs_error']/data['QTY']
    return data


#---------------------------------------------------------------------->对每个SKU进行真实和预测对比
def plot_compare(data,DC,id):
    print('正在画图并记录的仓库和sku是:'+str(int(DC)),str(int(id)))
    sku_name = data['Sku_name'].iloc[0]
    dc_name = data['Dc_name'].iloc[0]
    if data.empty==True:            #-------------------确保程序运行，有可能有的DC没有SKU的预测信息
        pass
    else:
        total_error_describe = data.describe()
        print('total_error_describe',total_error_describe)
        total_error_describe.to_csv('D:/jimmy-ye/AI_supply_chain/data/'
                                    'forecast_holiday/total_error_describe'+str(int(DC))+str(dc_name)+
                                    '_'+str(int(id))+str(sku_name)+'.csv',
                           encoding="utf_8_sig")
        date = data['Account_date']
        forecast_qty = data['Forecast_qty']
        real_qty = data['QTY']
        fig = plt.figure(figsize=(20,10),facecolor='white')
        ax1 = fig.add_subplot(111)
        # 左轴
        ax1.bar(date, real_qty, width=0.5, align='center', label='real_qty', color="black")
        plt.legend(loc='upper left', fontsize=10)
        ax1.plot(date, forecast_qty, color='red', marker='o', linestyle='dashed', label='forecast_qty',
                 markersize=0.8)
        plt.legend(loc='upper right', fontsize=10)
        ax1.set_xlabel('date')
        ax1.set_ylabel('real_qty')

        plt.savefig("D:/jimmy-ye/AI_supply_chain/data/forecast_holiday/compare" +
                    str(int(DC)) + str(dc_name) +
                    '_' + str(int(id)) + str(sku_name)+ '.jpg', dpi=600,
                    bbox_inches='tight')
        plt.close()





def main_function(old_path):
    data = read_forecast_data(old_path)
    get_real_data(data)




main_function('D:/jimmy-ye/AI_supply_chain/data/forecast_holiday/final_data_new_parameters.csv')
