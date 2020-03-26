# -*- coding: utf-8 -*-
# @Time    : 2020/3/2 17:14
# @Author  : Ye Jinyu__jimmy
# @File    : time_feature

import pandas as pd
import cx_Oracle
import os
import numpy as np
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,PolynomialFeatures
register_matplotlib_converters()

'''当前包是为了能够找出尽可能多的时间序列方面的特征，包括日期的基本特征，滑窗特征，'''


#——————————————————纯时间序列的特征————————————————————————————————
class time_series_feature():
    def period_of_month(self,day):
        if day in range(1, 11): return 1
        if day in range(11, 21): return 2
        if day in range(21, 32): return 3

    def period2_of_month(self,day):
        if day in range(1, 16): return 1
        if day in range(16, 32): return 2

    def week_of_month(self,day):
        if day in range(1, 8): return 1
        if day in range(8, 15): return 2
        if day in range(15, 22): return 3
        if day in range(22, 32): return 4

    def quarter(self,month):
        if month in range(1, 4): return 1
        if month in range(4, 7): return 2
        if month in range(7, 10): return 3
        if month in range(10, 13): return 4


    def time_subset(self,x):
        x['Account_date']       = pd.to_datetime(x['Account_date'])
        x["dayofweek"]          = x['Account_date'].apply(lambda x: x.dayofweek)
        x["weekofyear"]         = x["Account_date"].apply(lambda x: x.weekofyear)
        x['month']              = x['Account_date'].apply(lambda x: x.month)
        x['day']                = x['Account_date'].apply(lambda x: x.day)
        x['year']               = x['Account_date'].apply(lambda x: x.year)
        x['period_of_month']    = x['day'].apply(lambda x: time_series_feature.period_of_month(self,x))
        x['period2_of_month']   = x['day'].apply(lambda x: time_series_feature.period2_of_month(self,x))
        x['week_of_month']      = x['day'].apply(lambda x: time_series_feature.week_of_month(self,x))
        x['quarter']            = x['month'].apply(lambda x: time_series_feature.quarter(self,x))
        return x


class time_process():

    #——————————————————————————结合鲜丰订货提前一天的特性做出特殊时间特征————————————————————
    def xfsg_time_feature(self,data):
        data['lag_holiday'] = data['holiday']
        for i in range(len(data)-1):
            if data['lag_holiday'].iloc[i] == 'weekday' and data['lag_holiday'].iloc[i+1] == 'holiday':
                data['lag_holiday'].iloc[i] = 'holiday_lag'
            else:
                pass
        del data['holiday']
        return data


    #——————————————————————————再加入假期的时长作为一个时间特征————————————————————————————
    def xfsg_count_days(self,data):
        data['holiday_constant'] = 0
        list_lag_holiday = data['lag_holiday'].to_list()
        list_holiday_constant = data['holiday_constant'].to_list()
        for i in range(len(data)-2):
            if list_lag_holiday[i] == 'holiday_lag':
                num = 0
                k = 0
                while list_lag_holiday[i+k] != 'weekday':
                    num += 1
                    k += 1
                list_holiday_constant[i] = num
                for x in range(num):
                    list_holiday_constant[i+x] = num
            else:
                pass
        data['holiday_constant'] = pd.Series(list_holiday_constant)
        return data

    #————————————————————————————时间序列特征主函数————————————————————
    def time_series_function(self,data_date):
        data = time_process.xfsg_time_feature(self,data_date)
        new_df = time_process.xfsg_count_days(self,data)
        return new_df




    #——————————————————————————————对weekday和lag_holiday进行哑编码处理——————————
    def one_hot_encoding(self,data):
        holiday = data.reset_index(drop=True)
        gle = LabelEncoder()
        Weekday_labels = gle.fit_transform(holiday['week_day'])
        # genre_mappings = {index: label for index, label in enumerate(gle.classes_)}
        holiday['week_day_Label'] = Weekday_labels

        #采取ohe的方式对分类数据进行编码
        holiday_ohe = OneHotEncoder()
        holiday_feature_arr = holiday_ohe.fit_transform(holiday[['week_day_Label']]).toarray()
        holiday_feature_labels = list(gle.classes_)
        holiday_features = pd.DataFrame(holiday_feature_arr, columns=holiday_feature_labels)
        #最后将主数据和特征处理后的数据进行
        holiday_01 = pd.concat([holiday, holiday_features],axis=1)
        holiday_01 = holiday_01.drop(['week_day','week_day_Label'],axis=1)

        #-----------------------------------------------------------------------------
        #对Chinese_festival进行特征处理
        holiday_01['lag_holiday'] =holiday_01['lag_holiday'].astype(str)
        gle = LabelEncoder()
        lag_holiday_labels = gle.fit_transform(holiday_01['lag_holiday'])
        holiday_01 = holiday_01.drop(['lag_holiday'], axis=1)
        # genre_mappings = {index: label for index, label in enumerate(gle.classes_)}
        holiday_01['lag_holiday_Label'] = lag_holiday_labels
        # holiday = holiday.drop(['week_day', 'Account_date','lag_holiday'], axis=1)
        # holiday_02 = pd.concat([holiday_01, holiday], axis=1)

        return holiday_01

    #————————————————————————————主函数————————————————
    def main(self,data_date):
        new_df = time_process.time_series_function(self,data_date)
        get_time_series = time_series_feature()
        feature = get_time_series.time_subset(new_df)
        feature_df = time_process.one_hot_encoding(self,feature)

        return feature_df


if __name__ == '__main__':
    data_date = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/data_date.csv',encoding='utf_8_sig')
    time_class= time_process()
    time_df = time_class.main(data_date)
    time_df.to_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data//time_df.csv', encoding='utf_8_sig')



