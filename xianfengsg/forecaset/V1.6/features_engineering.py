# -*- coding: utf-8 -*-
# @Time    : 2019/7/23 9:51
# @Author  : Ye Jinyu__jimmy
# @File    : features_engineering.py

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
#如下是支持中文数字
# mpl.rcParams['font.sans-serif'] = ['SimHei']
#读取得到数据
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from tqdm import *
import itertools
import datetime
from datetime import datetime
import os
import copy
import sys
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import math
import warnings
# import chinese_calendar as calendar  #
import time
import pymysql
from random import randint
warnings.filterwarnings("ignore")
import get_holiday_inf_2019
import get_holiday_inf_2018

global parame_mysql


parame_mysql = { 'host':'rm-bp109y7z8s1hj3j64.mysql.rds.aliyuncs.com',
                 'port': 3306,
                 'user':"root",
                 'password':"xfsg@8888",
                 'database':"xfsg_rpt"
              } ## parameters for database

#定义函数读取天气的信息
def Mysql_Data(sql_name):
    conn = pymysql.connect(**parame_mysql)
    conn.cursor()
    data_mysql = pd.read_sql(sql_name, conn)
    return data_mysql


def read_weather():
    weather_sql = """
    SELECT weather_date,general,temperature,wind_direction,wind_power FROM weather_detail wd 
    WHERE wd.city_name='杭州' AND wd.weather_date > DATE('2019-01-01')"""
    weather_sql_read = Mysql_Data(weather_sql)
    return weather_sql_read


#这是得到历史的天气信息
def read_weather_his():
    print('连接到mysql服务器...')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123", port=3306, charset='utf8')
    print('连接成功')
    weather_sql = """SELECT * FROM weather_history"""
    db.cursor()
    read_data = pd.read_sql(weather_sql, db)
    db.close()
    return read_data


#------------------------------------------------先对得到的数据进行清洗
#先对温度进行处理，问题：温度不是结构化数据，存在有的数据是单独的数字，有的就是两个数据中间加/分开-----\℃|/|℃

class data_Cleaning():
    def cleaning_temperature(self,weather_sql_read):
        newDF = weather_sql_read['temperature'].str.split(pat = r'/', n = 2, expand = True)
        newDF.columns = ['max', 'min']
        # # print(newDF['min'].iloc[10])
        # # print(type(newDF['min'].iloc[10]))
        # # test = newDF[pd.isnull(newDF['min']) == False]
        def split_function(txt):
            if txt is None:
                return 1000
            else:
                result = txt.split('℃')
                return int(result[0])
        newDF['min'] = newDF['min'].apply(lambda x: split_function(x))
        newDF['max'] = newDF['max'].apply(lambda x: split_function(x))
        newDF['min_tem'] = pd.Series()

        newDF.loc[newDF['min'] >= 999,'min_tem'] = newDF['max']
        newDF.loc[newDF['min'] >= 999,'max'] = newDF['max']+randint(5,8)
        # newDF["min_tem"] = np.where(newDF["min"] > 999, newDF["min"],)
        newDF =newDF.fillna(method='ffill',axis=1)
        newDF = newDF.drop(columns=['min'],axis=1)

        weather_sql_read = weather_sql_read.drop(columns=['temperature'],axis=1)
        new_data = pd.concat([weather_sql_read,newDF],join="outer",axis=1)
        return new_data


    # print(new_data)
    def cleaning_wind_power(self,weather_sql_read):
        newDF = weather_sql_read['wind_power'].str.split(pat=r'-|<|级|级转|-|级', n=2, expand=True)
        newDF.columns = ['max', 'min_wind','none']
        newDF = newDF.drop(columns=['none'],axis=1)
        def split_function(txt):
            if len(txt) == 0:
                return int(1000)
            else:
                return int(txt)
        newDF['max'] = newDF['max'].apply(lambda x: split_function(x))
        newDF.loc[newDF['max'] >= 999,'max_wind'] = newDF['min_wind']
        newDF.insert(0, 'max_wind', newDF.pop('max_wind'))
        newDF = newDF.fillna(method='bfill',axis=1)
        newDF = newDF.drop(columns=['max'],axis=1)
        # newDF = newDF[newDF['max'].map(lambda x: len(str(x).strip())>0)].copy()
        weather_sql_read = weather_sql_read.drop(columns=['wind_power'], axis=1)
        new_data = pd.concat([weather_sql_read, newDF], join="outer", axis=1)
        return new_data

    def get_his(self):
        read_data = read_weather_his()
        weather_his_data = pd.DataFrame(
            columns=['weather_date', 'general',
                     'wind_direction', 'max', 'min_tem', 'max_wind', 'min_wind'])
        read_data['general'] = pd.Series(read_data.groupby("Date")["Weather_description"]
                                         .transform(lambda x: x.value_counts().index[0]))
        read_data['wind_direction'] = pd.Series(read_data.groupby("Date")["Wind_direction"]
                                                .transform(lambda x: x.value_counts().index[0]))
        weather_his_data["weather_date"] = read_data.groupby(["Date"], as_index=False).mean()['Date']
        weather_his_data["general"] = read_data.groupby(["Date"], as_index=False).max()['general']
        weather_his_data["wind_direction"] = read_data.groupby(["Date"], as_index=False).max()['wind_direction']
        weather_his_data["max"] = read_data.groupby(["Date"], as_index=False).max()['Temperature']
        weather_his_data["min_tem"] = read_data.groupby(["Date"], as_index=False).min()['Temperature']
        weather_his_data["max_wind"] = read_data.groupby(["Date"], as_index=False).max()['Wind_speed']
        weather_his_data["min_wind"] = read_data.groupby(["Date"], as_index=False).min()['Wind_speed']
        return weather_his_data

    def merge_data(self,weather_sql_read):
        new_data = data_Cleaning.cleaning_temperature(self,weather_sql_read)
        new_data = data_Cleaning.cleaning_wind_power(self,new_data)
        weather_his = data_Cleaning.get_his(self)
        all_weather = pd.concat([weather_his,new_data])
        all_weather = all_weather.reset_index(drop=True)
        print('all_weather',all_weather)
        return all_weather

# weather_sql_read = read_weather()
# get_class_cleaning = data_Cleaning()
# new_data = get_class_cleaning.merge_data(weather_sql_read)
# print(new_data)
# new_data.to_csv('D:/jimmy-ye/AI/AI_supply_chain/data/new_data.csv', encoding='utf_8_sig')
#----------------------------------------------------------------------以下对天气信息进行特征工程操作

class feature_enginering():
#对气温和风速特征进行的特征工程,数值型的特征构建特征工程
    def Inter_feature(self,new_data):
        pf = PolynomialFeatures(degree=2,interaction_only=False,include_bias=False)
        res = pf.fit_transform(new_data[['max','min_tem']])
        intr_tem_features = pd.DataFrame(res, columns=['max','min','max^2','min_x_max','min^2'])
        weather_tem = pd.concat([new_data, intr_tem_features], axis=1)

        res = pf.fit_transform(new_data[['min_wind','max_wind']])
        intr_wind_features = pd.DataFrame(res, columns=['max_wind','min_wind','max_wind^2',
                                                       'min_wind_x_max_wind','min_wind^2'])
        weather_wind = pd.concat([weather_tem, intr_wind_features], axis=1)
        #将原来的数据进行删除操作，可以选择性的执行与否
        weather_wind = weather_wind.drop(columns=['max','min_tem','max_wind','min_wind'], axis=1)
        return weather_wind



    def feature_discrete(self,weather_wind):
        #对离散的定性属性特征构建特征工程
        gle = LabelEncoder()
        genre_labels = gle.fit_transform(weather_wind['general'])
        # genre_mappings = {index: label for index, label in enumerate(gle.classes_)}
        weather_wind['general_Label'] = genre_labels

        #采取ohe的方式对分类数据进行编码
        # encode generation labels using one-hot encoding scheme
        gen_ohe = OneHotEncoder()
        gen_feature_arr = gen_ohe.fit_transform(weather_wind[['general_Label']]).toarray()
        gen_feature_labels = list(gle.classes_)
        gen_features = pd.DataFrame(gen_feature_arr, columns=gen_feature_labels)

        #最后将主数据和特征处理后的数据进行
        weather_ohe = pd.concat([weather_wind, gen_features], axis=1)

        #进行风向特征工程
        wind_direction_labels = gle.fit_transform(weather_ohe['wind_direction'])
        weather_ohe['wind_direction_Label'] = wind_direction_labels

        #对原始的中文数字采取更改名称的处理方式

        #采取ohe的方式对分类数据进行编码
        # encode generation labels using one-hot encoding scheme
        wind_dir_ohe = OneHotEncoder()
        wind_feature_arr = wind_dir_ohe.fit_transform(weather_ohe[['wind_direction_Label']]).toarray()
        wind_feature_labels = list(gle.classes_)
        wind_features = pd.DataFrame(wind_feature_arr, columns=wind_feature_labels)
        weather_feature = pd.concat([weather_ohe, wind_features], axis=1)
        weather_feature = weather_feature.drop(['wind_direction','general'],axis=1)
        return weather_feature

    #对天气的特征进行时间的重命名，并且对时间日期进行格式转化
    def weather_date(self,data):
        weather_data = data.rename(index=str, columns={'weather_date': 'Account_date'})
        weather_data['Account_date'] = pd.to_datetime(weather_data['Account_date']).dt.normalize()
        return weather_data



#得到最终的天气特征
def get_weather_features():
    weather_sql_read = read_weather()
    get_class_cleaning = data_Cleaning()
    new_data = get_class_cleaning.merge_data(weather_sql_read)
    #--------------------------------------
    get_feature = feature_enginering()
    weather_wind = get_feature.Inter_feature(new_data)
    weather_feature = get_feature.feature_discrete(weather_wind)
    weather_feature = get_feature.weather_date(weather_feature)
    return weather_feature


#================================================================================================
#以下是对节日信息进行特征工程
#------------------------------------------------------------------------------------------------
#先是对近两年的天气信息进行汇总
class get_holiday_reslut():
    def get_holiday(self):
        holiday_2019 = get_holiday_inf_2019.main()
        holiday_2019 = holiday_2019[1].reindex()
        holiday_2018 = get_holiday_inf_2018.main()
        holiday_2018 = holiday_2018[1].reindex()
        holiday_2019['Account_date'] = pd.to_datetime(holiday_2019['Account_date'], format='%Y-%m-%d', errors='ignore')
        holiday_2018['Account_date'] = pd.to_datetime(holiday_2018['Account_date'], format='%Y-%m-%d', errors='ignore')
        holiday = pd.concat([holiday_2018,holiday_2019],join='outer',axis=0)

        holiday = holiday.fillna(limit=2,method='bfill')
        holiday = holiday.fillna(limit=2,method='pad')
        holiday = holiday.fillna(0)
        return holiday

    #对节日信息进行处理
    def holiday_cleaning(self,holiday):
        for i in range(len(holiday)):
            if holiday['Chinese_festival'].iloc[i] == 0 :
                pass
            elif holiday['Chinese_festival'].iloc[i+1] != 0 and\
                holiday['Chinese_festival'].iloc[i+2] != 0 and\
                    holiday['Chinese_festival'].iloc[i+3] != 0 and\
                    holiday['Chinese_festival'].iloc[i+4] != 0:
                for k in range(0,5):
                    holiday['Chinese_festival'].iloc[i+k] = holiday['Chinese_festival'].iloc[i] + str(k)
            else:
                pass

            if holiday['Solar_festival'].iloc[i] == 0 :
                pass
            elif holiday['Solar_festival'].iloc[i+1] != 0 and\
                holiday['Solar_festival'].iloc[i+2] != 0 and\
                    holiday['Solar_festival'].iloc[i+3] != 0 and\
                    holiday['Solar_festival'].iloc[i+4] != 0:
                for k in range(0,5):
                    holiday['Solar_festival'].iloc[i+k] = holiday['Solar_festival'].iloc[i] + str(k)
            else:
                pass


            if holiday['Term_festival'].iloc[i] == 0 :
                pass
            elif holiday['Term_festival'].iloc[i+1] != 0 and\
                holiday['Term_festival'].iloc[i+2] != 0 and\
                    holiday['Term_festival'].iloc[i+3] != 0 and\
                    holiday['Term_festival'].iloc[i+4] != 0:
                for k in range(0,5):
                    holiday['Term_festival'].iloc[i+k] = holiday['Term_festival'].iloc[i] + str(k)
            else:
                pass


            if holiday['Lunar_festival'].iloc[i] == 0 :
                pass
            elif holiday['Lunar_festival'].iloc[i+1] != 0 and\
                holiday['Lunar_festival'].iloc[i+2] != 0 and\
                    holiday['Lunar_festival'].iloc[i+3] != 0  and\
                    holiday['Lunar_festival'].iloc[i+4] != 0:
                for k in range(0,5):
                    holiday['Lunar_festival'].iloc[i+k] = holiday['Lunar_festival'].iloc[i] + str(k)
            else:
                pass

        return holiday


#-----------------------------------------------------------------------------------------
#对节日信息进行特征工程
# class holiday_features():
#构建的节日的特征信息
def feature_discrete(holiday_original):
    #对离散的定性属性特征构建特征工程
    holiday = holiday_original.reset_index(drop=True)
    gle = LabelEncoder()
    Weekday_labels = gle.fit_transform(holiday['Weekday'])
    # genre_mappings = {index: label for index, label in enumerate(gle.classes_)}
    holiday['Weekday_Label'] = Weekday_labels

    #采取ohe的方式对分类数据进行编码
    holiday_ohe = OneHotEncoder()
    holiday_feature_arr = holiday_ohe.fit_transform(holiday[['Weekday_Label']]).toarray()
    holiday_feature_labels = list(gle.classes_)
    holiday_features = pd.DataFrame(holiday_feature_arr, columns=holiday_feature_labels)
    #最后将主数据和特征处理后的数据进行
    holiday_01 = pd.concat([holiday, holiday_features],axis=1)

    #-----------------------------------------------------------------------------
    #对Chinese_festival进行特征处理
    # holiday['Chinese_festival'] = holiday.to_string(columns = ['Chinese_festival'])
    holiday['Chinese_festival'] =holiday['Chinese_festival'].astype(str)
    gle = LabelEncoder()
    Weekday_labels = gle.fit_transform(holiday['Chinese_festival'])
    # genre_mappings = {index: label for index, label in enumerate(gle.classes_)}
    holiday['Chinese_festival_Label'] = Weekday_labels

    #采取ohe的方式对分类数据进行编码
    holiday_ohe = OneHotEncoder()
    holiday_feature_arr = holiday_ohe.fit_transform(holiday[['Chinese_festival_Label']]).toarray()
    holiday_feature_labels = list(gle.classes_)
    holiday_features = pd.DataFrame(holiday_feature_arr, columns=holiday_feature_labels)
    #最后将主数据和特征处理后的数据进行
    holiday_02 = pd.concat([holiday_01, holiday_features],axis=1)

    # #--------------------------------------------------------------------------------
    # #对Solar_festival进行特征处理
    holiday['Solar_festival'] =holiday['Solar_festival'].astype(str)
    gle = LabelEncoder()
    Weekday_labels = gle.fit_transform(holiday['Solar_festival'])
    # genre_mappings = {index: label for index, label in enumerate(gle.classes_)}
    holiday['Solar_festival_Label'] = Weekday_labels

    #采取ohe的方式对分类数据进行编码
    holiday_ohe = OneHotEncoder()
    holiday_feature_arr = holiday_ohe.fit_transform(holiday[['Solar_festival_Label']]).toarray()
    holiday_feature_labels = list(gle.classes_)
    holiday_features = pd.DataFrame(holiday_feature_arr, columns=holiday_feature_labels)
    #最后将主数据和特征处理后的数据进行
    holiday_03 = pd.concat([holiday_02, holiday_features],axis=1)
    # #----------------------------------------------------------------------
    # #对Term_festival进行特征处理
    holiday['Term_festival'] =holiday['Term_festival'].astype(str)
    gle = LabelEncoder()
    Weekday_labels = gle.fit_transform(holiday['Term_festival'])
    # genre_mappings = {index: label for index, label in enumerate(gle.classes_)}
    holiday['Term_festival_Label'] = Weekday_labels

    #采取ohe的方式对分类数据进行编码
    holiday_ohe = OneHotEncoder()
    holiday_feature_arr = holiday_ohe.fit_transform(holiday[['Term_festival_Label']]).toarray()
    holiday_feature_labels = list(gle.classes_)
    holiday_features = pd.DataFrame(holiday_feature_arr, columns=holiday_feature_labels)
    #最后将主数据和特征处理后的数据进行
    holiday_04 = pd.concat([holiday_03, holiday_features],axis=1)
    # #------------------------------------------------------------------------
    # #对Lunar_festival进行特征处理
    holiday['Lunar_festival'] =holiday['Lunar_festival'].astype(str)
    gle = LabelEncoder()
    Weekday_labels = gle.fit_transform(holiday['Lunar_festival'])
    # genre_mappings = {index: label for index, label in enumerate(gle.classes_)}
    holiday['Lunar_festival_Label'] = Weekday_labels

    #采取ohe的方式对分类数据进行编码
    holiday_ohe = OneHotEncoder()
    holiday_feature_arr = holiday_ohe.fit_transform(holiday[['Lunar_festival_Label']]).toarray()
    holiday_feature_labels = list(gle.classes_)
    holiday_features = pd.DataFrame(holiday_feature_arr, columns=holiday_feature_labels)
    #最后将主数据和特征处理后的数据进行
    holiday_05 = pd.concat([holiday_04, holiday_features],axis=1)
    holiday_05 = holiday_05.drop([ '0','Weekday','Chinese_festival', 'Solar_festival', 'Term_festival', 'Lunar_festival'],axis=1)
    print(holiday_05)
    return holiday_05


#============================================================================================================
#以下开始构建纯时间的特征
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

def made_feature():
    get_holiday = get_holiday_reslut()
    original_holiday = get_holiday.get_holiday()
    # result_holiday = get_holiday.holiday_cleaning(original_holiday)
    holiday_features = feature_discrete(original_holiday)
    weather_result = get_weather_features()
    merge_data_01 = pd.merge(weather_result,holiday_features,on=['Account_date'],how='inner')
    get_time_series = time_series_feature()
    get_time_feature = get_time_series.time_subset(merge_data_01)
    return get_time_feature


if __name__ == '__main__':
    get_time_feature = made_feature()
    get_time_feature = get_time_feature.reset_index(drop=True)
    print(get_time_feature)

    # get_time_feature.to_csv('D:/AI/xianfengsg/forecaset/V1.4/feature.csv', encoding = 'utf_8_sig',index=False)

# weather_feature.to_csv('D:\jimmy-ye\AI_supply_chain\data\weather_feature.csv',encoding='utf_8_sig')
