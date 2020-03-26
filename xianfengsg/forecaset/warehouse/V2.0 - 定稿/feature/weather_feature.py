# -*- coding: utf-8 -*-
# @Time    : 2020/3/4 9:48
# @Author  : Ye Jinyu__jimmy
# @File    : weather_feature

import pandas as pd
import cx_Oracle
import pymysql
import os
import numpy as np
from random import randint
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
from pandas.tseries.offsets import Day,WeekOfMonth,DateOffset
#parser是根据字符串解析成datetime,字符串可以很随意，可以用时间日期的英文单词，
# 可以用横线、逗号、空格等做分隔符。没指定时间默认是0点，没指定日期默认是今天，没指定年份默认是今年。
from dateutil.parser import parse

# from pylab import *
plt.switch_backend('agg')
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,PolynomialFeatures
register_matplotlib_converters()


'''纯粹针对各城市进行天气特征的处理'''

#——————————————————————————————————读取天气数据————————————————————————————————
class get_weather_Data():
    #定义函数读取天气的信息
    def Mysql_Data(self,sql_name):
        parame_mysql = {'host': 'rm-bp109y7z8s1hj3j64.mysql.rds.aliyuncs.com',
                        'port': 3306,
                        'user': "root",
                        'password': "xfsg@8888",
                        'database': "xfsg_rpt"
                        }  ## parameters for database
        conn = pymysql.connect(**parame_mysql)
        conn.cursor()
        data_mysql = pd.read_sql(sql_name, conn)
        return data_mysql


    def read_weather(slef,city_name):
        weather_sql = """
        SELECT weather_date,general,temperature,wind_direction,wind_power FROM weather_detail wd 
        WHERE wd.city_name= '%s' AND wd.weather_date > DATE('2019-01-01')""" %(city_name)
        weather_sql_read = get_weather_Data.Mysql_Data(slef,weather_sql)
        return weather_sql_read


    #这是得到历史的天气信息
    def read_weather_his(slef):
        print('连接到mysql服务器，读取天气数据')
        db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                             database="purchare_sys", user="purchare_sys",
                             password="purchare_sys@123", port=3306, charset='utf8')

        weather_sql = """SELECT * FROM weather_history"""
        print('连接成功，天气数据读取成功')
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
        get_class  = get_weather_Data()
        read_data = get_class.read_weather_his()
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
        return all_weather


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
        gle = LabelEncoder()  #先转换成成数值型
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
        weather_feature = pd.concat([weather_ohe, wind_features], axis = 1)
        weather_feature = weather_feature.drop(['wind_direction','general'],axis=1)
        return weather_feature

    #对天气的特征进行时间的重命名，并且对时间日期进行格式转化
    def weather_date(self,data):
        weather_data = data.rename(index=str, columns={'weather_date': 'Account_date'})
        weather_data['Account_date'] = pd.to_datetime(weather_data['Account_date']).dt.normalize()
        return weather_data



#得到最终的天气特征
def get_weather_features(city_name):
    get_class_weather = get_weather_Data()
    weather_sql_read = get_class_weather.read_weather(city_name)
    get_class_cleaning = data_Cleaning()
    new_data = get_class_cleaning.merge_data(weather_sql_read)
    #--------------------------------------
    get_feature = feature_enginering()
    weather_wind = get_feature.Inter_feature(new_data)
    weather_feature = get_feature.feature_discrete(weather_wind)
    weather_feature = get_feature.weather_date(weather_feature)
    return weather_feature


if __name__ == '__main__':
    weather_sql_read = get_weather_features('杭州')
    print(weather_sql_read)
    # weather_sql_read.to_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/weather_sql_read.csv', encoding='utf_8_sig')







