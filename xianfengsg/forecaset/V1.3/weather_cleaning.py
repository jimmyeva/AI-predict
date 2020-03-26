# -*- coding: utf-8 -*-
# @Time    : 2019/8/19 15:32
# @Author  : Ye Jinyu__jimmy
# @File    : weather_cleaning.py
import pandas as pd
import pymysql
from datetime import datetime,date
from datetime import timedelta
import sys
from sklearn.preprocessing import Binarizer,PolynomialFeatures
import string
import numpy as np
from random import randint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from collections import Counter


# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)


## parameters of Ess_data from database for mysql
global parame_mysql

parame_mysql = { 'host':"192.168.1.24",
                 'port': 3306,
                 'user':"xfsgdata",
                 'password':"xfsgdata@123",
                 'database':"xfsg_rpt"
              } ## parameters for database

#定义函数读取天气的信息
def Mysql_Data(sql_name):
    conn = pymysql.connect(**parame_mysql)
    conn.cursor()
    data_mysql = pd.read_sql(sql_name, conn)
    return data_mysql


def read_weather():
    weather_sql = """SELECT * FROM weather_detail wd WHERE wd.city_name='杭州'"""
    weather_sql_read = Mysql_Data(weather_sql)
    return weather_sql_read


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

    def merge_data(self,weather_sql_read):
        new_data = data_Cleaning.cleaning_temperature(self,weather_sql_read)
        new_data = data_Cleaning.cleaning_wind_power(self,new_data)
        return new_data


#需要将第三方的历史天气数据进行清洗和处理

def merge_his():
    read_data = pd.read_csv('D:/jimmy-ye/AI/AI_supply_chain/data/CN101210101-小时.csv',encoding='utf_8_sig')
    read_data['日期'] = pd.to_datetime(read_data['日期'], format='%Y-%m-%d', errors='ignore')


    weather_his_data = pd.DataFrame(
        columns=['weather_date','general',
                 'wind_direction','max','min_tem','max_wind','min_wind'])
    read_data['general'] = pd.Series(read_data.groupby("日期")["天气状况名称"]
                             .transform(lambda x: x.value_counts().index[0]))
    read_data['wind_direction'] = pd.Series(read_data.groupby("日期")["风向"]
                             .transform(lambda x: x.value_counts().index[0]))
    print(read_data)
    print([ column for column in read_data])
    weather_his_data["weather_date"] = read_data.groupby(["日期"], as_index=False).mean()['日期']
    weather_his_data["general"] = read_data.groupby(["日期"], as_index=False).max()['general']
    weather_his_data["wind_direction"] = read_data.groupby(["日期"], as_index=False).max()['wind_direction']
    weather_his_data["max"] = read_data.groupby(["日期"], as_index=False).max()['温度']
    weather_his_data["min_tem"] = read_data.groupby(["日期"], as_index=False).min()['温度']
    weather_his_data["max_wind"] = read_data.groupby(["日期"], as_index=False).max()['风速']
    weather_his_data["min_wind"] = read_data.groupby(["日期"], as_index=False).min()['风速']

    # read_data["general"] = (read_data.groupby("日期")["天气状况名称"]
    #                          .transform(lambda x: Counter(x).most_common(1)))
    # read_data["general"] = (read_data.groupby("日期")["天气状况名称"]
    #                          .transform(lambda x: x.value_counts().index[0]))

    # weather_his_data["general"] = read_data.groupby(["日期"], as_index=False).mean()['日期']
    print(weather_his_data)
    return weather_his_data

# #
# weather_sql_read = read_weather()
# get_class_cleaning = data_Cleaning()
# new_data = get_class_cleaning.merge_data(weather_sql_read)
#
# new_data.to_csv('D:/jimmy-ye/AI/AI_supply_chain/data/new_data.csv',encoding='utf_8_sig')



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
        weather_feature = weather_feature.drop(['wind_direction','general','city_name','id','city_code','upd_time'],axis=1)
        return weather_feature

# get_feature = feature_enginering()
# weather_wind = get_feature.Inter_feature(new_data)
# weather_feature = get_feature.feature_discrete(weather_wind)
# print(weather_feature)

# weather_feature.to_csv('D:\jimmy-ye\AI_supply_chain\data\weather_feature.csv',encoding='utf_8_sig')
#
# # columns = sum([['Name', 'Generation', 'Gen_Label'], gen_feature_labels], [])
#
# print(weather_ohe[columns].iloc[4:10])









