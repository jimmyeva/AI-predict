# -*- coding: utf-8 -*-
# @Time    : 2019/12/10 14:38
# @Author  : Ye Jinyu__jimmy
# @File    : process_data.py
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
# import get_holiday_inf_2019
# import get_holiday_inf_2018
# import get_holiday_inf_2020
import psycopg2
import datetime



'''该脚本是作为算法的前期数据预处理的操作步骤'''
def read_postgresql():
    conn = psycopg2.connect(database="dc_rpt", user="ads", password="ads@xfsg2019", host="192.168.1.205", port="3433")
    print("Opened database successfully")

    cur = conn.cursor()
    cur.execute("SELECT store_code,goods_code,goods_name,start_dt from "
                "ads_trd_str.ads_newstore_purchase_amount_more_d LIMIT 100 ")
    rows = cur.fetchall()
    print("Operation done successfully")
    conn.close()
    return rows

#-----------------------------------------------------------先选择单独一个门店的销售数据作为预测测试门店进行预测
#----------读取销售数据
def get_sales():
    conn = psycopg2.connect(database="dc_rpt", user="ads", password="ads@xfsg2019", host="192.168.1.205", port="3433")
    print("store门店销售数据读取开始")
    sales_sql = ("SELECT store_code,goods_code,goods_name,start_dt,sales"
                " from ads_trd_str.ads_newstore_purchase_amount_more_d"
                " WHERE store_code ='33010037' AND start_dt >'20170101' ")
    try:
        data_sales = pd.read_sql(sales_sql, conn)
    except:
        print("销售数据读取失败 !")
        data_sales = pd.DataFrame()
        exit()
    conn.close()
    data_sales['start_dt'] = pd.to_datetime(data_sales['start_dt'])
    data_sales = data_sales.rename(index=str, columns={'start_dt': 'Account_date'})
    print("销售数据读取完成！")
    return data_sales


#========================================================================天气数据
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


def read_weather(name,region):
    weather_sql = """
    SELECT weather_date,general,temperature,wind_direction,wind_power FROM weather_detail wd 
    WHERE wd.city_name='%s' AND wd.weather_date >= DATE('2019-01-01')"""%(name)
    weather_sql_read = Mysql_Data(weather_sql)
    if weather_sql_read.empty == True:
        weather_sql = """SELECT weather_date,general,temperature,wind_direction,wind_power FROM weather_detail wd 
         WHERE wd.city_name='%s' AND wd.weather_date >= DATE('2019-01-01')""" % (region)
        weather_sql_read = Mysql_Data(weather_sql)
    else:
        pass
    return weather_sql_read


#这是得到历史的天气信息
def read_weather_his():
    print('连接到mysql服务器...')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123", port=3306, charset='utf8')
    print('连接成功,正在读取历史天气数据')
    weather_sql = """SELECT * FROM weather_history"""
    db.cursor()
    read_data = pd.read_sql(weather_sql, db)
    read_data['Wind_speed'] = read_data['Wind_speed'].astype(int)
    db.close()
    return read_data



#------------读取天气数据
class data_Cleaning():
    def cleaning_temperature(self,weather_sql_read):
        newDF = weather_sql_read['temperature'].str.split(pat = r'/', n = 2, expand = True)
        newDF.columns = ['max', 'min']

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
        newDF = newDF.fillna(method='ffill', axis=1)
        newDF = newDF.drop(columns=['max'],axis=1)
        # newDF = newDF[newDF['max'].map(lambda x: len(str(x).strip())>0)].copy()
        weather_sql_read = weather_sql_read.drop(columns=['wind_power'], axis=1)
        new_data = pd.concat([weather_sql_read, newDF], join="outer", axis=1)
        new_data['max_wind'] = new_data['max_wind'].astype(int)
        new_data['min_wind'] = new_data['min_wind'].astype(int)
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
        return all_weather

#--------------------------设置主函数用于读取天气数据
def get_weather(name,region):
    weather_sql_read = read_weather(name,region)
    get_class_cleaning = data_Cleaning()
    new_data = get_class_cleaning.merge_data(weather_sql_read)
    new_data['weather_date'] = pd.to_datetime(new_data['weather_date'])
    new_data = new_data.rename(index=str, columns={'weather_date': 'Account_date'})
    return new_data


#-------------读取节假日信息
# def get_holiday():
#     holiday_2019 = get_holiday_inf_2019.main()
#     holiday_2019 = holiday_2019[1].reindex()
#     holiday_2018 = get_holiday_inf_2018.main()
#     holiday_2018 = holiday_2018[1].reindex()
#     holiday_2020 = get_holiday_inf_2020.main()
#     holiday_2020 = holiday_2020[1].reindex()
#     holiday_2019['Account_date'] = pd.to_datetime(holiday_2019['Account_date'], format='%Y-%m-%d', errors='ignore')
#     holiday_2018['Account_date'] = pd.to_datetime(holiday_2018['Account_date'], format='%Y-%m-%d', errors='ignore')
#     holiday_2020['Account_date'] = pd.to_datetime(holiday_2020['Account_date'], format='%Y-%m-%d', errors='ignore')
#     holiday = pd.concat([holiday_2018,holiday_2019,holiday_2020],join='outer',axis=0)
#     print(holiday_2020)
#     # holiday = holiday.fillna(limit=2,method='bfill')
#     # holiday = holiday.fillna(limit=2,method='pad')
#     holiday = holiday.fillna('empty')
#     return holiday




def main(end_date,name,region):
    weather_data = get_weather(name,region)
    sales_data = get_sales()
    # holiday_data = get_holiday()
    sales_data.to_csv('./sales_data.csv', encoding='utf_8_sig')
    #选择inner的方式是为了防止有的门店在因为某些原因在特定的日期就是没有销售行为
    feature_data = weather_data
    training_data = pd.merge(feature_data,sales_data,on='Account_date',how='inner')
    #用于预测的特征数据
    predict_data = feature_data[feature_data['Account_date'] >= end_date]
    return training_data,predict_data




if __name__ == '__main__':
    today = datetime.date.today()
    end_date = today.strftime('%Y%m%d')
    name ='杭州'
    region = '杭州'
    training_data,predict_data = main(end_date,name,region)
    print(predict_data)
    training_data.to_csv('./training_data.csv', encoding='utf_8_sig')














