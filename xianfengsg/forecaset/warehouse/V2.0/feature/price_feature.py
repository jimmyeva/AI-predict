# -*- coding: utf-8 -*-
# @Time    : 2020/3/4 8:47
# @Author  : Ye Jinyu__jimmy
# @File    : price_feature
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
from pandas.tseries.offsets import Day,WeekOfMonth,DateOffset
#parser是根据字符串解析成datetime,字符串可以很随意，可以用时间日期的英文单词，
# 可以用横线、逗号、空格等做分隔符。没指定时间默认是0点，没指定日期默认是今天，没指定年份默认是今年。
from dateutil.parser import parse
import psycopg2
# from pylab import *
plt.switch_backend('agg')
from pandas.plotting import register_matplotlib_converters

import copy
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,PolynomialFeatures
register_matplotlib_converters()





class price_process():
    '''将商品的价格进行清洗和特征工程的构建'''
    #————————————————————————————————————先获取商品的价格信息————————————————————————
    def get_price(self,time,code,city_name):
        conn = psycopg2.connect(database="proj_xfsg", user="LTAI4FtjhaC7HMVB5K9isR7D", password="C58h1Cs9hb8OsBLeD7zxY1Xo9Pr5vO",
                                host="lightning.cn-hangzhou.maxcompute.aliyun.com",
                                port="443")
        print("Opened database successfully,connected with PG DB-ALI")
        price_sql = """SELECT ds,avg(inventory_price) in_price,avg(mbr_price) mbr_price, avg(spec) unit 
        from dwd_xf_pur_str_sty_catalog_di1 
        where sty_code = '%s' and wrh_name like '%s%%' and ds <= '%s' GROUP BY ds ORDER BY ds ASC"""%(code,city_name,time)
        try:
            price = pd.read_sql(price_sql, conn)
            price.columns = ['Account_date', 'in_price', 'mbr_price', 'unit']
        except:
            print("load data from postgres failure !")
            price = pd.DataFrame()
            pass
        conn.close()
        return price


    #---------------------------价格数据的清洗和合并-----------------------
    def data_cleaning_price(self,end_time,sales,code,city_name):
        price = price_process.get_price(self,end_time,code,city_name)
        print('sales',sales)
        print('price',price)
        if price.empty == True:
            print('价格序列为空')
            new_df = pd.DataFrame()
        else:
            price['Account_date'] = pd.to_datetime(price['Account_date'])
            sales['Account_date'] = pd.to_datetime(sales['Account_date'])
            new_df = pd.merge(sales,price,on='Account_date',how='left').fillna(method = 'bfill').fillna(method='pad')
            new_df['price_unit']  = new_df['mbr_price']/new_df['unit']
            new_df['in_price_unit'] = new_df['in_price']/new_df['unit']
            new_df = new_df.drop(['in_price','mbr_price','unit'],axis =1)
        print('new_df',new_df)
        return new_df


    #————————————————————————————特征工程————————————————————————————————
    #————————————————————————————前一天，前三天，前7天——————————————————————
    def days_feature_price_unit(self,data):
        start_date = data['Account_date'].min()
        end_date = data['Account_date'].max()
        ts = pd.Series(data['price_unit'].values, index=pd.date_range(start_date, end_date, freq='D'))

        new_array_1day = (ts - ts.shift(1,'D'))
        df_1day = pd.DataFrame(new_array_1day)
        df_1day.reset_index(level=0, inplace=True)
        df_1day.columns = ['Account_date', 'back_1day_price_unit']

        new_array_3day = (ts - ts.shift(3,'D'))
        df_3day = pd.DataFrame(new_array_3day)
        df_3day.reset_index(level=0, inplace=True)
        df_3day.columns = ['Account_date', 'back_3day_price_unit']
        total_3day = pd.merge(df_1day,df_3day,on='Account_date',how='outer')

        new_array_7day = (ts - ts.shift(7,'D'))
        df_7day = pd.DataFrame(new_array_7day)
        df_7day.reset_index(level=0, inplace=True)
        df_7day.columns = ['Account_date', 'back_7day_price_unit']
        total_7day = pd.merge(total_3day,df_7day,on='Account_date',how='outer')
        data['Account_date'] = pd.to_datetime(data['Account_date'])
        total_7day['Account_date'] = pd.to_datetime(total_7day['Account_date'])

        total_7day = pd.merge(data,total_7day,on='Account_date',how='outer')
        total_7day= total_7day.fillna(method='pad',limit=7)
        return total_7day

    def days_feature_in_price_unit(self,data):
        start_date = data['Account_date'].min()
        end_date = data['Account_date'].max()
        ts = pd.Series(data['in_price_unit'].values, index=pd.date_range(start_date, end_date, freq='D'))

        new_array_1day = (ts - ts.shift(1,'D'))
        df_1day = pd.DataFrame(new_array_1day)
        df_1day.reset_index(level=0, inplace=True)
        df_1day.columns = ['Account_date', 'back_1day_in_price_unit']

        new_array_3day = (ts - ts.shift(3,'D'))
        df_3day = pd.DataFrame(new_array_3day)
        df_3day.reset_index(level=0, inplace=True)
        df_3day.columns = ['Account_date', 'back_3day_in_price_unit']
        total_3day = pd.merge(df_1day,df_3day,on='Account_date',how='outer')

        new_array_7day = (ts - ts.shift(7,'D'))
        df_7day = pd.DataFrame(new_array_7day)
        df_7day.reset_index(level=0, inplace=True)
        df_7day.columns = ['Account_date', 'back_7day_in_price_unit']
        total_7day = pd.merge(total_3day,df_7day,on='Account_date',how='outer')

        data['Account_date'] = pd.to_datetime(data['Account_date'])
        total_7day['Account_date'] = pd.to_datetime(total_7day['Account_date'])

        total_7day = pd.merge(data,total_7day,on='Account_date',how='outer')

        return total_7day

    #——————————————————————————————前7日销量的统计特征与前14天的特征和前21天的特征的变化——————————————————
    def pu_days_feature(self,df):
        data = copy.deepcopy(df)
        data['7d_p_mean'] = (data['price_unit'].shift(1) + data['price_unit'].shift(2) + data['price_unit'].shift(3) +
                                   data['price_unit'].shift(4) + data['price_unit'].shift(5) + data['price_unit'].shift(6) +
                                   data['price_unit'].shift(7)) / 7
        data['7d_p_var_'] = (data['price_unit'].shift(1) - data['7d_p_mean']) ** 2
        data['7d_p_trend'] = data['price_unit'].shift(1)/(data['7d_p_mean']+1)
        data['7d_p_cv'] = ((data['7d_p_var_'] ** 0.5) + 1) / (data['7d_p_mean'] + 7)

        data['14d_p_mean'] = (data['price_unit'].shift(8) + data['price_unit'].shift(9) + data['price_unit'].shift(10) +
                                   data['price_unit'].shift(11) + data['price_unit'].shift(12) + data['price_unit'].shift(13) +
                                   data['price_unit'].shift(14)) / 7
        data['14d_p_var_'] = (data['price_unit'].shift(1) - data['14d_p_mean']) ** 2
        data['14d_p_trend'] = data['price_unit'].shift(1)/(data['14d_p_mean']+1)
        data['14d_p_cv'] = ((data['14d_p_var_'] ** 0.5) + 1) / (data['14d_p_mean'] + 7)

        data['21d_p_mean'] = (data['price_unit'].shift(15) + data['price_unit'].shift(16) + data['price_unit'].shift(17) +
                                   data['price_unit'].shift(18) + data['price_unit'].shift(19) + data['price_unit'].shift(20) +
                                   data['price_unit'].shift(21)) / 7
        data['21d_p_var_'] = (data['price_unit'].shift(1) - data['21d_p_mean']) ** 2
        data['21d_p_trend'] = data['price_unit'].shift(1)/(data['21d_p_mean']+1)
        data['21d_p_cv'] = ((data['21d_p_var_'] ** 0.5) + 1) / (data['21d_p_mean'] + 7)

        return data

    def ipu_days_feature(self,df):
        data = copy.deepcopy(df)
        data['7d_ipu_mean'] = (data['in_price_unit'].shift(1) + data['in_price_unit'].shift(2) + data['in_price_unit'].shift(3) +
                                   data['in_price_unit'].shift(4) + data['in_price_unit'].shift(5) + data['in_price_unit'].shift(6) +
                                   data['in_price_unit'].shift(7)) / 7

        data['7d_ipu_var_'] = (data['in_price_unit'].shift(1) - data['7d_ipu_mean']) ** 2
        data['7d_ipu_trend'] = data['in_price_unit'].shift(1)/(data['7d_ipu_mean']+1)
        data['7d_ipu_cv'] = ((data['7d_ipu_var_'] ** 0.5) + 1) / (data['7d_ipu_mean'] + 7)

        data['14d_ipu_mean'] = (data['in_price_unit'].shift(8) + data['in_price_unit'].shift(9) + data['in_price_unit'].shift(10) +
                                   data['in_price_unit'].shift(11) + data['in_price_unit'].shift(12) + data['in_price_unit'].shift(13) +
                                   data['in_price_unit'].shift(14)) / 7
        data['14d_ipu_var_'] = (data['in_price_unit'].shift(1) - data['14d_ipu_mean']) ** 2
        data['14d_ipu_trend'] = data['in_price_unit'].shift(1)/(data['14d_ipu_mean']+1)
        data['14d_ipu_cv'] = ((data['14d_ipu_var_'] ** 0.5) + 1) / (data['14d_ipu_mean'] + 7)

        data['21d_ipu_mean'] = (data['in_price_unit'].shift(15) + data['in_price_unit'].shift(16) + data['in_price_unit'].shift(17) +
                                   data['in_price_unit'].shift(18) + data['in_price_unit'].shift(19) + data['in_price_unit'].shift(20) +
                                   data['in_price_unit'].shift(21)) / 7
        data['21d_ipu_var_'] = (data['in_price_unit'].shift(1) - data['21d_ipu_mean']) ** 2
        data['21d_ipu_trend'] = data['in_price_unit'].shift(1)/(data['21d_ipu_mean']+1)
        data['21d_ipu_cv'] = ((data['21d_ipu_var_'] ** 0.5) + 1) / (data['21d_ipu_mean'] + 7)
        return data

    #——————————————————————————————————————最后获得最终的价格特征————————————————————————
    def price_main(self,end_time,sales,code,city_name):
        new_df = price_process.data_cleaning_price(self,end_time,sales,code,city_name)
        if new_df.empty == True:
            merge_df = pd.DataFrame()
        else:
            price_df = price_process.days_feature_price_unit(self,new_df)
            # price_df.to_csv('./price_df.csv', encoding='utf_8_sig', index=False)
            pu_df = price_process.pu_days_feature(self,new_df)
            _merge = pd.merge(price_df,pu_df,on=['Account_date','sales_qty','price_unit','in_price_unit'],how='outer')
            ipu_df = price_process.ipu_days_feature(self,new_df)
            merge_df = pd.merge(ipu_df,_merge,
                                on=['Account_date','sales_qty','price_unit','in_price_unit']
                                ,how='outer').fillna(method='pad', limit=7)
        return merge_df

if __name__ == '__main__':
    sales = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/sales.csv',encoding='utf_8_sig')
    code = '02100'
    city_name = '杭州'
    price_class = price_process()
    merge_df = price_class.price_main(sales,code,city_name)
    merge_df.to_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/price_df.csv',encoding='utf_8_sig',index=False)















