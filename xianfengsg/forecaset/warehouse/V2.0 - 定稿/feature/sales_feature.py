# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 8:46
# @Author  : Ye Jinyu__jimmy
# @File    : sales_feature

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

# from pylab import *
plt.switch_backend('agg')
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,PolynomialFeatures
register_matplotlib_converters()


class sales_process():
    '''该包是要提取销售的特征，根据周作战计划的特性，选择滑窗的均值，求和，方差，以及斜率，偏度，峰度'''

    #————————————————————————首先读取销量数据————————————————————————————
    def get_sales(self):
        data = pd.read_csv('./sales.csv',encoding='utf_8_sig')
        return data

    #——————————————————————获取前一周的各类统计信息————————————————————————

    def back_1_week(self,data):

        start_date = data['Account_date'].min()
        end_date = data['Account_date'].max()
        ts = pd.Series(data['sales_qty'].values,index=pd.date_range(start_date,end_date,freq='D'))
        #----------------------均值------------------
        new_array_mean = ts.resample('W-SUN').mean()
        df_mean = pd.DataFrame(new_array_mean)
        df_mean.reset_index(level=0, inplace=True)
        df_mean.columns = ['Account_date','mean_1_week']
        #------------------------均值的斜率----------------
        slope_mean = (new_array_mean - new_array_mean.shift(1, 'W-SUN')) / 7
        df_slope_mean = pd.DataFrame(slope_mean)
        df_slope_mean.reset_index(level=0, inplace=True)
        df_slope_mean.columns = ['Account_date','slope_1_week']
        total_1_data = pd.merge(df_mean,df_slope_mean,on='Account_date')
        #----------------------求和------------------
        new_array_sum = ts.resample('W-SUN').sum()
        df_sum = pd.DataFrame(new_array_sum)
        df_sum.reset_index(level=0, inplace=True)
        df_sum.columns = ['Account_date','sum_1_week']
        total_2_data = pd.merge(df_sum,total_1_data,on='Account_date')
        #------------------------总销量的斜率----------------
        slope_sum = (new_array_sum - new_array_sum.shift(1,'W-SUN'))/7
        df_slope_sum = pd.DataFrame(slope_sum)
        df_slope_sum.reset_index(level=0, inplace=True)
        df_slope_sum.columns = ['Account_date','slope_sum_1_week']
        total_3_data = pd.merge(df_slope_sum,total_2_data,on='Account_date')
        #----------------------方差------------------
        new_array_var = ts.resample('W-SUN').var()
        df_var = pd.DataFrame(new_array_var)
        df_var.reset_index(level=0, inplace=True)
        df_var.columns = ['Account_date','var_1_week']
        total_4_data = pd.merge(df_var,total_3_data,on='Account_date')
        #------------------------方差的斜率----------------
        slope_var = (new_array_var - new_array_var.shift(1,'W-SUN'))/7
        df_slope_var = pd.DataFrame(slope_var)
        df_slope_var.reset_index(level=0, inplace=True)
        df_slope_var.columns = ['Account_date','slope_var_1_week']
        total_5_data = pd.merge(df_slope_var,total_4_data,on='Account_date')
        #----------------------分位数(0.75)------------------
        new_array_quantile_low = ts.resample('W-SUN').quantile(0.25)
        df_quantile_low = pd.DataFrame(new_array_quantile_low)
        df_quantile_low.reset_index(level=0, inplace=True)
        df_quantile_low.columns = ['Account_date','low_1_week']
        total_6_data = pd.merge(df_quantile_low,total_5_data,on='Account_date')
        #----------------------分位数(0.25)------------------
        new_array_quantile_up = ts.resample('W-SUN').quantile(0.75)
        df_quantile_up = pd.DataFrame(new_array_quantile_up)
        df_quantile_up.reset_index(level=0, inplace=True)
        df_quantile_up.columns = ['Account_date','up_1_week']
        total_7_data = pd.merge(df_quantile_up,total_6_data,on='Account_date')
        #----------------------中位数------------------
        new_array_median  = ts.resample('W-SUN').median()
        df_median  = pd.DataFrame(new_array_median)
        df_median.reset_index(level=0, inplace=True)
        df_median.columns = ['Account_date','median_1_week']
        total_8_data = pd.merge(df_median,total_7_data,on='Account_date')

        #---------------------补全之前的数据----------------------
        #格式转换
        data['Account_date'] = pd.to_datetime(data['Account_date'])
        total_8_data['Account_date'] = pd.to_datetime(total_8_data['Account_date'])
        total_9_data = pd.merge(data,total_8_data,on='Account_date',how='outer')
        #补全日期
        new_satrt = total_9_data['Account_date'].min()
        end_date  = total_9_data['Account_date'].max()
        fill_date = pd.DataFrame({'Account_date': pd.date_range(new_satrt, end_date, freq='D')})
        #合并并补充数据
        sales_feature = pd.merge(fill_date, total_9_data, how='left', on='Account_date').fillna(method='bfill')

        # #----------------------偏度和峰度------------------
        sales_feature['skew_1_week'] = ((sales_feature['sales_qty'].shift(periods=7,axis=0) - sales_feature['mean_1_week']+1)
                                       /(sales_feature['var_1_week']+7)) ** \
                                      3
        sales_feature['kurt_1_week'] = ((sales_feature['sales_qty'].shift(periods=7,axis=0) - sales_feature['mean_1_week']+1)
                                       /(sales_feature['var_1_week']+7)) ** \
                                      4
        # ------------------------变异系数--------------------------
        sales_feature['cv_1_week'] = ((sales_feature['var_1_week'] ** 0.5)+1) /(sales_feature['mean_1_week'] +7)
        return sales_feature

    #————————————————————————————————前两周的统计特征————————————————————
    def back_2_week(self,data):
        start_date = data['Account_date'].min()
        end_date = data['Account_date'].max()
        ts = pd.Series(data['sales_qty'].values,index=pd.date_range(start_date,end_date,freq='D'))

        #----------------------均值------------------
        new_array_mean = ts.resample('W-SUN').mean().shift(1, 'W-SUN')
        df_mean = pd.DataFrame(new_array_mean)
        df_mean.reset_index(level=0, inplace=True)
        df_mean.columns = ['Account_date','mean_2_week']


        #------------------------均值的斜率----------------
        slope_mean = (new_array_mean - new_array_mean.shift(1, 'W-SUN')) / 7
        df_slope_mean = pd.DataFrame(slope_mean)
        df_slope_mean.reset_index(level=0, inplace=True)
        df_slope_mean.columns = ['Account_date','slope_2_week']
        total_1_data = pd.merge(df_mean,df_slope_mean,on='Account_date')

        #----------------------求和------------------
        new_array_sum = ts.resample('W-SUN').sum().shift(1, 'W-SUN')
        df_sum = pd.DataFrame(new_array_sum)
        df_sum.reset_index(level=0, inplace=True)
        df_sum.columns = ['Account_date','sum_2_week']
        total_2_data = pd.merge(df_sum,total_1_data,on='Account_date')

        #------------------------总销量的斜率----------------
        slope_sum = (new_array_sum - new_array_sum.shift(1,'W-SUN'))/7
        df_slope_sum = pd.DataFrame(slope_sum)
        df_slope_sum.reset_index(level=0, inplace=True)
        df_slope_sum.columns = ['Account_date','slope_sum_2_week']
        total_3_data = pd.merge(df_slope_sum,total_2_data,on='Account_date')

        #----------------------方差------------------
        new_array_var = ts.resample('W-SUN').var().shift(1, 'W-SUN')
        df_var = pd.DataFrame(new_array_var)
        df_var.reset_index(level=0, inplace=True)
        df_var.columns = ['Account_date','var_2_week']
        total_4_data = pd.merge(df_var,total_3_data,on='Account_date')

        #------------------------方差的斜率----------------
        slope_var = (new_array_var - new_array_var.shift(1,'W-SUN'))/7
        df_slope_var = pd.DataFrame(slope_var)
        df_slope_var.reset_index(level=0, inplace=True)
        df_slope_var.columns = ['Account_date','slope_var_2_week']
        total_5_data = pd.merge(df_slope_var,total_4_data,on='Account_date')

        #----------------------分位数(0.75)------------------
        new_array_quantile_low = ts.resample('W-SUN').quantile(0.25).shift(1, 'W-SUN')
        df_quantile_low = pd.DataFrame(new_array_quantile_low)
        df_quantile_low.reset_index(level=0, inplace=True)
        df_quantile_low.columns = ['Account_date','low_2_week']
        total_6_data = pd.merge(df_quantile_low,total_5_data,on='Account_date')

        #----------------------分位数(0.25)------------------
        new_array_quantile_up = ts.resample('W-SUN').quantile(0.75).shift(1, 'W-SUN')
        df_quantile_up = pd.DataFrame(new_array_quantile_up)
        df_quantile_up.reset_index(level=0, inplace=True)
        df_quantile_up.columns = ['Account_date','up_2_week']
        total_7_data = pd.merge(df_quantile_up,total_6_data,on='Account_date')

        #----------------------中位数------------------
        new_array_median  = ts.resample('W-SUN').median().shift(1, 'W-SUN')
        df_median  = pd.DataFrame(new_array_median)
        df_median.reset_index(level=0, inplace=True)
        df_median.columns = ['Account_date','median_2_week']
        total_8_data = pd.merge(df_median,total_7_data,on='Account_date')

        #---------------------补全之前的数据----------------------
        #---------------------------------格式转换---------------------
        data['Account_date'] = pd.to_datetime(data['Account_date'])
        total_8_data['Account_date'] = pd.to_datetime(total_8_data['Account_date'])
        total_9_data = pd.merge(data,total_8_data,on='Account_date',how='outer')
        #------------------------补全日期------------------------
        new_satrt = total_9_data['Account_date'].min()
        end_date  = total_9_data['Account_date'].max()
        fill_date = pd.DataFrame({'Account_date': pd.date_range(new_satrt, end_date, freq='D')})
        #--------------------------合并并补充数据-------------------------------------
        sales_feature = pd.merge(fill_date, total_9_data, how='left', on='Account_date').fillna(method='bfill')

        # #----------------------偏度和峰度------------------
        sales_feature['skew_2_week'] = ((sales_feature['sales_qty'].shift(periods=14,axis=0)
                                           - sales_feature['mean_2_week']+1)
                                       /(sales_feature['var_2_week']+7)) ** \
                                      3
        sales_feature['kurt_2_week'] = ((sales_feature['sales_qty'].shift(periods=14,axis=0)
                                           - sales_feature['mean_2_week']+1)
                                       /(sales_feature['var_2_week']+7)) ** \
                                      4
        sales_feature['cv_2_week'] = ((sales_feature['var_2_week'] ** 0.5) + 1) / (sales_feature['mean_2_week'] + 7)
        return sales_feature

    #————————————————————————————————前三周的统计特征————————————————————
    def back_3_week(self,data):
        start_date = data['Account_date'].min()
        end_date = data['Account_date'].max()
        ts = pd.Series(data['sales_qty'].values,index=pd.date_range(start_date,end_date,freq='D'))

        #----------------------均值------------------
        new_array_mean = ts.resample('W-SUN').mean().shift(2, 'W-SUN')
        df_mean = pd.DataFrame(new_array_mean)
        df_mean.reset_index(level=0, inplace=True)
        df_mean.columns = ['Account_date','mean_3_week']


        #------------------------均值的斜率----------------
        slope_mean = (new_array_mean - new_array_mean.shift(1, 'W-SUN')) / 7
        df_slope_mean = pd.DataFrame(slope_mean)
        df_slope_mean.reset_index(level=0, inplace=True)
        df_slope_mean.columns = ['Account_date','slope_3_week']
        total_1_data = pd.merge(df_mean,df_slope_mean,on='Account_date')

        #----------------------求和------------------
        new_array_sum = ts.resample('W-SUN').sum().shift(2, 'W-SUN')
        df_sum = pd.DataFrame(new_array_sum)
        df_sum.reset_index(level=0, inplace=True)
        df_sum.columns = ['Account_date','sum_3_week']
        total_2_data = pd.merge(df_sum,total_1_data,on='Account_date')

        #------------------------总销量的斜率----------------
        slope_sum = (new_array_sum - new_array_sum.shift(1,'W-SUN'))/7
        df_slope_sum = pd.DataFrame(slope_sum)
        df_slope_sum.reset_index(level=0, inplace=True)
        df_slope_sum.columns = ['Account_date','slope_sum_3_week']
        total_3_data = pd.merge(df_slope_sum,total_2_data,on='Account_date')

        #----------------------方差------------------
        new_array_var = ts.resample('W-SUN').var().shift(2, 'W-SUN')
        df_var = pd.DataFrame(new_array_var)
        df_var.reset_index(level=0, inplace=True)
        df_var.columns = ['Account_date','var_3_week']
        total_4_data = pd.merge(df_var,total_3_data,on='Account_date')

        #------------------------方差的斜率----------------
        slope_var = (new_array_var - new_array_var.shift(1,'W-SUN'))/7
        df_slope_var = pd.DataFrame(slope_var)
        df_slope_var.reset_index(level=0, inplace=True)
        df_slope_var.columns = ['Account_date','slope_var_3_week']
        total_5_data = pd.merge(df_slope_var,total_4_data,on='Account_date')

        #----------------------分位数(0.75)------------------
        new_array_quantile_low = ts.resample('W-SUN').quantile(0.25).shift(2, 'W-SUN')
        df_quantile_low = pd.DataFrame(new_array_quantile_low)
        df_quantile_low.reset_index(level=0, inplace=True)
        df_quantile_low.columns = ['Account_date','low_3_week']
        total_6_data = pd.merge(df_quantile_low,total_5_data,on='Account_date')

        #----------------------分位数(0.25)------------------
        new_array_quantile_up = ts.resample('W-SUN').quantile(0.75).shift(2, 'W-SUN')
        df_quantile_up = pd.DataFrame(new_array_quantile_up)
        df_quantile_up.reset_index(level=0, inplace=True)
        df_quantile_up.columns = ['Account_date','up_3_week']
        total_7_data = pd.merge(df_quantile_up,total_6_data,on='Account_date')

        #----------------------中位数------------------
        new_array_median  = ts.resample('W-SUN').median().shift(2, 'W-SUN')
        df_median  = pd.DataFrame(new_array_median)
        df_median.reset_index(level=0, inplace=True)
        df_median.columns = ['Account_date','median_3_week']
        total_8_data = pd.merge(df_median,total_7_data,on='Account_date')

        #---------------------补全之前的数据----------------------
        #---------------------------------格式转换---------------------
        data['Account_date'] = pd.to_datetime(data['Account_date'])
        total_8_data['Account_date'] = pd.to_datetime(total_8_data['Account_date'])
        total_9_data = pd.merge(data,total_8_data,on='Account_date',how='outer')
        #------------------------补全日期------------------------
        new_satrt = total_9_data['Account_date'].min()
        end_date  = total_9_data['Account_date'].max()
        fill_date = pd.DataFrame({'Account_date': pd.date_range(new_satrt, end_date, freq='D')})
        #--------------------------合并并补充数据-------------------------------------
        sales_feature = pd.merge(fill_date, total_9_data, how='left', on='Account_date').fillna(method='bfill')

        # #----------------------偏度和峰度------------------
        sales_feature['skew_3_week'] = ((sales_feature['sales_qty'].shift(periods=21,axis=0)
                                           - sales_feature['mean_3_week']+1)
                                       /(sales_feature['var_3_week']+7)) ** \
                                      3
        sales_feature['kurt_3_week'] = ((sales_feature['sales_qty'].shift(periods=21,axis=0)
                                           - sales_feature['mean_3_week']+1)
                                       /(sales_feature['var_3_week']+7)) ** \
                                      4
        sales_feature['cv_3_week'] = ((sales_feature['var_3_week'] ** 0.5) + 1) / (sales_feature['mean_3_week'] + 7)

        return sales_feature


    #————————————————————————————————前四周的统计特征————————————————————
    def back_4_week(self,data):
        start_date = data['Account_date'].min()
        end_date = data['Account_date'].max()
        ts = pd.Series(data['sales_qty'].values,index=pd.date_range(start_date,end_date,freq='D'))

        #----------------------均值------------------
        new_array_mean = ts.resample('W-SUN').mean().shift(3, 'W-SUN')
        df_mean = pd.DataFrame(new_array_mean)
        df_mean.reset_index(level=0, inplace=True)
        df_mean.columns = ['Account_date','mean_4_week']


        #------------------------均值的斜率----------------
        slope_mean = (new_array_mean - new_array_mean.shift(1, 'W-SUN')) / 7
        df_slope_mean = pd.DataFrame(slope_mean)
        df_slope_mean.reset_index(level=0, inplace=True)
        df_slope_mean.columns = ['Account_date','slope_4_week']
        total_1_data = pd.merge(df_mean,df_slope_mean,on='Account_date')

        #----------------------求和------------------
        new_array_sum = ts.resample('W-SUN').sum().shift(3, 'W-SUN')
        df_sum = pd.DataFrame(new_array_sum)
        df_sum.reset_index(level=0, inplace=True)
        df_sum.columns = ['Account_date','sum_4_week']
        total_2_data = pd.merge(df_sum,total_1_data,on='Account_date')

        #------------------------总销量的斜率----------------
        slope_sum = (new_array_sum - new_array_sum.shift(1,'W-SUN'))/7
        df_slope_sum = pd.DataFrame(slope_sum)
        df_slope_sum.reset_index(level=0, inplace=True)
        df_slope_sum.columns = ['Account_date','slope_sum_4_week']
        total_3_data = pd.merge(df_slope_sum,total_2_data,on='Account_date')

        #----------------------方差------------------
        new_array_var = ts.resample('W-SUN').var().shift(3, 'W-SUN')
        df_var = pd.DataFrame(new_array_var)
        df_var.reset_index(level=0, inplace=True)
        df_var.columns = ['Account_date','var_4_week']
        total_4_data = pd.merge(df_var,total_3_data,on='Account_date')

        #------------------------方差的斜率----------------
        slope_var = (new_array_var - new_array_var.shift(1,'W-SUN'))/7
        df_slope_var = pd.DataFrame(slope_var)
        df_slope_var.reset_index(level=0, inplace=True)
        df_slope_var.columns = ['Account_date','slope_var_4_week']
        total_5_data = pd.merge(df_slope_var,total_4_data,on='Account_date')

        #----------------------分位数(0.75)------------------
        new_array_quantile_low = ts.resample('W-SUN').quantile(0.25).shift(3, 'W-SUN')
        df_quantile_low = pd.DataFrame(new_array_quantile_low)
        df_quantile_low.reset_index(level=0, inplace=True)
        df_quantile_low.columns = ['Account_date','low_4_week']
        total_6_data = pd.merge(df_quantile_low,total_5_data,on='Account_date')

        #----------------------分位数(0.25)------------------
        new_array_quantile_up = ts.resample('W-SUN').quantile(0.75).shift(3, 'W-SUN')
        df_quantile_up = pd.DataFrame(new_array_quantile_up)
        df_quantile_up.reset_index(level=0, inplace=True)
        df_quantile_up.columns = ['Account_date','up_4_week']
        total_7_data = pd.merge(df_quantile_up,total_6_data,on='Account_date')

        #----------------------中位数------------------
        new_array_median  = ts.resample('W-SUN').median().shift(3, 'W-SUN')
        df_median  = pd.DataFrame(new_array_median)
        df_median.reset_index(level=0, inplace=True)
        df_median.columns = ['Account_date','median_4_week']
        total_8_data = pd.merge(df_median,total_7_data,on='Account_date')

        #---------------------补全之前的数据----------------------
        #---------------------------------格式转换---------------------
        data['Account_date'] = pd.to_datetime(data['Account_date'])
        total_8_data['Account_date'] = pd.to_datetime(total_8_data['Account_date'])
        total_9_data = pd.merge(data,total_8_data,on='Account_date',how='outer')
        #------------------------补全日期------------------------
        new_satrt = total_9_data['Account_date'].min()
        end_date  = total_9_data['Account_date'].max()
        fill_date = pd.DataFrame({'Account_date': pd.date_range(new_satrt, end_date, freq='D')})
        #--------------------------合并并补充数据-------------------------------------
        sales_feature = pd.merge(fill_date, total_9_data, how='left', on='Account_date').fillna(method='bfill')

        # #----------------------偏度和峰度------------------
        sales_feature['skew_4_week'] = ((sales_feature['sales_qty'].shift(periods=28,axis=0)
                                           - sales_feature['mean_4_week']+1)
                                       /(sales_feature['var_4_week']+7)) ** \
                                      3
        sales_feature['kurt_4_week'] = ((sales_feature['sales_qty'].shift(periods=28,axis=0)
                                           - sales_feature['mean_4_week']+1)
                                       /(sales_feature['var_4_week']+7)) ** \
                                      4
        sales_feature['cv_4_week'] = ((sales_feature['var_4_week'] ** 0.5) + 1) / (sales_feature['mean_4_week'] + 7)

        return sales_feature


    #————————————————————————————————————周粒度的销量特征————————————————————
    def week_feature(self,data):
        sales_feature_1 = sales_process.back_1_week(self,data)
        sales_feature_2 = sales_process.back_2_week(self,data)
        new_df_1 = pd.merge(sales_feature_1,sales_feature_2,on=['Account_date','sales_qty'],how='outer')
        sales_feature_3 = sales_process.back_3_week(self,data)
        new_df_2 = pd.merge(new_df_1,sales_feature_3,on=['Account_date','sales_qty'],how='outer')
        sales_feature_4 = sales_process.back_4_week(self,data)
        new_df_3 = pd.merge(new_df_2,sales_feature_4,on=['Account_date','sales_qty'],how='outer')
        return new_df_3

    #——————————————————————————————————日粒度的销售特征——————————————————————————————
    #----------------------------前一天，前三天，前7天--------------------------------
    def days_feature(self,data):
        start_date = data['Account_date'].min()
        end_date = data['Account_date'].max()
        ts = pd.Series(data['sales_qty'].values, index=pd.date_range(start_date, end_date, freq='D'))

        new_array_1day = (ts - ts.shift(1,'D'))
        df_1day = pd.DataFrame(new_array_1day)
        df_1day.reset_index(level=0, inplace=True)
        df_1day.columns = ['Account_date', 'back_1day']

        new_array_3day = (ts - ts.shift(3,'D'))
        df_3day = pd.DataFrame(new_array_3day)
        df_3day.reset_index(level=0, inplace=True)
        df_3day.columns = ['Account_date', 'back_3day']
        total_3day = pd.merge(df_1day,df_3day,on='Account_date',how='outer')

        new_array_7day = (ts - ts.shift(7,'D'))
        df_7day = pd.DataFrame(new_array_7day)
        df_7day.reset_index(level=0, inplace=True)
        df_7day.columns = ['Account_date', 'back_7day']
        total_7day = pd.merge(total_3day,df_7day,on='Account_date',how='outer')

        data['Account_date'] = pd.to_datetime(data['Account_date'])
        total_7day['Account_date'] = pd.to_datetime(total_7day['Account_date'])

        total_7day = pd.merge(data,total_7day,on='Account_date',how='outer')

        return total_7day


    #----------------------------前7日销量的统计特征与前14天的特征和前21天的特征的变化----------------------
    def sales_days_feature(self,data):
        data['last_7days_mean'] = (data['sales_qty'].shift(1) + data['sales_qty'].shift(2) + data['sales_qty'].shift(3) +
                                   data['sales_qty'].shift(4) + data['sales_qty'].shift(5) + data['sales_qty'].shift(6) +
                                   data['sales_qty'].shift(7)) / 7
        data['last_7days_var'] = (data['sales_qty'].shift(1) - data['last_7days_mean']) ** 2
        data['last_7days_trend'] = data['sales_qty'].shift(1)/(data['last_7days_mean']+1)
        data['cv_7days'] = ((data['last_7days_var'] ** 0.5) + 1) / (data['last_7days_mean'] + 7)

        data['last_14days_mean'] = (data['sales_qty'].shift(8) + data['sales_qty'].shift(9) + data['sales_qty'].shift(10) +
                                   data['sales_qty'].shift(13) + data['sales_qty'].shift(12) + data['sales_qty'].shift(11) +
                                   data['sales_qty'].shift(14)) / 7
        data['last_14days_var'] = (data['sales_qty'].shift(1) - data['last_14days_mean']) ** 2
        data['last_14days_trend'] = data['sales_qty'].shift(1)/(data['last_14days_mean']+1)
        data['cv_14days'] = ((data['last_14days_var'] ** 0.5) + 1) / (data['last_14days_mean'] + 7)

        data['last_21days_mean'] = (data['sales_qty'].shift(8) + data['sales_qty'].shift(9) + data['sales_qty'].shift(10) +
                                   data['sales_qty'].shift(13) + data['sales_qty'].shift(12) + data['sales_qty'].shift(11) +
                                   data['sales_qty'].shift(14)) / 7
        data['last_21days_var'] = (data['sales_qty'].shift(1) - data['last_21days_mean']) ** 2
        data['last_21days_trend'] = data['sales_qty'].shift(1)/(data['last_21days_mean']+1)
        data['cv_21days'] = ((data['last_21days_var'] ** 0.5) + 1) / (data['last_21days_mean'] + 7)
        return data

#——————————————————————————————获取对销量的特征函数————————————————
def feature(sales):
    sales_class = sales_process()
    start_date = sales['Account_date'].min()
    end_date = sales['Account_date'].max()
    date_range_sku = pd.date_range(start_date, end_date, freq='D')
    data_sku = pd.DataFrame({'Account_date': date_range_sku})
    data_sku.to_csv('./data_sku.csv', encoding='utf_8_sig')
    result = pd.merge(sales, data_sku, how='right', on='Account_date')
    result["sales_qty"].iloc[np.where(np.isnan(result["sales_qty"]))] = 0
    result.to_csv('./result.csv', encoding='utf_8_sig')

    new_df_1 = sales_class.week_feature(result)
    new_df_1.to_csv('./new_df_1.csv', encoding='utf_8_sig')
    days_feature = sales_class.days_feature(result)
    sales_days_feature = sales_class.sales_days_feature(result)
    sales_days_feature.to_csv('./sales_days_feature.csv',encoding='utf_8_sig')
    new_df_2 = pd.merge(new_df_1, days_feature, on=['Account_date', 'sales_qty'], how='left')

    new_df_2.to_csv('./new_df_2.csv',encoding='utf_8_sig')
    new_df = pd.merge(new_df_2, sales_days_feature, on=['Account_date', 'sales_qty'], how='left')

    new_df.to_csv('./new_df.csv',encoding='utf_8_sig')
    return new_df


if __name__ == '__main__':
    sales = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/sales.csv',encoding='utf_8_sig')
    # sales_class = sales_process()
    # new_df_1 = sales_class.week_feature(sales)
    # days_feature = sales_class.days_feature(sales)
    # sales_days_feature = sales_class.sales_days_feature(sales)
    #
    # new_df_2 = pd.merge(new_df_1, days_feature, on=['Account_date', 'sales_qty'], how='left')
    # new_df = pd.merge(new_df_2, sales_days_feature, on=['Account_date', 'sales_qty'], how='left')

    new_df  = feature(sales)
    new_df.to_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/new_df.csv',encoding='utf_8_sig')