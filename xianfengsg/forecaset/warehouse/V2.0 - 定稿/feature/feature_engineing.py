# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 14:52
# @Author  : Ye Jinyu__jimmy
# @File    : feature_engineing

from sklearn import preprocessing
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import math
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA

from matplotlib.pylab import rcParams
import psycopg2
import pymysql
import time
import datetime
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

'''先将所有的特征与数据进行合并,因为后面将进行树模型的学习和训练，因此不需要归一化'''
#——————————————————————————————————先获取各个的原始销量与特征数据————————————————————————————————
def data_prepare(weather_feature,sales_feature,price,time_df):
    # weather_feature = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/weather_sql_read.csv',encoding='utf_8_sig',index_col=0)
    # # sales = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data//sales.csv',encoding='utf_8_sig',index_col=0)
    # price = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data//price_df.csv',encoding='utf_8_sig',index_col=0)
    # time_df = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/time_df.csv',encoding='utf_8_sig',index_col=0)
    # sales_feature = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data//new_df.csv',encoding='utf_8_sig',index_col=0)
    merge_01 = pd.merge(price,sales_feature,on=['Account_date','sales_qty'],how='left').fillna(method='pad')
    price.to_csv('./price.csv', encoding='utf_8_sig', index=False)
    weather_feature.to_csv('./weather_feature.csv', encoding='utf_8_sig', index=False)
    sales_feature.to_csv('./sales_feature.csv', encoding='utf_8_sig', index=False)
    merge_01.to_csv('./merge_01.csv',encoding='utf_8_sig',index=False)
    merge_02 = pd.merge(merge_01,weather_feature,on='Account_date',how='right')
    merge_02.to_csv('./merge_02.csv', encoding='utf_8_sig', index=False)
    merge = pd.merge(merge_02,time_df,on=['Account_date'],how='left')

    merge = merge.dropna(how='any')
    merge["sales_qty"].iloc[np.where(merge["sales_qty"] < 0)] = 0
    merge.to_csv('./merge.csv', encoding='utf_8_sig', index=False)
    return merge

#————————————————————————————返回所有的测试——————————————
if __name__ == '__main__':
    weather_feature = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/weather_sql_read.csv',encoding='utf_8_sig',index_col=0)
    # sales = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data//sales.csv',encoding='utf_8_sig',index_col=0)
    price = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data//price_df.csv',encoding='utf_8_sig',index_col=0)
    time_df = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/time_df.csv',encoding='utf_8_sig',index_col=0)
    sales_feature = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data//new_df.csv',encoding='utf_8_sig',index_col=0)

    merge = data_prepare(weather_feature,sales_feature,price,time_df)






