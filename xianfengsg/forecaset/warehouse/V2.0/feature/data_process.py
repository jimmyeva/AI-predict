# -*- coding: utf-8 -*-
# @Time    : 2020/2/29 15:37
# @Author  : Ye Jinyu__jimmy
# @File    : data_process

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
import matplotlib
matplotlib.use('TkAgg')

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


'''这个函数是用来读取数据，并进行原始的数据处理'''

#——————————————————————————————封装销量数据清洗的类————————————————
class sales_cleaning():
    #——————————————————————————————————对销量进行基础的清洗————————————————————————————
    #----------------------通过统计学方式选择最先产生有效数据的点的时间节点------------------------
    def data_cleaning(self,df_sales):
        # df_sales.to_csv('./df_sales.csv',encoding='utf_8_sig')
        ts = df_sales['sales_qty']
        low = ts.quantile(0.25) - 1.5 * (ts.quantile(0.75) - ts.quantile(0.25))
        if low < 0:
            low = 0
        else:
            pass
        noise_data = df_sales[df_sales['sales_qty'] >= low]

        #进行切片操作，连续十四天没有销量
        noise_data = noise_data.sort_values('Account_date',ascending='asc')
        ts = pd.Series(noise_data['Account_date'].values, index = None)
        #---------------------做一次分解画图——---------------------
        # sales_cleaning.decompose(self,noise_data)

        dif = ts.diff().dropna()  # 差分序列
        dates = datetime.timedelta(7)
        forbid_index = dif[(dif >= dates)].index
        if len(forbid_index) > 1:
            new_df = pd.DataFrame()
            for i in range(len(forbid_index)+1):
                print('正在进行切片计算的长度',i)
                if i ==0:
                    segment_df = noise_data[0:forbid_index[0]]
                    print('segment_df',len(segment_df),segment_df)
                    seg_df = sales_cleaning.anomaly_process(self,segment_df)
                    new_df = new_df.append(seg_df)
                elif i == len(forbid_index):
                    segment_df = noise_data[forbid_index[i-1]:]
                    seg_df = sales_cleaning.anomaly_process(self, segment_df)
                    new_df = new_df.append(seg_df)
                else:
                    segment_df = noise_data[forbid_index[i-1]:forbid_index[i]]
                    print('segment_df',len(segment_df),segment_df)
                    seg_df = sales_cleaning.anomaly_process(self,segment_df)
                    new_df = new_df.append(seg_df)
        else:
            new_df = df_sales

        # new_df.to_csv('./new_df00.csv', encoding='utf_8_sig')
        #
        # date =noise_data['Account_date']
        # Qty = noise_data['sales_qty']
        # fig = plt.figure(figsize=(20, 10), facecolor='white')
        # ax1 = fig.add_subplot(111)
        # # 左轴
        # ax1.bar(date, Qty, width=0.5, align='center', label='real_qty', color="black")
        # plt.show()

        return new_df


    #-----------------------近似熵-------------------
    def ApEn(self,U, m, r):

        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

        def _phi(m):
            x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
            C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
            return (N - m + 1.0)**(-1) * sum(np.log(C))

        N = len(U)

        return abs(_phi(m+1) - _phi(m))

    # Usage example
    # U = np.array([85, 80, 89] * 100)




    def decompose(self,timeseries):
        # 返回包含三个部分 trend（趋势部分） ， seasonal（季节性部分） 和residual (残留部分)
        start_date = timeseries['Account_date'].min()
        end_date = timeseries['Account_date'].max()
        full_date = pd.DataFrame({'Account_date':pd.date_range(start=start_date,end=end_date)})
        merge_df = pd.merge(full_date,timeseries,on='Account_date',how='left').fillna(0)
        ts = pd.Series(merge_df['sales_qty'].values, index= pd.date_range(start_date, end_date, freq='D'))
        decomposition = seasonal_decompose(ts)

        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        print('画图')
        # print('trend',trend)
        plt.subplot(411)
        plt.plot(ts, label='Original')
        plt.legend(loc='best')
        plt.subplot(412)
        plt.plot(trend, label='Trend')
        plt.legend(loc='best')
        plt.subplot(413)
        plt.plot(seasonal, label='Seasonality')
        plt.legend(loc='best')
        plt.subplot(414)
        plt.plot(residual, label='Residuals')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('./decompose.jpg', dpi=600, bbox_inches='tight')
        plt.close()
        # plt.show()

        # return trend, seasonal, residual

    #对数据进行清洗
    def diff_smooth(self,data):
        start_date = data['Account_date'].min()
        end_date = data['Account_date'].max()
        ts = pd.Series(data['sales_qty'].values,index=pd.date_range(start_date,end_date,freq='D'))
        dif = ts.diff().dropna() # 差分序列
        td = dif.describe() # 描述性统计得到：min，25%，50%，75%，max值
        high = td['75%'] + 1.5 * (td['75%'] - td['25%']) # 定义高点阈值，1.5倍四分位距之外
        low = td['25%'] - 1.5 * (td['75%'] - td['25%']) # 定义低点阈值，同上
        # 变化幅度超过阈值的点的索引
        forbid_index = dif[(dif > high) | (dif < low)].index
        i = 0
        while i < len(forbid_index) - 1:
            n = 1 # 发现连续多少个点变化幅度过大，大部分只有单个点
            start = forbid_index[i] # 异常点的起始索引
            # print(i)
            while forbid_index[i+n] == start + datetime.timedelta(days=n):
                n += 1
                if i+n == len(forbid_index):
                    n -= 1
                    break
            i += n - 1
            end = forbid_index[i] # 异常点的结束索引
            # 用前后值的中间值均匀填充
            value = np.linspace(ts[start - datetime.timedelta(days=1)], ts[end + datetime.timedelta(days=1)], n)
            ts[start: end] = value
            i += 1

        new_df = pd.DataFrame({'Account_date':ts.index,'sales_smooth':ts.values})
        # ts.columns = ['Account_date','sales_smooth']
        return new_df


    #-------------------EWMA--------------
    def process_abnormal(self,data):
        mid_data = data
        # Q1 = mid_data['sales_qty'].quantile(q=0.25)
        # Q3 = mid_data['sales_qty'].quantile(q=0.75)
        # IQR = Q3 - Q1
        # anomaly_points_how = float(Q3 + 1.5 * IQR)
        # anomaly_points_low = float(Q1 - 1.5 * IQR)
        sales_list = pd.Series(mid_data['sales_qty'].values,index=None)

        td = sales_list.describe() # 描述性统计得到：min，25%，50%，75%，max值
        high = td['75%'] + 1.5 * (td['75%'] - td['25%']) # 定义高点阈值，1.5倍四分位距之外
        low = td['25%'] - 1.5 * (td['75%'] - td['25%']) # 定义低点阈值，同上
        half_life = 3
        alpha = 1 - math.exp(math.log(0.5) / half_life)
        for i in range(len(sales_list)):
            if i >= 3:
                if sales_list[i] > high :
                    mean = np.array(sales_list[i-3] +sales_list[i-2]+sales_list[i-1])/3
                    sales_list[i] = alpha *sales_list[i] + (1-alpha) * mean
                elif sales_list[i] < low:
                    # print('i',i)
                    # print('sales_list',sales_list)
                    # print('sales_list[i - 1]',sales_list[i - 1])
                    mean = np.array(sales_list[i - 3] + sales_list[i - 2] + sales_list[i - 1]) / 3
                    sales_list[i] = alpha * sales_list[i] + (1 - alpha) * mean
                else:
                    pass
            elif i == 2:
                # print('sales_list',sales_list)
                # print('high',high)
                # print('sales_list[i]',sales_list[i])
                if sales_list[i] > high :
                    mean = np.array(sales_list[i-2]+sales_list[i-1])/2
                    sales_list[i] = alpha *sales_list[i] + (1-alpha) * mean
                elif sales_list[i] < low:
                    mean = np.array(sales_list[i - 2] + sales_list[i - 1]) / 2
                    sales_list[i] = alpha * sales_list[i] + (1 - alpha) * mean
                else:
                    pass
            else:
                pass

        return mid_data




    def sigle_holt_winters(self,data):
        #先修正那些明显的错误的数据
        # data["Sales_qty"].iloc[np.where(data["Sales_qty"] < 0) ] = 0
        sales = data.drop(data[data.sales_qty <= 0].index)

        y = pd.Series(sales['sales_qty'].values)
        date = pd.Series(sales['Account_date'].values)
        seasonal = round(len(sales) / 4)

        ets3 = ExponentialSmoothing(y, trend='add', seasonal='add',seasonal_periods=seasonal)
        r3 = ets3.fit()
        anomaly_data = pd.DataFrame({
            'Account_date': date,
            'fitted': r3.fittedvalues,
        })
        merge_data = pd.merge(data,anomaly_data,on='Account_date',how='inner')
        merge_data.drop('sales_qty',axis=1, inplace=True)
        merge_data = merge_data.rename(columns={'fitted':'sales_qty'})
        #三阶指数平滑有可能也出现的负值，因此作出处理

        # except OSError as reason:
        #     print('出错原因是:' +str(reason))
        #     pass
        #     merge_data = pd.DataFrame()
        return merge_data



    #-------------------------数据噪声处理--------------------------
    def anomaly_process(self,df):
        if len(df) > 365:
            new_df = sales_cleaning.sigle_holt_winters(self,df)
        else:
            new_df = sales_cleaning.process_abnormal(self,df)
        return new_df



    #-------------------------长期异常点检查------------------------
    def LS_detection(self,data):
        exp_avg = pd.stats.moments.ewma(data,span=50)
        std_dev = pd.stats.moments.ewmstd(data,com=50)
        if abs(data.values[-1] - exp_avg.values[-1]) > 3 * std_dev.values[-1]:
            print('异常')



    #——————————————————————————补全日期————————————————————————————
    def fill_date(self,data,start,end_date):
        date_range_sku = pd.date_range(start,end_date,freq='D')
        data_sku = pd.DataFrame({'Account_date': date_range_sku})
        result = pd.merge(data, data_sku, how='right', on='Account_date')
        result["sales_qty"].iloc[np.where(np.isnan(result["sales_qty"]))] = 0
        result = result.sort_values('Account_date',ascending='asc')
        return result


if __name__ == '__main__':
    print('运行当前类')


