# -*- coding: utf-8 -*-
# @Time    : 2019/7/11 11:12
# @Author  : Ye Jinyu__jimmy
# @File    : compare_forecast_real.py
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


#读取预测数据这里是csv的文件
def read_forecast_data(old_path,new_path):
    data_old = pd.read_csv(old_path,encoding='utf_8_sig',low_memory=False)
    data_old = data_old.rename(index=str, columns={'forecast_qty':'old_forecast'})
    data_new = pd.read_csv(new_path, encoding='utf_8_sig', low_memory=False)
    data = pd.merge(data_new,data_old,on=['sku_id','account_date'],how='inner')
    return data


#读取真实的历史销售
#------------------------------------------------------------------>根据SKU 的id来获取每个SKU的具体的销售明细数据

def get_detail_sales_data(sku_id,start_date,end_date):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    dsn = cx_Oracle.makedsn(host, port, sid)

    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码)
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", dsn)
    # 查看出货详细单的数据
    stkout_detail_sql = """SELECT s.GDGID,b.NUM,s.RTOTAL,b.OCRDATE,s.CRTOTAL,s.MUNIT,s.QTY,s.QTYSTR,
                            s.TOTAL,s.PRICE,s.QPC FROM STKOUTDTL s INNER JOIN(
                            select *
                            from STKOUT s
                            INNER JOIN STORE s1  ON s.sender = s1.gid
                            INNER JOIN STORE s2 ON s.CLIENT = s2.gid 
                            WHERE bitand(s1.property,32)=32 
                            AND bitand(s2.property,32)<>32 
                            AND substr(s2.AREA,2,3)<'8000' 
                            AND s.CLS='统配出')b ON s.NUM = b.NUM AND s.CLS='统配出' 
                            and s.GDGID = %s and  s.TOTAL> 0 and b.OCRDATE >= to_date('%s','yyyy-mm-dd')
                                and b.OCRDATE <= to_date('%s','yyyy-mm-dd')""" % (sku_id,start_date,end_date)
    stkout_detail = pd.read_sql(stkout_detail_sql, conn)
    conn.close
    return stkout_detail

#--------------------------------------------------------------------------------->日的标准化转化
def date_normalize(data_frame):
    data_frame_sort = data_frame.sort_values(by = ['account_date'],ascending=False )
    data_frame_sort['account_date'] = pd.to_datetime(data_frame_sort['account_date']).dt.normalize()
    return data_frame_sort


#------------------------------------------------------------------------->设置函数按照预测的数据
def get_real_data(forecast_data):
    sku_id_list = set(forecast_data['sku_id'])
    print(sku_id_list)
    start_date = forecast_data['account_date'].min()
    end_date = forecast_data['account_date'].max()
    print(end_date)
    print(type(end_date))
    final_data = pd.DataFrame()
    for x in tqdm(sku_id_list):
        real_sales = get_detail_sales_data(x,start_date,end_date)
        real_sales = real_sales.rename(index=str, columns={'OCRDATE': 'account_date', 'GDGID': 'sku_id'})
        real_sales = date_normalize(real_sales)
        real_sales = real_sales.groupby(["account_date",'sku_id'],as_index = False).agg(sum)
        print(x)
        print(real_sales)
        final_data = final_data.append(real_sales)
    return final_data



#---------------------------------------------------------------------->将真实销售数据和预测值进行匹配
def merge_forecast_real(forecast_data):
    real_sales = get_real_data(forecast_data)
    forecast_data = date_normalize(forecast_data)
    merge_data = pd.merge(forecast_data,real_sales,on=['sku_id','account_date'],how='inner')
    merge_data = merge_data.fillna(0)
    return merge_data


#----------------------------------------->查看和比较真实和预测数据的真实误差，并查看误差的统计学指标，保存到csv
def description_error(data):
    data['new_abs_error'] = data['forecast_qty'] -data['QTY']
    data['relative_error'] = data['new_abs_error']/data['QTY']
    data['old_abs_error'] = data['old_forecast'] -data['QTY']
    data['old_relative_error'] = data['old_abs_error']/data['QTY']
    return data


#---------------------------------------------------------------------->对每个SKU进行真实和预测对比
def plot_compare(data):
    sku_id = set(data['sku_id'])
    # box_data = data[['relative_error']]
    # f = box_data.boxplot(sym='.', vert=True, whis=1.5,
    #                      patch_artist=True, meanline=False,
    #                      showmeans=True, showbox=True,
    #                      showcaps=True, showfliers=True,
    #                      notch=False, return_type='dict')
    # for box in f['boxes']:
    #     box.set(color='b', linewidth=1)  # 箱体边框颜色
    #     box.set(facecolor='b', alpha=0.5)  # 箱体内部填充颜色
    # for whisker in f['whiskers']:
    #     whisker.set(color='k', linewidth=0.5, linestyle='-')
    # for cap in f['caps']:
    #     cap.set(color='gray', linewidth=2)
    # for median in f['medians']:
    #     median.set(color='DarkBlue', linewidth=2)
    # for flier in f['fliers']:
    #     flier.set(marker='.', color='y', alpha=0.5)
    # # boxes： 箱线
    # # medians： 中位值的横线,
    # # whiskers： 从box到error bar之间的竖线.
    # # fliers： 异常值
    # # caps： error bar横线
    # # means： 均值的横线
    #
    # plt.title('boxplot')
    # plt.savefig('D:/jimmy-ye/AI_supply_chain/data/forecast/boxplot.jpg', dpi=600,
    #             bbox_inches='tight')
    # plt.close()
    # data.to_csv('D:/jimmy-ye/AI_supply_chain/data/forecast/data_total.csv',
    #                    encoding="utf_8_sig")
    for id in tqdm(sku_id):
        print(id)
        mid_data = data[data['sku_id'] == id]
        total_error = mid_data.describe()
        print(total_error)
        total_error.to_csv('D:/jimmy-ye/AI_supply_chain/data/forecast_holiday/total_corr'+str(id)+'.csv',
                           encoding="utf_8_sig")
        print(mid_data)
        date = mid_data['account_date']
        forecast_qty = mid_data['forecast_qty']
        old_forecast = mid_data['old_forecast']
        real_qty = mid_data['QTY']
        fig = plt.figure(figsize=(20,10),facecolor='white')
        ax1 = fig.add_subplot(111)
        # 左轴
        ax1.bar(date, real_qty, width=0.5, align='center', label='real_qty', color="black")
        plt.legend(loc='upper left', fontsize=10)
        ax1.plot(date, forecast_qty, color='red', marker='o', linestyle='dashed', label='forecast_qty',
                 markersize=0.8)
        ax1.plot(date, old_forecast, color='green', marker='o', linestyle='dashed', label='old_forecast',
                 markersize=0.8)
        plt.legend(loc='upper right', fontsize=10)
        ax1.set_xlabel('date')
        ax1.set_ylabel('real_qty')

        plt.savefig("D:/jimmy-ye/AI_supply_chain/data/forecast_holiday/compare" + str(int(id)) + '.jpg', dpi=600,
                    bbox_inches='tight')
        plt.close()


def main_function(old_path,new_path):
    data = read_forecast_data(old_path,new_path)
    final_data = merge_forecast_real(data)
    final_data = description_error(final_data)
    static_data = final_data.groupby(['sku_id'],as_index=False).mean
    print(static_data)
    plot_compare(final_data)





main_function('D:/jimmy-ye/AI_supply_chain/data/forecast/final.csv',
              'D:/jimmy-ye/AI_supply_chain/data/forecast_holiday/final_holiday_total.csv')
# data = pd.read_csv('D:/jimmy-ye/AI_supply_chain/data/forecast/data_total.csv',
#                        encoding="utf_8_sig")
# static_data = data.groupby(['sku_id'],as_index=False).mean()
# static_data['relative_error'] = static_data['relative_error'].apply(lambda x: abs(x))
# static_data = static_data.sort_values(by = 'relative_error',ascending=True)
# print(static_data)