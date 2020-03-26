# -*- coding: utf-8 -*-
# @Time    : 2019/7/25 14:22
# @Author  : Ye Jinyu__jimmy
# @File    : compare_diff_DC.py

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
import matplotlib.ticker as ticker
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
from datetime import datetime
from chinese_calendar import is_workday, is_holiday
# import chinese_calendar as calendar  #
import time
warnings.filterwarnings("ignore")


def print_in_log(string):
    print(string)
    date_1 = datetime.now()
    str_10 = datetime.strftime(date_1, '%Y-%m-%d')
    file = open('D:/jimmy-ye/AI_supply_chain/data/forecast_holiday/weather_forecast/' + 'log_compare_diff_DC' + str(str_10) + '.txt', 'a')
    file.write(str(string) + '\n')

#读取预测数据这里是csv的文件
def read_forecast_data(old_path):
    data_old = pd.read_csv(old_path,encoding='utf_8_sig',low_memory=False)
    return data_old

#读取真实的历史销售，这里是分配送中心和sku_id读取
#---------------------------------------------------------------->根据SKU 的id来获取每个SKU的具体的销售明细数据

def get_detail_sales_data(sku_id,DC_CODE,start_date,end_date):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    dsn = cx_Oracle.makedsn(host, port, sid)

    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码)
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", dsn)
    # 查看出货详细单的数据
    stkout_detail_sql = """SELECT ss.sender,s.name Dc_name      
                                ,wrh.GID,wrh.NAME warehouse_name
                                ,ss.num ,b.gdgid,G.NAME sku_name
                                ,trunc(c.time) OCRDATE,b.CRTOTAL,b.munit,b.qty,b.QTYSTR
                                ,b.TOTAL,b.price,b.qpc
                               FROM stkout ss ,stkoutdtl b, stkoutlog c,store s ,goods g,warehouseh wrh
                                where ss.num = b.num  and ss.num =c.num and b.gdgid=g.gid and ss.sender =s.gid
                                  and ss.cls='统配出' and ss.cls=c.cls and ss.cls=b.cls and ss.wrh = wrh.gid
                                     and b.GDGID = %s
                                   and c.stat IN ('700','720','740','320','340')
                                and c.time>=  to_date('%s','yyyy-mm-dd')
                                   and c.time <  to_date('%s','yyyy-mm-dd') 
                                  AND wrh.NAME LIKE'%%商品仓%%'  AND ss.SENDER= %s""" % (sku_id,start_date,end_date,DC_CODE)
    stkout_detail = pd.read_sql(stkout_detail_sql, conn)
    conn.close
    return stkout_detail

#--------------------------------------------------------------------------------->日的标准化转化
def date_normalize(data_frame):
    data_frame['Account_date'] = pd.to_datetime(data_frame['Account_date']).dt.normalize()
    return data_frame


#------------------------------------------------------------------------->设置函数按照预测的数据
def get_real_data(forecast_data):

    DC_list = set(forecast_data['Dc_code'])
    start_date = forecast_data['Account_date'].min()
    end_date = forecast_data['Account_date'].max()
    print('读取的截止日期和开始日期是：%s,%s，日期格式是%s'%(start_date,end_date,type(end_date)))
    final_data = pd.DataFrame()     #----------用于接受最后总的真实销售数据
    for DC in tqdm(DC_list):
        DC_data = pd.DataFrame()  #----------用于接受每个仓库单的的真实销售数据
        mid_forecast = forecast_data[forecast_data['Dc_code'] == DC]
        Dc_name = mid_forecast['Dc_name'].iloc[0]
        sku_id_list = set(mid_forecast['Sku_id'])
        for x in tqdm(sku_id_list):
            Sku_name = mid_forecast[mid_forecast['Sku_id']==x]['Sku_name'].iloc[0]
            print_in_log('正在读取' + str(DC)+str(Dc_name) + '的出库销量数据,sku是'+str(x)+str(Sku_name))
            real_sales = get_detail_sales_data(x,DC,start_date,end_date)
            real_sales.to_csv('D:/jimmy-ye/AI_supply_chain/data/'
                                        'forecast_holiday/weather_forecast/original_real_sales' + str(int(DC))+str(x) + '.csv', encoding="utf_8_sig")
            #--------------------------用于记录真实的销售数据中在查询的日期内，一共有多少的单销售，有多少的退单
            total_bill = int(len(real_sales))
            refund_bill = int(len((real_sales[real_sales['QTY']<0])))
            print_in_log('已经完成'+str(Dc_name)+'仓库的'+str(Sku_name)+'商品的读取，统计的时间区间是：'+str(start_date)+','
                         +str(end_date)+'总共的订单数是：'+str(total_bill)+'其中退单的数量是：'+str(refund_bill))
            real_sales = real_sales.rename(index=str, columns={'OCRDATE': 'Account_date', 'GDGID':
                                        'Sku_id','SENDER':'Dc_code'})
            real_sales = date_normalize(real_sales)
            real_sales = real_sales.groupby(["Account_date",'Sku_id','Dc_code'],as_index = False).agg(sum)
            real_sales.to_csv('D:/jimmy-ye/AI_supply_chain/data/'
                                        'forecast_holiday/weather_forecast/real_sales' + str(int(DC))+str(x) + '.csv', encoding="utf_8_sig")
            DC_data = DC_data.append(real_sales)
        final_data = final_data.append(DC_data)
    return final_data



#---------------------------------------------------------------------->将真实销售数据和预测值进行匹配
def merge_forecast_real(forecast_data):
    real_sales = get_real_data(forecast_data)
    real_sales.to_csv('D:/jimmy-ye/AI_supply_chain/data/'
                      'forecast_holiday/weather_forecast/real_sales_total.csv', encoding="utf_8_sig")
    forecast_data = date_normalize(forecast_data)
    if real_sales.empty == True:
        merge_data = forecast_data
        merge_data['QTY'] = 0.1
    else:
        merge_data = pd.merge(forecast_data,real_sales,on=['Sku_id','Account_date','Dc_code'],how='inner')
        merge_data = merge_data.fillna(0)
    return merge_data


#----------------------------------------->查看和比较真实和预测数据的真实误差，并查看误差的统计学指标，保存到csv
def description_error(data):
    data['Abs_error'] = data['Forecast_qty'] -data['QTY']
    data['Relative_error'] = data['Abs_error']/data['QTY']
    return data


#---------------------------------------------------------------------->对每个SKU进行真实和预测对比
def plot_compare(data):
    DC_code = set(data['Dc_code'])
    for DC in tqdm(DC_code):
        print(DC)
        dc_data = data[data['Dc_code'] ==DC]
        Dc_name = dc_data['Dc_name'].iloc[0]
        sku_id = set(dc_data['Sku_id'])
        start_date = data['Account_date'].min()
        end_date = data['Account_date'].max()
        for id in tqdm(sku_id):
            print_in_log('正在画图并记录的仓库和sku是:'+str(int(DC))+str(int(id)))
            mid_data = dc_data[dc_data['Sku_id'] == id]

            Sku_name = mid_data['Sku_name'].iloc[0]
            if mid_data.empty==True:            #-------------------确保程序运行，有可能有的DC没有SKU的预测信息
                pass
            else:
                total_error_describe = mid_data.describe()
                total_error_describe.to_csv('D:/jimmy-ye/AI_supply_chain/data/'
                                            'forecast_holiday/weather_forecast/'+str(int(DC))+'_'+str(Dc_name)
                                            +str(int(id))+'_'+str(Sku_name)+'.csv',encoding="utf_8_sig")
                mid_data = mid_data.sort_values(by = ['Account_date'],ascending=True)
                print(mid_data)
                date = mid_data['Account_date']
                forecast_qty = mid_data['Forecast_qty']
                real_qty = mid_data['QTY']
                fig = plt.figure(figsize=(20,10),facecolor='white')
                ax1 = fig.add_subplot(111)
                # 左轴
                ax1.bar(date, real_qty, width=0.5, align='center', label='real_qty', color="black")
                plt.legend(loc='upper left', fontsize=10)
                ax1.plot(date, forecast_qty, color='red', marker='o', linestyle='dashed', label='forecast_qty',
                         markersize=0.8)
                # tick_spacing = math.ceil(len(mid_data)/6)
                # ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
                # plt.xticks(pd.date_range(str(start_date), str(end_date)), rotation=45)
                plt.legend(loc='upper right', fontsize=10)
                ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))

                ax1.set_xlabel('date')
                ax1.set_ylabel('real_qty')

                plt.savefig("D:/jimmy-ye/AI_supply_chain/data/forecast_holiday/weather_forecast/compare" +
                            str(int(DC)) + '_' + str(Dc_name)
                            + str(int(id)) + str(Sku_name)+ 'weather.jpg', dpi=600,
                            bbox_inches='tight')
                plt.close()


def main_function(old_path):
    data = read_forecast_data(old_path)
    final_data = merge_forecast_real(data)
    final_data.to_csv('D:/jimmy-ye/AI_supply_chain/data/'
                                'forecast_holiday/weather_forecast/merge_data_weather.csv',
                                encoding="utf_8_sig")
    final_data = description_error(final_data)
    plot_compare(final_data)




#设置函数来看每个配送中心的MAE误差在某一个阈值的数量
def MAE_sort(data,threshold,name):
    mid_data = data[data['Dc_name']==name]
    mid_data = mid_data.replace(np.inf, 0)
    print(mid_data)
    new_data = mid_data.groupby(['Sku_id'],as_index=False).mean()
    new_data = new_data.sort_values(by = ['Relative_error'],ascending=True)
    print(new_data)
    new_data.to_csv('D:/jimmy-ye/AI_supply_chain/data/'
                    'forecast_holiday/weather_forecast/sort_relative_error_' +str(name)+'.csv',
    encoding = "utf_8_sig")
    return new_data

main_function('D:/jimmy-ye/AI_supply_chain/data/forecast_holiday/final_holiday_weather.csv')
# static_data = data.groupby(['sku_id'],as_index=False).mean()
# final_data = pd.read_csv('D:/jimmy-ye/AI_supply_chain/data/'
#                                 'forecast_holiday/merge_data_weather.csv',
#                                 encoding="utf_8_sig")
# final_data = description_error(final_data)
# mid_data = MAE_sort(final_data,0.2,'杭州配送中心')
# plot_compare(final_data)