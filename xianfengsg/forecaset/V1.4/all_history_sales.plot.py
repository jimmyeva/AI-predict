# -*- coding: utf-8 -*-
# @Time    : 2019/8/9 9:24
# @Author  : Ye Jinyu__jimmy
# @File    : all_history_sales.plot.py
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


'''这个sdk是查看每个sku在机器学习过程中所有的销量值'''


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
                                  AND wrh.NAME LIKE'%%商品仓%%'  AND ss.SENDER= %s
                               """ % (sku_id,start_date,end_date,DC_CODE)
    stkout_detail = pd.read_sql(stkout_detail_sql, conn)
    conn.close
    return stkout_detail




#---------------------------------------------------------------------->对每个SKU进行真实和预测对比
def plot_compare(data,DC,id,sku_name,dc_name):
    data = data.sort_values(by = ['Account_date'],ascending=False )
    print('正在画图并记录的仓库和sku是:'+str(int(DC)),str(int(id)))

    if data.empty==True:            #-------------------确保程序运行，有可能有的DC没有SKU的预测信息
        pass
    else:
        date = data['Account_date']
        forecast_qty = data['Forecast_qty']
        real_qty = data['QTY']
        fig = plt.figure(figsize=(20,10),facecolor='white')
        ax1 = fig.add_subplot(111)
        # 左轴
        ax1.bar(date, real_qty, width=0.5, align='center', label='real_qty', color="black")
        plt.legend(loc='upper left', fontsize=10)
        ax1.plot(date, forecast_qty, color='red', marker='o', linestyle='dashed', label='forecast_qty',
                 markersize=0.8)
        plt.xticks(rotation=45)
        plt.legend(loc='upper right', fontsize=10)
        ax1.set_xlabel('date',rotation=45)
        ax1.set_ylabel('real_qty')

        plt.savefig("D:/jimmy-ye/AI/AI_supply_chain/data/XGBOOST/all_history_compare" +
                    str(int(DC)) + str(dc_name) +
                    '_' + str(int(id)) + str(sku_name)+ '.jpg', dpi=600,
                    bbox_inches='tight')
        plt.close()



#读取预测数据这里是csv的文件
def read_forecast_data(old_path):
    data_old = pd.read_csv(old_path,encoding='utf_8_sig',low_memory=False)
    data_old = data_old.rename(columns={'forecast_qty':'Forecast_qty'})
    return data_old


#--------------------------------------------------------------------------------->日的标准化转化
def date_normalize(data_frame):
    data_frame_sort = data_frame.sort_values(by = ['Account_date'],ascending=False )
    data_frame_sort['Account_date'] = pd.to_datetime(data_frame_sort['Account_date']).dt.normalize()
    return data_frame_sort

#------------------------------------------------------------------------->设置函数按照预测的数据,并完成画图
def get_real_data(forecast_data,start_date):

    DC_list = set(forecast_data['Dc_code'])
    end_date = forecast_data['Account_date'].max()
    print('读取的截止日期和开始日期是：%s,%s，日期格式是%s'%(start_date,end_date,type(end_date)))
    for DC in tqdm(DC_list):
        DC= math.ceil(DC)
        forecast_data_mid = forecast_data[forecast_data['Dc_code'] == DC]
        sku_id_list = set(forecast_data_mid['Sku_id'])
        dc_name = forecast_data_mid['Dc_name'].iloc[0]
        for x in tqdm(sku_id_list):
            x = math.ceil(x)
            print('正在读取' + str(DC) + '的统计学指标,sku是'+str(x))
            forecast_data_mid_sku = forecast_data_mid[forecast_data_mid['Sku_id'] == x ]
            sku_name = forecast_data_mid_sku['Sku_name'].iloc[0]
            new = re.findall(r'[\u4e00-\u9fa5]', sku_name)
            sku_name = ''.join(new)
            real_sales = get_detail_sales_data(x,DC,start_date,end_date)

            real_sales = real_sales.rename(index=str, columns={'OCRDATE': 'Account_date', 'GDGID':
                                        'Sku_id','SENDER':'Dc_code','DC_NAME':'Dc_name','SKU_NAME':'Sku_name'})
            real_sales = date_normalize(real_sales)
            if real_sales.empty == True:
                pass
            else:
                real_sales = real_sales.groupby(["Account_date", 'Sku_id', 'Dc_code','Dc_name'],
                                                as_index=False).agg(sum)
                mid_forecast = forecast_data[forecast_data['Dc_code'] == DC]
                mid_forecast = mid_forecast[mid_forecast['Sku_id'] == x]
                mid_forecast = date_normalize(mid_forecast)
                print('real_sales')
                print(real_sales)
                print('mid_forecast')
                print(mid_forecast)
                merge_data = pd.merge(real_sales, mid_forecast,
                                      on=['Sku_id', 'Account_date', 'Dc_code','Dc_name'], how='outer')
                merge_data['Sku_name'] = merge_data['Sku_name'].fillna(method='ffill')
                merge_data['Dc_name'] = merge_data['Dc_name'].fillna(method='ffill')
                # merge_data = merge_data.fillna(0)
                print('final_data',merge_data)
                merge_data.to_csv\
                    ('D:/jimmy-ye/AI/AI_supply_chain/data/XGBOOST/all_history_compare'
                     +str(int(DC)) + str(dc_name) +
                    '_' + str(int(x)) + str(sku_name)+'.csv',encoding='utf_8_sig')
                plot_compare(merge_data, DC, x, sku_name, dc_name)



def main_function(old_path,start_date):
    data = read_forecast_data(old_path)
    get_real_data(data,start_date)


start = '20181220'


main_function('D:/jimmy-ye/AI/AI_supply_chain/data/XGBOOST/result.csv',start)