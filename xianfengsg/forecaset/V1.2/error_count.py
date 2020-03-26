# -*- coding: utf-8 -*-
# @Time    : 2019/7/31 9:14
# @Author  : Ye Jinyu__jimmy
# @File    : error_count.py
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
def read_forecast_data(old_path):
    data_old = pd.read_csv(old_path,encoding='utf_8_sig',low_memory=False)
    return data_old

#读取sku的名称
#---------------------------------------------------------------->根据SKU 的id来获取每个SKU的具体的销售明细数据
def get_sku_name(sku_id):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    dsn = cx_Oracle.makedsn(host, port, sid)

    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码)
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", dsn)
    # 查看出货详细单的数据
    stkout_detail_sql = """SELECT g.NAME FROM GOODSH g WHERE g.gid = %s """ % (sku_id)
    sku_name = pd.read_sql(stkout_detail_sql, conn)
    conn.close
    return sku_name

#---------------------------------------------------------------->根据SKU 的id来获取每个SKU的具体的销售明细数据
def get_dc_name(dc_code):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    dsn = cx_Oracle.makedsn(host, port, sid)

    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码)
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", dsn)
    # 查看出货详细单的数据
    stkout_detail_sql = """SELECT s.NAME FROM STORE s where s.gid = %s """ % (dc_code)
    dc_name = pd.read_sql(stkout_detail_sql, conn)
    conn.close
    return dc_name


#--------------------------------------------------------------------------------->日的标准化转化
def date_normalize(data_frame):
    data_frame['Account_date'] = pd.to_datetime(data_frame['Account_date']).dt.normalize()
    return data_frame


#--------------------------------------------------------------------------------->原始日期的标准转化
def date_original_normalize(data_frame):
    data_frame['OCRDATE'] = pd.to_datetime(data_frame['OCRDATE']).dt.normalize()
    return data_frame


#------------------------------------------------------------------------------->以日期作为分组内容查看每天每个SKU的具体的销量
def data_group(data):
    #这里的毛利是门店卖出的总金额与仓库进货的总金额的差值比
    data['GROSS_PROFIT_RATE'] = (data['RTOTAL'] - data['TOTAL']) / data['TOTAL']
    #计算仓库销售的正确单价
    data['PRICE'] = data['PRICE']/ data['QTY']
    #以下是用来保存分组后的数据
    sales_data = pd.DataFrame(columns = ["Account_date","Sku_id",'Dc_name',"Sales_qty","Price",'Gross_profit_rate','Dc_code',
                                         'Wrh','Warehouse_name','Sku_name','Munit'])
    sales_data["Sales_qty"]=data.groupby(["OCRDATE"],as_index = False).sum()["QTY"]
    sales_data["Price"] = data.groupby(["OCRDATE"],as_index = False).mean()["PRICE"]
    sales_data["Gross_profit_rate"] = data.groupby(["OCRDATE"],as_index = False).mean()["GROSS_PROFIT_RATE"]
    sales_data["Account_date"]= data.groupby(['OCRDATE']).sum().index
    sales_data["Sku_id"] = [data["GDGID"].iloc[0]]*len(sales_data["Sales_qty"])
    sales_data["Dc_name"] = [data["DC_NAME"].iloc[0]] * len(sales_data["Sku_id"])
    sales_data["Dc_code"] = [data["SENDER"].iloc[0]] * len(sales_data["Sku_id"])
    sales_data["Munit"] = [data["MUNIT"].iloc[0]] * len(sales_data["Sales_qty"])
    sales_data["Wrh"] = [data["WRH"].iloc[0]] * len(sales_data["Sales_qty"])
    sales_data["Warehouse_name"] = [data["WAREHOUSE_NAME"].iloc[0]] * len(sales_data["Sales_qty"])
    sales_data["Sku_name"] = [data["SKU_NAME"].iloc[0]] * len(sales_data["Sales_qty"])
    sales_data = sales_data.sort_values( by = ['Account_date'], ascending = False)
    return sales_data

#------------------------------------------------------------------>根据SKU 的id来获取每个SKU的具体的销售明细数据
def get_detail_sales_data(sku_id,start_date,end_date,DC_CODE):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    dsn = cx_Oracle.makedsn(host, port, sid)

    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码)
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", dsn)
    # 查看出货详细单的数据
    stkout_detail_sql = """SELECT b.SENDER,s1.NAME AS Dc_name,S.WRH,w.NAME AS 
                                warehouse_name,g.NAME AS sku_name,s.GDGID,b.NUM,s.RTOTAL,b.OCRDATE,
                                s.CRTOTAL,s.MUNIT,s.QTY,s.QTYSTR,
                            s.TOTAL,s.PRICE,s.QPC FROM STKOUTDTL s INNER JOIN(
                            select *
                            from STKOUT s
                            INNER JOIN STORE s1 ON s.sender = s1.gid
                            INNER JOIN STORE s2 ON s.CLIENT = s2.gid
                            WHERE bitand(s1.property,32)=32 
                            AND bitand(s2.property,32)<>32 
                            AND substr(s2.AREA,2,3)<'8000' 
                            AND s.CLS='统配出')b ON s.NUM = b.NUM 
                            AND s.CLS='统配出' AND s.GDGID= %s
                            and b.OCRDATE >= to_date('%s','yyyy-mm-dd') 
                            and b.OCRDATE <= to_date('%s','yyyy-mm-dd') 
                            and b.sender = %s
                            INNER JOIN(SELECT * FROM goodsh G WHERE G.SORT<'8000' )g ON s.gdgid=g.gid
                            INNER JOIN(SELECT * FROM WAREHOUSE w WHERE w.NAME LIKE'%%商品仓%%' )w ON w.GID = S.WRH
                            INNER JOIN(SELECT * FROM STORE s1 )S1 ON S1.GID = b.SENDER """ % \
                        (sku_id,start_date,end_date,DC_CODE)
    stkout_detail = pd.read_sql(stkout_detail_sql, conn)
    conn.close
    return stkout_detail



#------------------------------------------------------------------------->设置函数按照预测的数据,并完成画图
def get_real_data(forecast_data,start_date):

    DC_list = set(forecast_data['Dc_code'])
    end_date = forecast_data['Account_date'].max()
    print('读取的截止日期和开始日期是：%s,%s，日期格式是%s'%(start_date,end_date,type(end_date)))
    final_data = pd.DataFrame()
    for DC in tqdm(DC_list):
        DC_data = pd.DataFrame()
        forecast_data_mid = forecast_data[forecast_data['Dc_code'] == DC]
        sku_id_list = set(forecast_data_mid['Sku_id'])
        dc_name = forecast_data_mid['Dc_name'].iloc[0]
        for x in tqdm(sku_id_list):
            print('正在读取' + str(DC) + '的统计学指标,sku是'+str(x))
            detail_sales = get_detail_sales_data(x,start_date,end_date,DC)
            forecast_data_mid_sku = forecast_data_mid[forecast_data_mid['Sku_id'] == x ]
            sku_name = forecast_data_mid_sku['Sku_name'].iloc[0]
            print(detail_sales)
            if detail_sales.empty ==True:
                print('该资源为空')
                pass
            else:
                detail_sales = date_original_normalize(detail_sales)
                sales_group = data_group(detail_sales)
                sales_shop_data = sales_group.reset_index(drop=True, inplace=False)
                sales_shop_data = date_normalize(sales_shop_data)
                forecast_data_mid_sku = date_normalize(forecast_data_mid_sku)
                print('sales_shop_data',sales_shop_data)
                print('forecast_data_mid_sku', forecast_data_mid_sku)
                DC_data = pd.merge(sales_shop_data,forecast_data_mid_sku,
                                   on=['Account_date','Sku_id','Dc_name',
                                    'Dc_code','Munit','Warehouse_name'
                                    ,'Sku_name','Wrh'],how='outer')

                # if os.path.exists('D:/jimmy-ye/AI_supply_chain/data/'
                #                             'forecast_holiday/total_compare' + str(int(DC)) + str(dc_name) +
                #                             '_' + str(int(x)) + str(sku_name) + '.csv'):
                DC_data.to_csv('D:/jimmy-ye/AI_supply_chain/data/'
                                 'forecast_holiday/total_compare' + str(int(DC)) + str(dc_name) +
                                 '_' + str(int(x)) + str(sku_name) + '.csv',encoding='utf_8_sig')

                mean_error = DC_data.fillna(method='ffill')
                # print('mean_error')
                # print(mean_error)
                DC_data = DC_data.append(mean_error)
                # else:
                #     pass
            # print(DC_data)
        final_data = final_data.append(DC_data)
    return final_data





def main_function(old_path,start_date):
    data = read_forecast_data(old_path)
    final_data= get_real_data(data,start_date)
    print(final_data)
    final_data.to_csv('D:/jimmy-ye/AI_supply_chain/data/'
                                'forecast_holiday/total_compare.csv',
                                encoding="utf_8_sig")




start = '20180101'


main_function('D:/jimmy-ye/AI_supply_chain/data/forecast_holiday/final_holiday.csv',start)
