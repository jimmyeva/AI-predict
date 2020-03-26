# -*- coding: utf-8 -*-
# @Time    : 2019/7/1 19:16
# @Author  : Ye Jinyu__jimmy
# @File    : analysis_sales.py

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
import math
import time
import multiprocessing
import re
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdate
#parser是根据字符串解析成datetime,字符串可以很随意，可以用时间日期的英文单词，可以用横线、逗号、空格等做分隔符。没指定时间默认是0点，没指定日期默认是今天，没指定年份默认是今年。
from dateutil.parser import parse
# from pylab import *
plt.switch_backend('agg')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
#如下是支持中文数字
# mpl.rcParams['font.sans-serif'] = ['SimHei']

#第一版本的设计
def draft_bar():
    sales_top = pd.read_csv('D:\jimmy-ye\AI_supply_chain\data.csv',encoding="utf_8_sig")
    good_id = sales_top['GDGID']
    good_id = set(good_id)
    print(good_id)
    for i in good_id:
        sales_top_Dataframe = sales_top[sales_top['GDGID'] == i ]
        sales_top_Dataframe = sales_top_Dataframe[sales_top_Dataframe['TOTAL']>0]
        sales_top_Dataframe_mid = sales_top_Dataframe.groupby(['OCRDATE'], as_index=False).agg(sum)
        print(sales_top_Dataframe_mid)
        x = sales_top_Dataframe_mid['OCRDATE']
        y1 = sales_top_Dataframe_mid['QTY']
        y2 = sales_top_Dataframe_mid['TOTAL']
        fig = plt.figure()

        ax1 = fig.add_subplot(111)
        ax1.plot(x, y1)
        ax1.set_ylabel('sales_qty')
        ax1.set_title("Double Y axis")

        ax2 = ax1.twinx()  # this is the important function
        ax2.plot(x, y2, 'r')
        # ax2.set_xlim([0, np.e])
        ax2.set_ylabel('TOTAL_price')
        # ax2.set_xlabel('Same X for both exp(-x) and ln(x)')
        plt.savefig("D:/jimmy-ye/AI_supply_chain/sales_total"+str(i) + '.jpg', dpi=400,
            bbox_inches='tight')
        plt.close()


#第二版本的设计，加入了门店的数量

#设置一个函数用来选择链接oracle拿到对应的某款SKU的售卖门店数量的时间序列,GROUPBY在sql里面运行的话会耗费好久的时间，每个SKU需要至少5-8分钟，因此把合并的操作放在
def sku_shops_count(gdgid):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    dsn = cx_Oracle.makedsn(host, port, sid)

    # scott是数据用户名，tiger是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", dsn)
    # 查看出货详细单的数据
    stkout_detail_sql = """SELECT b.OCRDATE,b.CLIENT,s.NUM  FROM STKOUTDTL s INNER JOIN(select *
                            from STKOUT s
                            INNER JOIN STORE s1  ON s.sender = s1.gid  
                            INNER JOIN STORE s2 ON s.CLIENT = s2.gid 
                            WHERE bitand(s1.property,32)=32 
                            AND bitand(s2.property,32)<>32 
                            AND substr(s2.AREA,2,3)<'8000' 
                            AND s.CLS='统配出')b ON s.NUM = b.NUM AND s.CLS='统配出' 
                            AND s.GDGID= %s AND s.TOTAL> 0 """ % (gdgid,)
    GDGID_Count = pd.read_sql(stkout_detail_sql, conn)
    # stkout_detail = pd.read_sql(stkout_detail_sql, conn)
    conn.close
    return GDGID_Count



def function_sales(sales_top,i):
    sales_top_Dataframe = sales_top[sales_top['GDGID'] == i]
    sales_top_Dataframe = sales_top_Dataframe[sales_top_Dataframe['TOTAL'] > 0]
    sales_top_Dataframe['OCRDATE'] = pd.to_datetime(sales_top_Dataframe['OCRDATE']).dt.normalize()
    sales_top_Dataframe_mid = sales_top_Dataframe.groupby(['OCRDATE'], as_index=False).agg(sum)
    print('sales_top_Dataframe_mid')
    GDGID_Count = sku_shops_count(i)
    # 这一步操作是因为存在同一个num同一个SKU，但是有不同批次的情况
    GDGID_Count = GDGID_Count.drop_duplicates()
    GDGID_Count['OCRDATE'] = pd.to_datetime(GDGID_Count['OCRDATE']).dt.normalize()
    GDGID_Count = GDGID_Count.groupby(["OCRDATE"], as_index=False)['CLIENT'].count()
    merge_sales_top = pd.merge(sales_top_Dataframe_mid, GDGID_Count, on=['OCRDATE'], how='outer')
    # 如下的计算是看商品的毛利率
    merge_sales_top['profit_rate'] = (merge_sales_top['RTOTAL'] - merge_sales_top['TOTAL']) / merge_sales_top['TOTAL']
    merge_sales_top['profit_rate'] = merge_sales_top['profit_rate'].map(lambda x: ('%.2f') % x)
    # 以下是看每天平均每家门店进货的数量
    merge_sales_top['per_shop_sales'] = merge_sales_top['QTY'] / merge_sales_top['CLIENT']
    merge_sales_top['per_shop_sales'] = merge_sales_top['per_shop_sales'].map(lambda x: ('%.2f') % x)
    merge_sales_top.to_csv('D:\jimmy-ye\AI_supply_chain\data_analysis\merge_sales_top' + str(i) + '.csv',
                           encoding="utf_8_sig")



sales_top = pd.read_csv('D:\jimmy-ye\AI_supply_chain\data.csv',encoding="utf_8_sig", low_memory=False)
good_id = sales_top['GDGID']
good_id = set(good_id)
print(good_id)
for i in good_id:
    print(i)
    function_sales(sales_top, i)
    merge_sales_top = pd.read_csv('D:\jimmy-ye\AI_supply_chain\data_analysis\merge_sales_top' + str(i) + '.csv',
                           encoding="utf_8_sig")
    #以下就是开始进行画图的操作
    date = merge_sales_top['OCRDATE']
    sales_qty = merge_sales_top['QTY']
    sales_profit_rate = merge_sales_top['profit_rate']
    print(sales_profit_rate)
    shop_counts = merge_sales_top['CLIENT']
    per_shop_sales = merge_sales_top['per_shop_sales']

    # 如下是进行画图的操作
    fig = plt.figure(figsize=(20, 10), facecolor='white')
    ax1 = fig.add_subplot(311)
    # 左轴
    ax1.bar(date, sales_qty, width=0.7, align='center', label='sales_qty', color="black")
    ax1.set_xlabel('date')
    ax1.set_ylabel('sales_qty')

    # 右轴
    ax2 = ax1.twinx()
    ax2.plot(date, sales_profit_rate, color='red', marker='o', linestyle='dashed', label='sales_profit_rate',
             markersize=0.8)

    plt.xticks(date, rotation=0)
    ax2.set_ylabel('sales_profit_rate')
    # ax2.set_ylim((merge_sales_top['profit_rate'].min(),merge_sales_top['profit_rate'].max()))

    # 将毛利率坐标轴以百分比格式显示
    def to_percent(temp, position):
        return '%2.1f' % (100 * temp) + '%'

    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.locator_params(nbins=8)
    plt.legend(loc='upper right', fontsize=10)
    ax1.legend(loc='upper left', fontsize=10)
    plt.grid(True, linestyle='-.')
    # 标题
    plt.title('sales_qty & sales_profit')

    fig.add_subplot(312)
    plt.bar(date, shop_counts, width=0.7, align='center', label='shop_counts', color="blue")
    plt.xticks(date, rotation=0)
    plt.ylabel('shop_counts')
    plt.locator_params(nbins=8)
    plt.legend(loc='upper right', fontsize=10)

    ax3 = fig.add_subplot(313)
    ax3.bar(date, per_shop_sales, width=0.7, align='center', label='per_shop_sales', color="green")
    plt.xticks(date, rotation=0)
    ax3.set_ylabel('per_shop_sales')
    plt.grid(True, linestyle='-.')
    plt.locator_params(nbins=8)
    plt.legend(loc='upper right', fontsize=10)
    plt.savefig("D:/jimmy-ye/AI_supply_chain/data_analysis/sales_total" + str(i) + '.jpg', dpi=600,
                bbox_inches='tight')
    plt.close()


