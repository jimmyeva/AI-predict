# -*- coding: utf-8 -*-
# @Time    : 2019/11/20 11:56
# @Author  : Ye Jinyu__jimmy
# @File    : sales_his_analysis


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
import tqdm
import pymysql
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdate
#parser是根据字符串解析成datetime,字符串可以很随意，可以用时间日期的英文单词，可以用横线、逗号、空格等做分隔符。没指定时间默认是0点，没指定日期默认是今天，没指定年份默认是今年。
from dateutil.parser import parse
from pylab import *
plt.switch_backend('agg')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
#如下是支持中文数字
mpl.rcParams['font.sans-serif'] = ['SimHei']
from statsmodels.graphics.mosaicplot import mosaic
'''这个算法程序是看每个sku的历史销售情况，查看每个画出每个sku的销售图,其中结合统计学的指标'''

#------------------------------设置函数用于读取每个sku的具体的销售数据
def get_sales(code_7th):
    # code_7th = '3000101'
    print('正在进行销量的读取')
    dbconn = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com", database="purchare_sys",
                             user="purchare_sys", password="purchare_sys@123", port=3306,
                             charset='utf8')
    get_sales_sql = """ SELECT sh.SENDER,sh.GDGID,sh.Sku_name,sh.Ocrdate,sh.Qty 
    FROM sales_his sh WHERE sh.GDGID='%s'"""%(code_7th)
    get_sales = pd.read_sql(get_sales_sql,dbconn)
    get_sales['GDGID'] = get_sales['GDGID'].astype(str)
    print('销量数据读取完成...')
    dbconn.close()
    return get_sales


def process_abnormal(data):
    mid_data = data
    # Q1 = mid_data['Qty'].quantile(q=0.25)
    # Q3 = mid_data['Qty'].quantile(q=0.75)
    # IQR = Q3 - Q1
    # mid_data["Qty"].iloc[np.where(mid_data["Qty"] > Q3 + 2 * IQR)] =  2 *np.median(mid_data['Qty'])
    # mid_data["Qty"].iloc[np.where(mid_data["Qty"] < Q1 - 2 * IQR)] = np.median(mid_data['Qty'])
    # Get the 98th and 2nd percentile as the limits of our outliers

    upper_limit = np.percentile(mid_data['Qty'].values, 98)
    lower_limit = np.percentile(mid_data['Qty'].values, 2)
    # Filter the outliers from the dataframe
    data['Qty'].loc[data['Qty'] > upper_limit] = upper_limit
    data['Qty'].loc[data['Qty'] < lower_limit] = lower_limit

    return mid_data


#----------------------------如下函数是获取叫货目录的7位sku
def get_order_sku():
    print('正在进行叫货的读取操作')
    dbconn = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com", database="purchare_sys",
                             user="purchare_sys", password="purchare_sys@123", port=3306,
                             charset='utf8')
    get_orders_sql = """ SELECT pcr.goods_id Sku_id,pcr.goods_code GOODS_CODE,pcr.goods_name FROM p_call_record pcr 
        WHERE pcr.warehouse_id ='1000255' GROUP BY pcr.goods_id """
    get_orders = pd.read_sql(get_orders_sql,dbconn)
    get_orders['Sku_id'] = get_orders['Sku_id'].astype(str)
    print('叫货目录读取完成')
    dbconn.close()
    return get_orders

#-------------------------设置函数用于判断是好的SKU还是差的，返回1/0，1代表好，0代表差
def compare_sku():
    get_good_sku = pd.read_csv('D:/AI/xianfengsg/TEST/statistics_analysis/good_sku.csv', encoding='utf_8_sig')
    get_all_sku  = pd.read_csv('D:/AI/xianfengsg/TEST/statistics_analysis/result.csv',encoding='utf_8_sig')
    get_good_sku.rename(columns={'name': 'label'}, inplace=True)
    get_good_sku['label'] = 1
    all_sku = get_all_sku[['Sku_id']]
    # 求差集
    all_sku = all_sku.append(get_good_sku)
    all_sku = all_sku.append(get_good_sku)
    bad_sku = all_sku.drop_duplicates(subset=['Sku_id'], keep=False)
    bad_sku['label'] = 0
    data_total = pd.concat([bad_sku, get_good_sku])
    data_total.reset_index(drop=True, inplace=True)
    return data_total




#----------------------------对获取的销售数据进行画图并保存
def plot_compare(data,DC,id,sku_name,result):
    # DC = '浙北公司'
    # id = '3000101'
    # sku_name = '百香果（盒）'
    # data = get_sales(id)
    data_describe = data.describe()
    data_describe_T = pd.DataFrame(data_describe.values.T, index=data_describe.columns, columns=data_describe.index)
    data_describe_T['indicator'] = data_describe_T['std'] *(data_describe_T['count'] ** 0.5)
    # result = pd.DataFrame(data.values.T, index=data.columns, columns=data.index)
    columns = data_describe_T.columns.values

    # index = data_describe.index.values

    #------------设置阈值，将sku进行切割
    # min = 2802
    # max = 24602
    # indicator = data['indicator'].iloc[0]
    # if indicator > max:
    #     result = 0
    # elif indicator < min:
    #     result = 0
    # else:
    #     result = 1

    text = str(columns[0]) + ':' + str(data_describe_T['count'].iloc[0]) + \
           '\n' +str(columns[1]) + ':' + str(data_describe_T['mean'].iloc[0]) + '\n' + \
           str(columns[2]) + ':' + str(data_describe_T['std'].iloc[0]) + '\n' \
           +str(columns[3]) + ':' + str(data_describe_T['min'].iloc[0]) + '\n' \
            +str(columns[8]) + ':' + str(data_describe_T['indicator'].iloc[0]) + '\n' \
            +'result:' + str(result) + '\n'


    print('正在画图并记录的仓库和sku是:'+str(DC),str(int(id)))
    print(data)
    if data.empty==True:                 #-------------------确保程序运行，有可能有的DC没有SKU的预测信息
        pass
    else:
        date = data['Ocrdate']
        real_qty = data['Qty']
        sum = data['Qty'].max()

        text = text
        fig = plt.figure(figsize=(20,10),facecolor='white')
        ax1 = fig.add_subplot(111)
        # 左轴
        ax1.bar(date, real_qty, width=0.5, align='center', label='real_qty', color="black")
        # ax1.plot(date, real_qty, color='red', marker='o', linestyle='dashed', label='forecast_qty',
        #          markersize=0.8)
        plt.legend(loc='upper left', fontsize=10)
        plt.text('2019-10-01',sum,text,fontdict={'size': 20, 'color': 'y'},verticalalignment ='top',horizontalalignment='left')
        ax1.set_xlabel('date')
        ax1.set_ylabel('real_qty')

        plt.savefig("D:/AI/xianfengsg/TEST/statistics_analysis/" +
                    str(DC) +
                    '_' + str(int(id)) + str(sku_name) + '.jpg', dpi=600,
                    bbox_inches='tight')
        plt.close()


#------------------------------------设置主函数用于循环运行历史销售数据的信息

data_total = compare_sku()
sku_list = data_total['Sku_id'].to_list()
result_list = data_total['label'].to_list()
for i in range(len(sku_list)):
    sku  = sku_list[i]
    result = result_list[i]
    sales = get_sales(sku)
    if sales.empty ==  True:
        pass
    else:
        sales = process_abnormal(sales)
        sku_name = sales['Sku_name'].iloc[0]
        plot_compare(sales,'浙北',sku,sku_name,result)
# print(data_total)









