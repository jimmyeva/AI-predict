# -*- coding: utf-8 -*-
# @Time    : 2019/7/4 8:58
# @Author  : Ye Jinyu__jimmy
# @File    : corresponding_analysis.py

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

import matplotlib as mpl

mpl.use('Agg')
from matplotlib import pyplot as plt

import math
import time
import multiprocessing
import re
from matplotlib.ticker import FuncFormatter,MultipleLocator, FormatStrFormatter,AutoMinorLocator
import matplotlib.dates as mdate
#parser是根据字符串解析成datetime,字符串可以很随意，可以用时间日期的英文单词，可以用横线、逗号、空格等做分隔符。没指定时间默认是0点，没指定日期默认是今天，没指定年份默认是今年。
from dateutil.parser import parse
# from pylab import *
plt.switch_backend('agg')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from statsmodels.graphics.mosaicplot import mosaic
import matplotlib as mpl
#如下是支持中文数字
# mpl.rcParams['font.sans-serif'] = ['SimHei']


# data = pd.read_csv('D:\jimmy-ye\AI_supply_chain\data_analysis\merge_sales_top' + str(i) + '.csv',
#                            encoding="utf_8_sig")



def analysis_function(i):
    data = pd.read_csv('D:\jimmy-ye\AI_supply_chain\data_analysis\merge_sales_top' + str(i) + '.csv',
                               encoding="utf_8_sig")
    describe_date = data[['QTY','TOTAL','PRICE','QPC','CLIENT','profit_rate','RTOTAL','CRTOTAL','per_shop_sales']]
    describe_date= describe_date.describe()
    describe_date.to_csv('D:\jimmy-ye\AI_supply_chain\data_analysis\corresponding_analysis\ddescribe_date'
                         +str(i)+'.csv',encoding="utf_8_sig")
    print(describe_date)
    #交叉列表的制作，查看某一个变量与标签的相关大小
    y = pd.crosstab(pd.qcut(data['CLIENT'],[0,0.25,0.5,0.75,1]),data['QTY'])

    #打印一个datafram，可以看看两两变量的相关性结果
    corresponding_date = describe_date.corr()
    corresponding_date.to_csv('D:\jimmy-ye\AI_supply_chain\data_analysis\corresponding_analysis\corresponding_date'
                         +str(i)+'.csv',encoding="utf_8_sig")
    print(data[u'CLIENT'].corr(data[u'QTY']))
    #制作变量的箱形图
    select_date = data[['QTY','TOTAL','PRICE','QPC','CLIENT','profit_rate','RTOTAL','CRTOTAL','per_shop_sales']]


    plt.figure(figsize=(10, 4))

    f = select_date.boxplot(sym='.',vert = True, whis = 1.5,
                   patch_artist = True,meanline = False,
                   showmeans = True,showbox = True,
                   showcaps = True,showfliers = True,
                    notch = False,return_type = 'dict')
    for box in f['boxes']:
        box.set( color='b', linewidth=1)        # 箱体边框颜色
        box.set( facecolor = 'b' ,alpha=0.5)    # 箱体内部填充颜色
    for whisker in f['whiskers']:
        whisker.set(color='k', linewidth=0.5,linestyle='-')
    for cap in f['caps']:
        cap.set(color='gray', linewidth=2)
    for median in f['medians']:
        median.set(color='DarkBlue', linewidth=2)
    for flier in f['fliers']:
        flier.set(marker='.', color='y', alpha=0.5)
    # boxes： 箱线
    # medians： 中位值的横线,
    # whiskers： 从box到error bar之间的竖线.
    # fliers： 异常值
    # caps： error bar横线
    # means： 均值的横线

    plt.title('boxplot'+str(i))
    plt.savefig('D:\jimmy-ye\AI_supply_chain\data_analysis\corresponding_analysis\cboxplot'
                         +str(i)+'.jpg', dpi=600,
                    bbox_inches='tight')
    plt.close()

    '''
    参数说明：
    sym:表示异常点的形状
    vert:是否垂直，箱线图是横向的(False)还是竖向的(True)
    whis: IQR，默认1.5，也可以设置区间比如[5,95]，代表强制上下边缘为数据95%和5%位置
    patch_artist:上下四分位框内是否填充，True为填充
    meanline:是否用线的形式表示均值，默认用点表示
    showmeans:是否显示均值，默认不显示
    showbox:是否显示箱线图的箱体
    showcaps:是否显示边缘线，箱线图顶端和末端的两条线默认显示
    showfliers:是否显示异常值
    notch:中间箱体是否缺口
    return_type:返回类型
    其它
    positions:指定箱线图的位置，默认为[0.1.2...]
    widths:指定箱线图的宽度，默认为0.5
'''



#门店数量与销售量相关性分析并保存
def corr_analysis_qty_client_save(path):
    sales_top = pd.read_csv(path,encoding="utf_8_sig", low_memory=False)
    print(sales_top)
    good_id = sales_top['GDGID']
    good_id = set(good_id)
    total_corr = pd.DataFrame(columns =['GDDID','CORR_COEFFIICIENT'])

    # 此函数是用来统计销售数量与售卖门店数的线性相关性
    def corr_qty_client(i):
        data = pd.read_csv('D:\jimmy-ye\AI_supply_chain\data_analysis\merge_sales_top' + str(i) + '.csv',
                           encoding="utf_8_sig")
        sku_id = i
        # print(data)
        corr_coefficient = data[u'CLIENT'].corr(data[u'QTY'])
        print(corr_coefficient)
        corr_data = pd.DataFrame(columns=['GDDID', 'CORR_COEFFIICIENT'])
        corr_data['GDDID'] = pd.Series(sku_id)
        corr_data['CORR_COEFFIICIENT'] = pd.Series(corr_coefficient)
        print(corr_data)
        return corr_data

    for i in good_id:
        print(i)
        corr_data = corr_qty_client(i)
        total_corr = total_corr.append(corr_data)
    print(total_corr)
    total_corr_description = total_corr.describe()
    print(total_corr_description)
    total_corr = total_corr.append(total_corr_description)
    print(total_corr)
    total_corr.to_csv('D:/jimmy-ye/AI_supply_chain/data_analysis/total_corr.csv',
                                   encoding="utf_8_sig")


# corr_analysis_save('D:\jimmy-ye\AI_supply_chain\data.csv')

#商品毛利率与销售量相关性分析并保存
def corr_analysis_qty_profit_save(path):
    sales_top = pd.read_csv(path,encoding="utf_8_sig", low_memory=False)
    print(sales_top)
    good_id = sales_top['GDGID']
    good_id = set(good_id)
    total_corr = pd.DataFrame(columns =['GDDID','CORR_COEFFIICIENT'])

    # 此函数是用来统计销售数量与售卖门店数的线性相关性
    def corr_qty_client(i):
        data = pd.read_csv('D:\jimmy-ye\AI_supply_chain\data_analysis\merge_sales_top' + str(i) + '.csv',
                           encoding="utf_8_sig")
        sku_id = i
        # print(data)
        corr_coefficient = data[u'profit_rate'].corr(data[u'QTY'])
        print(corr_coefficient)
        corr_data = pd.DataFrame(columns=['GDDID', 'CORR_COEFFIICIENT'])
        corr_data['GDDID'] = pd.Series(sku_id)
        corr_data['CORR_COEFFIICIENT'] = pd.Series(corr_coefficient)
        print(corr_data)
        return corr_data

    for i in good_id:
        print(i)
        corr_data = corr_qty_client(i)
        total_corr = total_corr.append(corr_data)
    print(total_corr)
    total_corr_description = total_corr.describe()
    print(total_corr_description)
    total_corr = total_corr.append(total_corr_description)
    print(total_corr)
    total_corr.to_csv('D:/jimmy-ye/AI_supply_chain/data_analysis/total_corr_profit_rate.csv',
                                   encoding="utf_8_sig")

corr_analysis_qty_profit_save('D:\jimmy-ye\AI_supply_chain\data.csv')