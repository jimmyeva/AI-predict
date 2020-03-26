# -*- coding: utf-8 -*-
# @Time    : 2020/3/22 13:56
# @Author  : Ye Jinyu__jimmy
# @File    : test.py

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
import matplotlib as mpl
mpl.use('Agg')
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
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

wh_code     = '001'
name        = '金手指'
code        = '01100'
end_date    = '20200310'
predict_df  = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/compare_df金手指.csv',encoding='utf_8_sig')



#---------------------------------------------以下为画图的逻辑----------------------
# max_date                        = predict_df['Account_date'].max()
# min_date                        = predict_df['Account_date'].min()
#
# predict_df.sort_values(by = 'Account_date',axis=0,ascending=True)
# forecast_old_list               = predict_df['forecast_old'].tolist()
# forecast_list                   = predict_df['forecast'].tolist()
# sales_qty_list                  = predict_df['sales_qty'].tolist()
#
# df = pd.DataFrame({'forecast_old':forecast_old_list,
#                    'sales_qty':sales_qty_list,
#                     'forecast':forecast_list},
#                   index=pd.date_range(min_date,max_date,  freq='D'),)
#


df        = predict_df[['Account_date','forecast_old','sales_qty','forecast']]
print('df',df)
print(type(df['Account_date'][2]))
print(type(df['forecast_old'][2]))
print(type(df['forecast'][2]))
print(type(df['sales_qty'][3]))


df.sort_values(by = 'Account_date',axis=0,ascending=True)
font_size = 10  # 字体大小
fig_size  = (8, 6)  # 图表大小

names = (u'New_forecast', u'Old_forecast',u'Real')  # 姓名
subjects = df.index.values  # 科目
print(subjects)
date                            = df['Account_date']
predict_old                     = df['forecast_old']
real_qty                        = df['sales_qty']
predict                         = df['forecast']

#
# 更新字体大小
mpl.rcParams['font.size'] = font_size
# 更新图表大小
mpl.rcParams['figure.figsize'] = fig_size
# 设置柱形图宽度
bar_width = 0.3

index = np.arange(len(subjects))
# 绘制「new」的成绩
rects1 = plt.bar(index, predict, bar_width, color='#0072BC', label=names[0])
# 绘制「old」的成绩
rects2 = plt.bar(index + bar_width, predict_old, bar_width, color='#ED1C24', label=names[1])
# 绘制「real」的成绩
plt.plot(date, real_qty, color='black', marker='o', linestyle='dashed', label='real_qty',
         markersize=0.8)
# X轴标题
plt.xticks(index + bar_width, df['Account_date'],rotation=90)

# 图表标题
plt.title(u'forecast VS Real')
# 图例显示在图表下方
plt.legend(loc='upper left', fontsize=10, fancybox=True, ncol=5)


# 添加数据标签
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')
        # 柱形图边缘用白色填充，纯粹为了美观
        rect.set_edgecolor('white')

add_labels(rects1)
add_labels(rects2)

# 图表输出到本地
plt.savefig("D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/" +
            str(code) +
            '_' + str(name) +
            '.jpg',
            dpi=300,
            bbox_inches='tight')
