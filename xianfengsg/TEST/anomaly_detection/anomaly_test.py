# -*- coding: utf-8 -*-
# @Time    : 2019/11/26 14:21
# @Author  : Ye Jinyu__jimmy
# @File    : anomaly_test.py

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pymysql
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
import warnings
warnings.simplefilter("ignore")

def exponential_smoothing_3(alpha, s):
    '''
   三次指数平滑
   :param alpha:  平滑系数
   :param s:      数据序列， list
   :return:       返回三次指数平滑模型参数a, b, c， list
   '''
    s_single = ExponentialSmoothing(alpha, s)
    s_double = ExponentialSmoothing(alpha, s_single)
    s_triple = ExponentialSmoothing(alpha, s_double)

    a_triple = [0 for i in range(len(s))]
    b_triple = [0 for i in range(len(s))]
    c_triple = [0 for i in range(len(s))]
    for i in range(len(s)):
        a_triple[i] = 3 * s_single[i] - 3 * s_double[i] + s_triple[i]
        b_triple[i] = (alpha / (2 * ((1 - alpha) ** 2))) * (
                    (6 - 5 * alpha) * s_single[i] - 2 * ((5 - 4 * alpha) * s_double[i]) + (4 - 3 * alpha) * s_triple[i])
        c_triple[i] = ((alpha ** 2) / (2 * ((1 - alpha) ** 2))) * (s_single[i] - 2 * s_double[i] + s_triple[i])
    return a_triple, b_triple, c_triple



#--------------------------先读取销售数据
def get_sales():
    code_7th = '3019540'
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



sales = get_sales()
sales = sales.drop(sales[sales.Qty <= 0].index)
y = pd.Series(sales['Qty'].values)
date = pd.Series(sales['Ocrdate'].values)
seaonal = round(len(sales) / 4)
print('seaonal',seaonal)
ets3= ExponentialSmoothing(y, trend='add', seasonal='mul', seasonal_periods=seaonal)
r3 = ets3.fit()
data = pd.DataFrame({
    'Ocrdate': date,
    'Qty': y,
    'fitted': r3.fittedvalues,
    # 'pred': pred3
})
date = data['Ocrdate']
Qty = data['Qty']
Fitted = data['fitted']

fig = plt.figure(figsize=(20, 10), facecolor='white')
ax1 = fig.add_subplot(111)
# 左轴
ax1.bar(date, Qty, width=0.5, align='center', label='real_qty', color="black")
plt.legend(loc='upper left', fontsize=10)
ax1.plot(date, Fitted, color='red', marker='o', linestyle='dashed', label='forecast_qty',
         markersize=0.8)

plt.legend(loc='upper right', fontsize=10)
ax1.set_xlabel('date')
ax1.set_ylabel('real_qty')
plt.show()
plt.close()

# x3 = np.linspace(0, 4 * np.pi, 100)
# print(x3)
# y3 = pd.Series(20 + 0.1 * np.multiply(x3, x3) + 8 * np.cos(2 * x3) + 2 * np.random.randn(100))
# print(y3)
# ets3 = ExponentialSmoothing(y3, trend='add', seasonal='add', seasonal_periods=25)
# r3 = ets3.fit()
# pred3 = r3.predict(start=len(y3), end=len(y3) + len(y3)//2)
#
# data = pd.DataFrame({
#     'origin': y3,
#     'fitted': r3.fittedvalues,
#     # 'pred': pred3
# })
#
