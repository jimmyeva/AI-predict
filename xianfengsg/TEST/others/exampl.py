# -*- coding: utf-8 -*-
# @Time    : 2019/7/3 18:18
# @Author  : Ye Jinyu__jimmy
# @File    : exampl.py

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
from tqdm import *
import datetime
import math
import time
import multiprocessing
import re
import matplotlib
from matplotlib.ticker import FuncFormatter,MultipleLocator, FormatStrFormatter,AutoMinorLocator
import matplotlib.dates as mdate
from itertools import combinations
#parser是根据字符串解析成datetime,字符串可以很随意，可以用时间日期的英文单词，可以用横线、逗号、空格等做分隔符。没指定时间默认是0点，没指定日期默认是今天，没指定年份默认是今年。
from dateutil.parser import parse
# from pylab import *
plt.switch_backend('agg')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
'''如下是支持中文数字'''
# mpl.rcParams['font.sans-serif'] = ['SimHei']
import re

# news  = 'mini雪糕组合装1*5支'
# new = re.findall(r'[\u4e00-\u9fa5]',news)
# test  = ''.join(new)
# print(new)
# print(test)
#

def plot_compare(data,DC,id):
    data = data.sort_values(by = ['Account_date'],ascending=True )
    print('正在画图并记录的仓库和sku是:'+str(int(DC)),str(int(id)))

    if data.empty==True:            #-------------------确保程序运行，有可能有的DC没有SKU的预测信息
        pass
    else:

        # matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
        # # 构建数据
        # x = data['Account_date']
        # y = data['Forecast_qty']
        # y1= data['QTY']
        # # 绘图
        # plt.bar(x=x, height=y, label='书库大全', color='steelblue', alpha=0.8)
        # # 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
        # for x1, yy in zip(x, y):
        #     plt.text(x1, yy + 1, str(yy), ha='center', va='bottom', fontsize=20, rotation=0)
        # # 设置标题
        # plt.title("每日销量"+str(sku_name)+'_'+str(dc_name))
        # # 为两条坐标轴设置名称
        # plt.xlabel("出库日期")
        # plt.ylabel("出库数量")
        # # 显示图例
        # plt.legend()
        # # 画折线图
        # plt.plot(x, y, "r", marker='*', ms=10, label="a")
        # plt.xticks(rotation=45)
        # plt.legend(loc="upper left")
        # plt.savefig("a.jpg")
        # plt.show()


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

        plt.savefig("D:/jimmy-ye/AI_supply_chain/data/forecast_holiday/all_history_compare" +
                 '.jpg', dpi=600,
                    bbox_inches='tight')

        plt.close()


# data = pd.read_csv('D:/jimmy-ye/AI_supply_chain/data/forecast_holiday/weather_forecast/all_history_compare1000255杭州配送中心_3016720水晶樱桃.csv',encoding= 'utf_8_sig')
# plot_compare(data,1000255,3012624)




'''测试全排列'''

# a = [1,'Dave',3.14,["Mark",7,9,[100,101]],10]
# def foo(arr):
#     lit = ''
#     for i in arr:
#         if isinstance(i,list):
#             lit += foo(i)
#         else:
#             lit += str(i) + "-"
#     return  lit
# str1 = foo(a)
# print (str1[0:-1].split('-'))


'''毛利算法测试'''
# raw_data = pd.read_excel('D:/AI/xianfengsg/other_project/product_profit/毛利测算测试数据0902美团先试下.xlsx', encoding='utf_8_sig')
#
# raw_data = raw_data.rename(index=str, columns={'Unnamed: 0': 'name', 'Unnamed: 1': 'code', 'Unnamed: 2': 'sales_price',
#                                                'Unnamed: 3': 'profit',
#                                                'Unnamed: 4': 'discount_price_rate', 'Unnamed: 5': 'discount',
#                                                'Unnamed: 6': 'threshold',
#                                                'Unnamed: 7': 'participate', 'Unnamed: 8': 'seckill',
#                                                'Unnamed: 9': 'second_kill',
#                                                '1代表满减，2代表折扣': 'customer', '商家券选最高折扣': 'store_discount',
#                                                'Unnamed: 12': 'store_threshold',
#                                                'Unnamed: 13': 'discont_rate', 'Unnamed: 14': 'channel'})
# raw_data.reset_index(drop=True)
# raw_data.drop(raw_data.index[0], inplace=True)
# print(raw_data.index)
# raw_data.reset_index(drop=True)
# print(raw_data)
# raw_data['price_after_discount'] = raw_data['sales_price'] * raw_data['discount_price_rate']
# raw_data['profit_after'] = raw_data['profit'] * raw_data['discount_price_rate']
# raw_data['cost'] = raw_data['sales_price'] - raw_data['profit']
# raw_data['origin_profit_rate'] = raw_data['profit_after'] / raw_data['cost']
#
#
# seckill_data = raw_data[raw_data['seckill'] == 'Y']
# seckill_name = seckill_data['name'].tolist()
# seckill_price_after_discount = seckill_data['price_after_discount'].tolist()
# seckill_profit_after = seckill_data['profit_after'].tolist()
# seckill_price = seckill_data['sales_price'].tolist()
#
# # 第二步是特价商品的集合
# second_kill = raw_data[raw_data['second_kill'] < 100]
# def second_set(second_name):
#     new_list = []
#     for i in range(0,len(second_kill)):
#         for k in range(0,second_num[i]):
#             new_list.append(second_name[i])
#     return new_list
#
# second_kill.reset_index()
# second_name_mid = second_kill['name'].tolist()
# second_price_after_discount_mid = second_kill['price_after_discount'].tolist()
# second_profit_after_mid = second_kill['profit_after'].tolist()
# second_price_mid = second_kill['sales_price'].tolist()
# second_num = second_kill['second_kill'].tolist()
#
# second_name = second_set(second_name_mid)
# second_price_after_discount = second_set(second_price_after_discount_mid)
# second_profit_after = second_set(second_profit_after_mid)
# second_price = second_set(second_price_mid)
#
#
# price = [c for c in  combinations(second_price_after_discount, 2)]
# def sum_list(x):
#     result = 0
#     for i in range(len(x)):
#         result += x[i]
#     return result
# x = price[2]
# total_price = sum_list(x)
# print(total_price)

# if second_kill_total > max_price:
#     print('全部会在特价商品中进行选择')
#     max_price = max_price - np.sum(seckill_price_after_discount)
#     # 记录一个包含了前两种商品的name列表
#     first_second_name = seckill_name
#     # first_second_name.extend(second_name)
#
#     first_second_price = seckill_price
#     # first_second_price.extend(second_price)
#
#     first_second_all_profit = seckill_profit_after
#     # first_second_all_profit.extend(second_profit_after)
#     # 记录下前秒杀商品和特价商品总价和总毛利
#     first_value = np.sum(seckill_price_after_discount)
#     # two_profit_total = np.sum(second_profit_after) + seckill_profit_after_total
#     first_profit_total = seckill_profit_after_total
#     # 所有特价商品的毛利总价
#     second_profit_after_total = np.sum(second_profit_after)
#     for x in range(len(second_name)):
#         if max_price < second_price_after_discount[x]:
#             print('1st')
#             list_01 = [str(second_name[x])]
#             mid_list_name = first_second_name + list_01
#             all_permutation.append(mid_list_name)
#
#             list_price = [int(second_price[x])]
#             mid_list_price = first_second_price + list_price
#             all_sales_price.append(mid_list_price)
#
#             list_02 = [int(second_profit_after[x])]
#             mid_list_all_profit = first_second_all_profit + list_02
#             all_profit.append(mid_list_all_profit)
#
#             all_value.append(first_value + second_price_after_discount[x])
#             all_profit_total.append(first_profit_total + second_profit_after[x])
#         else:
#             two_max_price = max_price - second_price_after_discount[x]
#             for y in range(len(second_name)):
#                 if two_max_price < second_price_after_discount[y]:
#                     print('2nd')
#                     list_01 = [str(second_name[x]), str(second_name[y])]
#                     mid_list_name = first_second_name + list_01
#                     all_permutation.append(mid_list_name)
#
#                     list_price = [int(second_price[x]), int(second_price[y])]
#                     mid_list_price = first_second_price + list_price
#                     all_sales_price.append(mid_list_price)
#
#                     list_02 = [int(second_profit_after[x]), int(second_profit_after[y])]
#                     mid_list_all_profit = first_second_all_profit + list_02
#                     all_profit.append(mid_list_all_profit)
#
#                     all_value.append(first_value + second_price_after_discount[x] + second_price_after_discount[y])
#                     all_profit_total.append(first_profit_total + second_profit_after[x] + second_profit_after[y])
#                 else:
#                     three_max_price = max_price - second_price_after_discount[x] - second_price_after_discount[y]
#                     for z in range(len(second_name)):
#                         if three_max_price < second_price_after_discount[z]:
#                             list_01 = [str(second_name[x]), str(second_name[y]), str(second_name[z])]
#                             mid_list_name = first_second_name + list_01
#                             all_permutation.append(mid_list_name)
#
#                             list_price = [int(second_price[x]), int(second_price[y]), int(second_price[z])]
#                             mid_list_price = first_second_price + list_price
#                             all_sales_price.append(mid_list_price)
#
#                             # c = zip(a, b)
#                             # list_new = [row[i] for i in range(len(0)) for row in c]
#                             list_02 = [int(second_profit_after[x]), int(second_profit_after[y]),
#                                        int(second_profit_after[z])]
#                             mid_list_all_profit = first_second_all_profit + list_02
#                             all_profit.append(mid_list_all_profit)
#
#                             all_value.append(
#                                 first_value + second_price_after_discount[x] +
#                                 second_price_after_discount[y] + second_price_after_discount[z])
#                             all_profit_total.append(
#                                 first_profit_total + second_profit_after[x] + second_profit_after[y] +
#                                 second_profit_after[z])
#                         else:
#                             four_max_price = max_price - second_price_after_discount[x] - \
#                                              second_price_after_discount[y] - second_price_after_discount[z]
#                             for w in range(len(second_name)):
#                                 if four_max_price < second_price_after_discount[w]:
#                                     list_01 = [str(second_name[x]), str(second_name[y]),
#                                                str(second_name[z]), str(second_name[w])]
#                                     mid_list_name = first_second_name + list_01
#                                     all_permutation.append(mid_list_name)
#
#                                     list_price = [int(second_price[x]), int(second_price[y]),
#                                                   int(second_price[z]), int(second_price[w])]
#                                     mid_list_price = first_second_price + list_price
#                                     all_sales_price.append(mid_list_price)
#
#                                     list_02 = [int(second_profit_after[x]), int(second_profit_after[y]),
#                                                int(second_profit_after[z]), int(second_profit_after[w])]
#                                     mid_list_all_profit = first_second_all_profit + list_02
#                                     all_profit.append(mid_list_all_profit)
#
#                                     all_value.append(
#                                         first_value + second_price_after_discount[x] +
#                                         second_price_after_discount[y] +
#                                         second_price_after_discount[z] + second_price_after_discount[w])
#                                     all_profit_total.append(first_profit_total + second_profit_after[x] +
#                                                             second_profit_after[z] + second_profit_after[y] +
#                                                             second_profit_after[w])
#                                 else:
#                                     five_max_price = max_price - second_price_after_discount[x] - \
#                                                      second_price_after_discount[y] - second_price_after_discount[z] \
#                                                      - second_price_after_discount[w]
#                                     for g in range(len(second_name)):
#                                         if five_max_price < second_price_after_discount[g]:
#                                             list_01 = [str(second_name[x]), str(second_name[y]),
#                                                        str(second_name[z]), str(second_name[w]),
#                                                        str(second_name[g])]
#                                             mid_list_name = first_second_name + list_01
#                                             all_permutation.append(mid_list_name)
#
#                                             list_price = [int(second_price[x]), int(second_price[y]),
#                                                           int(second_price[z]), int(second_price[w]),
#                                                           int(second_price[g])]
#                                             mid_list_price = first_second_price + list_price
#                                             all_sales_price.append(mid_list_price)
#
#                                             list_02 = [int(second_profit_after[x]), int(second_profit_after[y]),
#                                                        int(second_profit_after[z]), int(second_profit_after[w]),
#                                                        int(second_profit_after[g])]
#                                             mid_list_all_profit = first_second_all_profit + list_02
#                                             all_profit.append(mid_list_all_profit)
#
#                                             all_value.append(
#                                                 first_value + second_price_after_discount[x] +
#                                                 second_price_after_discount[y] +
#                                                 second_price_after_discount[z] + second_price_after_discount[w] +
#                                                 second_price_after_discount[g])
#                                             all_profit_total.append(first_profit_total + second_profit_after[x] +
#                                                                     second_profit_after[z] + second_profit_after[
#                                                                         y] +
#                                                                     second_profit_after[w] + second_profit_after[g])
#     # for i in range(len(second_name)):
#     #     second_kill_total = np.sum(second_price_after_discount[0:i])
#     #     second_profit_total = np.sum(second_profit_after[0:i]) + seckill_profit_after_total
#     #     if max_price *0.9 < second_kill_total:
#     #         first_list = seckill_data['name'].tolist()
#     #         first_list.extend(second_name[0:i])
#     #         all_permutation.append(first_list)
#     #         first_list_profit = seckill_profit_after
#     #         first_list_profit.extend(second_profit_after[0:i])
#     #         all_profit.append(first_list_profit)
#     #
#     #         first_price = seckill_data['sales_price'].tolist()
#     #         first_price.extend(seckill_price[0:i])
#     #         all_sales_price.append(first_price)
#     #
#     #         all_value.append(second_kill_total)
#     #         all_profit_total.append(second_profit_total)
#     #         for k in range(len(normal_name)):
#     #             first_list.append(str(normal_name[k]))
#     #             all_permutation.append(first_list)
#     #
#     #             first_price.append(int(normal_price[k]))
#     #             all_sales_price.append(first_price)
#     #             first_list_profit.append(int(normal_profit_after[k]))
#     #             all_profit.append(first_list_profit)
#     #             all_value.append(second_kill_total+normal_price_after_discount[k])
#     #             all_profit_total.append(normal_profit_after[k])
#     #     else:
#
# else:
#     max_price = max_price - second_kill_total
#     # 记录一个包含了前两种商品的name列表
#     first_second_name = seckill_name
#     first_second_name.extend(second_name)
#
#     first_second_price = seckill_price
#     first_second_price.extend(second_price)
#
#     first_second_all_profit = seckill_profit_after
#     first_second_all_profit.extend(second_profit_after)
#     # 记录下前秒杀商品和特价商品总价和总毛利
#     two_value = second_kill_total
#     two_profit_total = np.sum(second_profit_after) + seckill_profit_after_total
#     # 所有特价商品的毛利总价
#     second_profit_after_total = np.sum(second_profit_after)
#     for x in range(len(normal_name)):
#         if max_price < normal_price_after_discount[x]:
#             print('1st')
#             list_01 = [str(normal_name[x])]
#             mid_list_name = first_second_name + list_01
#             all_permutation.append(mid_list_name)
#
#             list_price = [int(normal_price[x])]
#             mid_list_price = first_second_price + list_price
#             all_sales_price.append(mid_list_price)
#
#             list_02 = [int(normal_profit_after[x])]
#             mid_list_all_profit = first_second_all_profit + list_02
#             all_profit.append(mid_list_all_profit)
#
#             all_value.append(two_value + normal_price_after_discount[x])
#             all_profit_total.append(two_profit_total + normal_profit_after[x])
#         else:
#             two_max_price = max_price - normal_price_after_discount[x]
#             for y in range(len(normal_name)):
#                 if two_max_price < normal_price_after_discount[y]:
#                     print('2nd')
#                     list_01 = [str(normal_name[x]), str(normal_name[y])]
#                     mid_list_name = first_second_name + list_01
#                     all_permutation.append(mid_list_name)
#
#                     list_price = [int(normal_price[x]), int(normal_price[y])]
#                     mid_list_price = first_second_price + list_price
#                     all_sales_price.append(mid_list_price)
#
#                     list_02 = [int(normal_profit_after[x]), int(normal_profit_after[y])]
#                     mid_list_all_profit = first_second_all_profit + list_02
#                     all_profit.append(mid_list_all_profit)
#
#                     all_value.append(two_value + normal_price_after_discount[x] + normal_price_after_discount[y])
#                     all_profit_total.append(two_profit_total + normal_profit_after[x] + normal_profit_after[y])
#                 else:
#                     three_max_price = max_price - normal_price_after_discount[x] - normal_price_after_discount[y]
#                     for z in range(len(normal_name)):
#                         if three_max_price < normal_price_after_discount[z]:
#                             list_01 = [str(normal_name[x]), str(normal_name[y]), str(normal_name[z])]
#                             mid_list_name = first_second_name + list_01
#                             all_permutation.append(mid_list_name)
#
#                             list_price = [int(normal_price[x]), int(normal_price[y]), int(normal_price[z])]
#                             mid_list_price = first_second_price + list_price
#                             all_sales_price.append(mid_list_price)
#
#                             # c = zip(a, b)
#                             # list_new = [row[i] for i in range(len(0)) for row in c]
#                             list_02 = [int(normal_profit_after[x]), int(normal_profit_after[y]),
#                                        int(normal_profit_after[z])]
#                             mid_list_all_profit = first_second_all_profit + list_02
#                             all_profit.append(mid_list_all_profit)
#
#                             all_value.append(
#                                 two_value + normal_price_after_discount[x] +
#                                 normal_price_after_discount[y] + normal_price_after_discount[z])
#                             all_profit_total.append(two_profit_total + normal_profit_after[x] + normal_profit_after[y] +
#                                                     normal_profit_after[z])
#                         else:
#                             four_max_price = max_price - normal_price_after_discount[x] - \
#                                              normal_price_after_discount[y] - normal_price_after_discount[z]
#                             for w in range(len(normal_name)):
#                                 if four_max_price < normal_price_after_discount[w]:
#                                     list_01 = [str(normal_name[x]), str(normal_name[y]),
#                                                str(normal_name[z]), str(normal_name[w])]
#                                     mid_list_name = first_second_name + list_01
#                                     all_permutation.append(mid_list_name)
#
#                                     list_price = [int(normal_price[x]), int(normal_price[y]),
#                                                   int(normal_price[z]), int(normal_price[w])]
#                                     mid_list_price = first_second_price + list_price
#                                     all_sales_price.append(mid_list_price)
#
#                                     list_02 = [int(normal_profit_after[x]), int(normal_profit_after[y]),
#                                                int(normal_profit_after[z]), int(normal_profit_after[w])]
#                                     mid_list_all_profit = first_second_all_profit + list_02
#                                     all_profit.append(mid_list_all_profit)
#
#                                     all_value.append(
#                                         two_value + normal_price_after_discount[x] +
#                                         normal_price_after_discount[y] +
#                                         normal_price_after_discount[z] + normal_price_after_discount[w])
#                                     all_profit_total.append(two_profit_total + normal_profit_after[x] +
#                                                             normal_profit_after[z] + normal_profit_after[y] +
#                                                             normal_profit_after[w])
#                                 else:
#                                     five_max_price = max_price - normal_price_after_discount[x] - \
#                                                      normal_price_after_discount[y] - normal_price_after_discount[z] \
#                                                      - normal_price_after_discount[w]
#                                     for g in range(len(normal_name)):
#                                         if five_max_price < normal_price_after_discount[g]:
#                                             list_01 = [str(normal_name[x]), str(normal_name[y]),
#                                                        str(normal_name[z]), str(normal_name[w]), str(normal_name[g])]
#                                             mid_list_name = first_second_name + list_01
#                                             all_permutation.append(mid_list_name)
#
#                                             list_price = [int(normal_price[x]), int(normal_price[y]),
#                                                           int(normal_price[z]), int(normal_price[w]),
#                                                           int(normal_price[g])]
#                                             mid_list_price = first_second_price + list_price
#                                             all_sales_price.append(mid_list_price)
#
#                                             list_02 = [int(normal_profit_after[x]), int(normal_profit_after[y]),
#                                                        int(normal_profit_after[z]), int(normal_profit_after[w]),
#                                                        int(normal_profit_after[g])]
#                                             mid_list_all_profit = first_second_all_profit + list_02
#                                             all_profit.append(mid_list_all_profit)
#
#                                             all_value.append(
#                                                 two_value + normal_price_after_discount[x] +
#                                                 normal_price_after_discount[y] +
#                                                 normal_price_after_discount[z] + normal_price_after_discount[w] +
#                                                 normal_price_after_discount[g])
#                                             all_profit_total.append(two_profit_total + normal_profit_after[x] +
#                                                                     normal_profit_after[z] + normal_profit_after[y] +
#                                                                     normal_profit_after[w] + normal_profit_after[g])
#
#
#






# 最后一部是所有不含秒杀商品的其他所有商品
# normal_mid_data = raw_data[raw_data['seckill'] == 'N']
# normal_data = normal_mid_data[normal_mid_data['second_kill'] == 'N']
# normal_name = normal_data['name'].tolist()
# normal_price_after_discount = normal_data['price_after_discount'].tolist()
# normal_profit_after = normal_data['profit_after'].tolist()
# normal_data_back_second = normal_mid_data[normal_mid_data['second_kill'] == 'Y']
# normal_name.extend(normal_data_back_second['name'].tolist())
# normal_price_after_discount.extend(normal_data_back_second['sales_price'].tolist())
# normal_profit_after.extend(normal_data_back_second['profit'].tolist())
# normal_price = normal_mid_data['sales_price'].tolist()
#
# max_price = raw_data['threshold'].iloc[0]
# discount = raw_data['discount'].iloc[0]
# mid_max_price = max_price * 1.1
# seckill_total = np.sum(seckill_price_after_discount)
# x = raw_data['customer'].iloc[0]
#



'''测试激活函数，做非线性转换'''
# list = [1,1,1,1]
# if 1 in list and len(set(list))==1:
#     print('1')
# else:
#     print('-1')
# x = -1
# def sigmoid(x):
#     s = 1 / (1 + np.exp(-x))
#     print(s)
#     # return s
# sigmoid(x)

'''制作流转时间脚本的程序'''
# sku_data = pd.read_excel('D:/AI/xianfengsg/TEST/test.xlsx')
# # print(sku_data)
# list_terminal= ['成都','重庆','西安','郑州','武汉','长沙','合肥','天津','苏州','上海','杭州','宁波','温州','南昌','南京','福州','嘉兴']
# data = pd.DataFrame(columns={'商品名称','商品系统代码','商品代码','目的地','供应月份/月'})
# for i in range(len(sku_data)):
#     for k in range(len(list_terminal)):
#         for v in range(1,13):
#             print('i',i,'k',k,'v',v,sku_data['商品名称'].iloc[i])
#             data= data.append({'商品名称': sku_data['商品名称'].iloc[i],'商品代码': sku_data['商品代码'].iloc[i],
#                                '商品系统代码': sku_data['商品系统代码'].iloc[i],'目的地': list_terminal[k],
#                                '供应月份/月': v},ignore_index=True)
# print(data)
# data.to_excel('D:/AI/xianfengsg/TEST/data.xlsx')


'''测试时间'''
# import datetime
# today=datetime.date.today()
# print(today)
# print(today.strftime('%Y%m%d'))
#
'''测试从采购2.0库中读取历史的天气数据'''
import pymysql

print('连接到mysql服务器...')
db = pymysql.connect(host="rm-bp1jfj82u002onh2tco.mysql.rds.aliyuncs.com",
                     database="purchare_sys", user="purchare_sys",
                     password="purchare_sys@123", port=3306, charset='utf8')
print('连接成功')
weather_sql = """SELECT * FROM weather_history"""
db.cursor()
read_data = pd.read_sql(weather_sql, db)
db.close()
print(read_data)

