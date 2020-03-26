# -*- coding = utf-8 -*-
'''
@Time: 2019/2/20 11:44
@Author: Ye Jinyu
'''
import pandas as pd
import numpy as np
import time
from datetime import datetime,date
from datetime import timedelta
import math
import warnings
import pymysql
import sys
import os
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
from Parameters_evaluation import *
from datetime import timedelta
import datetime
import time
from Parameters import *
import Data_cleaning
import random
import warnings
import math

warnings.filterwarnings("ignore")


def select_good_SKU(finace_Dataframe,manufacture):
    finace_Dataframe_select = finace_Dataframe[finace_Dataframe['custom_business_num'] == manufacture]
    # print(finace_Dataframe_select)
    SKU_sum = finace_Dataframe_select.groupby(['piece_bar_code'], as_index=False).agg(sum)
    # print(SKU_sum)
    SKU_sum['ROI_AI'] = (SKU_sum['Sales_AI']-SKU_sum['tc_AI'])/SKU_sum['tc_AI']
    SKU_sum['ROI_manual'] = (SKU_sum['Sales_manual'] - SKU_sum['tc_manual']) / SKU_sum['tc_manual']
    SKU_sum['error'] = SKU_sum['ROI_AI'] - SKU_sum['ROI_manual']
    # 用于接受那些ROI不比人差的SKU数据
    Dataframe_good = SKU_sum[SKU_sum['error'] >= 0]
    k = len(Dataframe_good)
    ratio_sku = '%.4f%%' % ((k/len((SKU_sum))) *100)
    ratio_sku_sales =  '%.4f%%' % ((Dataframe_good['Sales_AI'].sum()/SKU_sum['Sales_manual'].sum()) *100)
    print('ROI比人工好的sku数量占总sku数量的比值：',ratio_sku)
    print('ROI比人工好的sku销售额占总sku销售额的比值',ratio_sku_sales)
    return Dataframe_good




f_get_calculate_finance = open("D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/v1---仿真"
                                    "/02_仿真输出/evaluation-15/result/2018-09-16/calculate_finance.csv",'rb')
get_calculate_finance = pd.read_csv(f_get_calculate_finance, index_col=0 , keep_default_na=False)
#先对数据进行预处理，保证生产商的格式是数字格式
get_calculate_finance['manufacturer_num'] = get_calculate_finance['manufacturer_num'].apply(pd.to_numeric)
#筛选出需要的生产商
get_calculate_finance = get_calculate_finance[~get_calculate_finance.manufacturer_num.isin(['53'])]
get_calculate_finance = get_calculate_finance[~get_calculate_finance.custom_business_num.isin(['7'])]
#这部操作是为了去除那些异常值的情况，实际计算中可能存在TC_AI是负值的
get_calculate_finance = get_calculate_finance[get_calculate_finance['tc_AI'] > 0]


f_get_calculate_finance_323 = open("D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/v1---仿真"
                                    "/02_仿真输出/evaluation-15/result/新建文件夹/calculate_finance_2.csv",'rb')
get_calculate_finance_323 = pd.read_csv(f_get_calculate_finance_323, index_col=0 , keep_default_na=False)
#先对数据进行预处理，保证生产商的格式是数字格式
get_calculate_finance_323['manufacturer_num'] = get_calculate_finance_323['manufacturer_num'].apply(pd.to_numeric)
#筛选出需要的生产商
get_calculate_finance_323 = get_calculate_finance_323[~get_calculate_finance_323.manufacturer_num.isin(['53'])]

get_calculate_finance_group_323 = get_calculate_finance_323.groupby(['piece_bar_code'], as_index=False).agg(sum)

get_good_sku1 = select_good_SKU(get_calculate_finance,1)
get_good_sku1.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/v1---仿真/get_good_sku1.csv',encoding="utf_8_sig")
get_good_sku2 = select_good_SKU(get_calculate_finance_323,2)
get_good_sku2.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/v1---仿真/get_good_sku2.csv',encoding="utf_8_sig")



get_sku2 = pd.DataFrame(columns=['piece_bar_code'])
get_sku2['piece_bar_code'] = get_good_sku2['piece_bar_code']

get_sku1 = pd.DataFrame(columns=['piece_bar_code'])
get_sku1['piece_bar_code'] = get_good_sku1['piece_bar_code']
# print(get_sku2)
# print(type(get_sku2))
select_good_sku = get_sku1.append(get_sku2)
select_good_sku = select_good_sku.reset_index()
select_good_sku = select_good_sku.drop(['index'], axis=1)
#select_good_sku,数据处理，得到合并后，所有好于人工的SKU


select_good_sku['piece_bar_code'] = select_good_sku['piece_bar_code'].apply(pd.to_numeric)
# print(select_good_sku)
print(type(select_good_sku))
get_calculate_finance_group = get_calculate_finance.groupby(['piece_bar_code'], as_index=False).agg(sum)
# print(get_calculate_finance_group)
get_calculate_finance_group['piece_bar_code'] = get_calculate_finance_group['piece_bar_code'].apply(pd.to_numeric)
# print(get_calculate_finance_group)
final_good_sku = pd.merge(select_good_sku,get_calculate_finance_group, on='piece_bar_code', how='inner')
# sum_tc_AI = final_good_sku['tc_AI'].sum()
# print('宝洁和尤妮佳，AI好于人工的所有SKU，AI的总成本：' , sum_tc_AI)
# sum_tc_manual = final_good_sku['tc_manual'].sum()
# print('宝洁和尤妮佳，AI好于人工的所有SKU，人工的总成本:',sum_tc_manual)
# sum_sales_AI = final_good_sku['Sales_AI'].sum()
# print('宝洁和尤妮佳，AI好于人工的所有SKU，AI的总销售额：' , sum_sales_AI)
# sum_sales_manual = final_good_sku['Sales_manual'].sum()
# print('宝洁和尤妮佳，AI好于人工的所有SKU，人工总销售额:',sum_sales_manual)
# save_money = (sum_sales_AI -sum_tc_AI) - (sum_sales_manual- sum_tc_manual)
# print('宝洁和尤妮佳，从成本角度考虑共节省成本:' , save_money)

#定义一个函数用于具体看那些宝洁产品中AI好于人工的SKU的成本和销售额的评估
# def P_G_good_sku():
final_good_sku_320 = pd.merge(get_sku1,get_calculate_finance_group, on='piece_bar_code', how='inner')
sum_tc_AI_320 = final_good_sku_320['tc_AI'].sum()
print('宝洁，AI好于人工的所有SKU，AI的总成本：' , sum_tc_AI_320)
sum_tc_manual_320 = final_good_sku_320['tc_manual'].sum()
print('宝洁，AI好于人工的所有SKU，人工的总成本:' , sum_tc_manual_320)
sum_sales_AI_320 = final_good_sku_320['Sales_AI'].sum()
print('宝洁，AI好于人工的所有SKU，AI的总销售额：' , sum_sales_AI_320)
sum_sales_manual_320 = final_good_sku_320['Sales_manual'].sum()
print('宝洁，AI好于人工的所有SKU，人工总销售额:',sum_sales_manual_320)
print((sum_sales_AI_320 -sum_tc_AI_320))
print((sum_sales_manual_320- sum_tc_manual_320))
save_money_320 = (sum_sales_AI_320 -sum_tc_AI_320) - (sum_sales_manual_320- sum_tc_manual_320)
print('宝洁，从成本角度考虑共节省成本:' , save_money_320)


get_calculate_finance_group_323['piece_bar_code'] = get_calculate_finance_group_323['piece_bar_code'].apply(pd.to_numeric)

final_good_sku_323 = pd.merge(get_sku2,get_calculate_finance_group_323, on='piece_bar_code', how='inner')
final_good_sku_323.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/v1---仿真'
                      '/v1---final_good_sku_323.csv',encoding="utf_8_sig")
final_good_sku_320.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/v1---仿真'
                      '/v1---final_good_sku_320.csv',encoding="utf_8_sig")

sum_tc_AI_323 = final_good_sku_323['tc_AI'].sum()
print('尤妮佳，AI好于人工的所有SKU，AI的总成本：' , sum_tc_AI_323)
sum_tc_manual_323 = final_good_sku_323['tc_manual'].sum()
print('尤妮佳，AI好于人工的所有SKU，人工的总成本:' , sum_tc_manual_323)
sum_sales_AI_323 = final_good_sku_323['Sales_AI'].sum()
print('尤妮佳，AI好于人工的所有SKU，AI的总销售额：' , sum_sales_AI_323)
sum_sales_manual_323 = final_good_sku_323['Sales_manual'].sum()
print('尤妮佳，AI好于人工的所有SKU，人工总销售额:',sum_sales_manual_323)
save_money_323 = (sum_sales_AI_323 -sum_tc_AI_323) - (sum_sales_manual_323- sum_tc_manual_323)
print('尤妮佳，从成本角度考虑共节省成本:' , save_money_323)


