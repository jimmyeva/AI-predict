# -*- coding = utf-8 -*-
'''
@Time: 2018/11/15 16:59
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
import codecs
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import mpl_toolkits.axisartist.axislines as ax
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
from Parameters_evaluation import *
#导入如下两个API用于对SKU进行LR分析
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.graphics.mosaicplot import mosaic


#------------------------------------------------------------------------------------
#数据修改操作，此步骤是将仿真表格内毛利是负值的情况变成正值
def convert_profit(Data_Frame):
    #此步骤是将负值变正
    Data_Frame_01 =Data_Frame
    i = 0
    while i < len(Data_Frame_01):
        if Data_Frame_01['gross_profit_untax'].iloc[i] < 0:
            Data_Frame_01['gross_profit_untax'].iloc[i] = Data_Frame_01['gross_profit_untax'].iloc[i] * (-1)
        else:
            pass
        i+=1
    return Data_Frame_01
#------------------------------------------------------------------------------------
#数据修改操作，此步骤是将仿真表格内成本是删除成本是负值的数据
def negative_cost(Data_Frame):
    Data_Frame_01 = Data_Frame[Data_Frame['cost_untax'] > 0]
    Data_Frame_01 =Data_Frame_01.reset_index(drop=True)
    return Data_Frame_01


#------------------------------------------------------------------------------------
#数据修改操作，此步骤是将仿真表格内库存是删除成本是负值的数据start_interest_inv	end_interest_inv	start_inv_qty	end_inv_qty

def negative_inv(Data_Frame):
    Data_Frame_01 = Data_Frame[Data_Frame['start_interest_inv'] > 0]
    Data_Frame_02 = Data_Frame_01[Data_Frame_01['end_interest_inv'] > 0]
    Data_Frame_03 = Data_Frame_02[Data_Frame_02['start_inv_qty'] > 0]
    Data_Frame_04 = Data_Frame_03[Data_Frame_03['end_inv_qty'] > 0]
    Data_Frame_04 =Data_Frame_04.reset_index(drop=True)
    return Data_Frame_04




def Calculate_finance(Data_Frame,parameter):
    logistics_fee = per_sku_oc()
    stock_fee = per_day_sku_sc()

    # 以下的操作是用来计算每一个的仿真指标
    # #------------------------------------------------------------------------------------------
    calculate_finance = pd.DataFrame(columns=['cnt_at', 'manufacturer_num', 'custom_business_num', 'custom_stock_num',
                                              'custom_terminal_num', 'piece_bar_code', 'manufacturer_num', 'ROCC_manual', 'ROCC_AI',
                                              'logistical_manual', 'logistical_AI', 'stock_manual',
                                              'stock_AI', 'sc_manual', 'sc_AI', 'tc_manual',
                                              'tc_AI', 'revenue_manual',
                                              'revenue_AI', 'profit_manual', 'profit_AI','Sales_manual','Sales_AI'])

    # 以下为设计表格的基本信息
    calculate_finance['cnt_at'] = Data_Frame['account_date']
    calculate_finance['manufacturer_num'] = Data_Frame['manufacturer_num']
    calculate_finance['custom_business_num'] = Data_Frame['custom_business_num']
    calculate_finance['custom_stock_num'] = Data_Frame['custom_stock_num']
    calculate_finance['custom_terminal_num'] = Data_Frame['custom_terminal_num']
    # calculate_finance['interval_time'] = Data_Frame['interval_time']
    calculate_finance['piece_bar_code'] = Data_Frame['piece_bar_code']
    # 计算人工补货的ROCC费用，物流费用，以及仓储费用，但是仓储费用针对每一个SKU应该是一个
    calculate_finance['ROCC_manual'] = (Data_Frame['start_interest_inv'] - Data_Frame['delivery_qty']) * (
        Data_Frame['cost_untax']) * 0.0002 + Data_Frame['delivery_qty']*Data_Frame['cost_untax']
    calculate_finance['logistical_manual'] = Data_Frame['delivery_qty'] * logistics_fee
    calculate_finance['stock_manual'] = (Data_Frame['start_inv_qty']+Data_Frame['end_inv_qty']) * 0.5\
                                         * stock_fee
    # 人工缺货成本的考虑，暂时先 不考虑
    calculate_finance['sc_manual'] = Data_Frame['shortage_qty_manual'] * Data_Frame['gross_profit_untax'] * parameter
    # 接下来是计算如果按照机器补货的策略应该会产生的成本
    calculate_finance['ROCC_AI'] = (Data_Frame['start_interest_inv_AI'] - Data_Frame['delivery_qty_AI']) * (
        Data_Frame['cost_untax']) * 0.0002 +Data_Frame['delivery_qty_AI']*Data_Frame['cost_untax']
    calculate_finance['logistical_AI'] = Data_Frame['delivery_qty_AI'] * logistics_fee
    calculate_finance['stock_AI'] = (Data_Frame['start_inv_qty_AI']+Data_Frame['end_inv_qty_AI']) * 0.5\
                                         * stock_fee
    calculate_finance['sc_AI'] = Data_Frame['shortage_qty_AI'] * Data_Frame['gross_profit_untax'] * parameter
    calculate_finance['tc_manual'] = calculate_finance['sc_manual'] + calculate_finance['stock_manual'] + \
                                     calculate_finance['logistical_manual'] \
                                     + calculate_finance['ROCC_manual']
    calculate_finance['profit_manual'] = Data_Frame['delivery_qty'] * Data_Frame['gross_profit_untax']
    calculate_finance['tc_AI'] = calculate_finance['sc_AI'] + calculate_finance['stock_AI'] + \
                                 calculate_finance['logistical_AI'] \
                                         + calculate_finance['ROCC_AI']
    calculate_finance['profit_AI'] = Data_Frame['delivery_qty_AI'] * Data_Frame['gross_profit_untax']
    calculate_finance['Sales_manual'] = Data_Frame['delivery_qty'] * \
                                        (Data_Frame['gross_profit_untax']+Data_Frame['cost_untax'])
    calculate_finance['Sales_AI'] = Data_Frame['delivery_qty_AI'] \
                                    *  (Data_Frame['gross_profit_untax']+Data_Frame['cost_untax'])
    calculate_finance['revenue_AI'] = calculate_finance['Sales_AI'] - calculate_finance['tc_AI']
    calculate_finance['revenue_manual'] = calculate_finance['Sales_manual'] - calculate_finance['tc_manual']
    return calculate_finance


#此函数是用来区分生产商，并返回一个财务财务数据和原始的仿真数据，
def distinguish_manufacture(simulation_Dataframe,finance_Dataframe,manufacture):
    simulation_Dataframe = simulation_Dataframe[simulation_Dataframe['custom_business_num'] == manufacture]
    finance_Dataframe = finance_Dataframe[finance_Dataframe['custom_business_num'] == manufacture]
    # finance_Dataframe.to_csv('D:/project/P&G/Code/output/evaluation-15/result/finance_Dataframe.csv',encoding="utf_8_sig")
    return simulation_Dataframe,finance_Dataframe

# 以下是观察期末库存的情况，看每天期末库存的变化
def stock_comparison(simulation_Dataframe,manufacture,file_document):
    simulation_Dataframe_stock = simulation_Dataframe.groupby(['account_date'],as_index=False).agg(sum)
    x = simulation_Dataframe_stock['account_date']
    y1 = simulation_Dataframe_stock['end_inv_qty']
    y2 = simulation_Dataframe_stock['end_inv_qty_AI']
    fig, ax = plt.subplots(1, 1)
    # 包含每个柱子下标的序列
    index = np.arange(len(x))
    # 绘制柱状图, 每根柱子的颜色为紫罗兰色
    p1 = plt.bar(index+0.4,y1, width= 0.4,align = 'center', label='end_inv_qty', color="black")
    p2 = plt.bar(index,y2, width=0.4,align = 'center', label='end_inv_qty_AI', color="#87CEFA")
    # 设置横轴标签
    plt.xlabel('Date')
    # 设置纵轴标签
    plt.ylabel('qty')
    end_inv = '%.2f' % simulation_Dataframe_stock['end_inv_qty'].sum()
    end_inv_AI = '%.2f' % simulation_Dataframe_stock['end_inv_qty_AI'].sum()
    # mean = (simulation_Dataframe_stock['end_inv_qty']+simulation_Dataframe_stock['end_inv_qty_AI']).mean()/3
    max =simulation_Dataframe_stock['end_inv_qty'].max()
    t = "人工的期末实有库存总和为："\
        +str(end_inv)+',AI的期末实有库存为：'+ \
        str(end_inv_AI)
    plt.text(0, -(max/4), t, bbox=dict(facecolor='w', edgecolor='blue', alpha=0.65 ), verticalalignment='bottom',wrap=True)
    # 添加纵横轴的刻度
    plt.xticks(index, x, rotation=45, size=6)
    A = math.floor(len(x) / 5)
    print(A)
    for label in ax.get_xticklabels():
        label.set_visible(False)
    for label in ax.get_xticklabels()[::A]:
        label.set_visible(True)
    # 添加图例
    plt.legend(loc="upper left")
    plt.title('overview stock manual & AI')
    # plt.show() #一定要把这行注释，否则保存下来的图标就是一张白色的图片什么内容都没有
    plt.savefig("D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/"
                            "v1---仿真/02_仿真输出/evaluation-17/"+str(file_document)+"/"+"simulation_Dataframe_stock" + str(manufacture) + '.jpg', dpi=400,
        bbox_inches='tight')
    plt.close()


def interest_stock(simulation_Dataframe,manufacture,file_document):
    simulation_Dataframe_stock = simulation_Dataframe.groupby(['account_date'],as_index=False).agg(sum)
    x = simulation_Dataframe_stock['account_date']
    y1 = simulation_Dataframe_stock['end_interest_inv']
    y2 = simulation_Dataframe_stock['end_interest_inv_AI']
    fig, ax = plt.subplots(1, 1)
    # 包含每个柱子下标的序列
    index = np.arange(len(x))
    # 绘制柱状图, 每根柱子的颜色为紫罗兰色
    p1 = plt.bar(index+0.4,y1, width= 0.4,align = 'center', label='end_interest_inv', color="black")
    p2 = plt.bar(index,y2, width=0.4,align = 'center', label='end_interest_inv_AI', color="#87CEFA")
    # 设置横轴标签
    plt.xlabel('Date')
    # 设置纵轴标签
    plt.ylabel('qty')

    end_inv = '%.2f' % simulation_Dataframe_stock['end_interest_inv'].sum()
    end_inv_AI = '%.2f' % simulation_Dataframe_stock['end_interest_inv_AI'].sum()
    # mean = (simulation_Dataframe_stock['end_interest_inv']+simulation_Dataframe_stock['end_interest_inv_AI']).mean()/3
    max = simulation_Dataframe_stock['end_interest_inv'].max()
    t = "人工的期末物权库存总和为："\
        +str(end_inv)+',AI的期末物权库存为：'+ \
        str(end_inv_AI)
    plt.text(0, -(max/4), t, bbox=dict(facecolor='w', edgecolor='blue', alpha=0.65 ), verticalalignment='bottom',wrap=True)
    # 添加纵横轴的刻度
    plt.xticks(index, x, rotation=45, size=6)
    A = math.floor(len(x) / 5)
    #存在真实数据为0的情况
    if A == 0:
        A = 1
    else:
        pass
    for label in ax.get_xticklabels():
        label.set_visible(False)
    for label in ax.get_xticklabels()[::A]:
        label.set_visible(True)
    # 添加图例
    plt.legend(loc="upper left")
    plt.title('overview interest stock manual & AI')
    # plt.show() #一定要把这行注释，否则保存下来的图标就是一张白色的图片什么内容都没有
    plt.savefig("D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/"
                            "v1---仿真/02_仿真输出/evaluation-17/"+str(file_document)+"/"+"simulation_Dataframe_interest_stock" + str(manufacture) + '.jpg', dpi=400,
        bbox_inches='tight')
    plt.close()

def shortages_comparison(simulation_Dataframe,manufacture,file_document):
    simulation_shortages = simulation_Dataframe.groupby(['account_date'],as_index=False).agg(sum)
    x = simulation_shortages['account_date']
    y1 = simulation_shortages['shortage_qty_manual']
    y2 = simulation_shortages['shortage_qty_AI']
    fig, ax = plt.subplots(1, 1)
    # 包含每个柱子下标的序列
    index = np.arange(len(x))
    # 绘制柱状图, 每根柱子的颜色为紫罗兰色
    p1 = plt.bar(index+0.4,y1, width= 0.4,align = 'center', label='shortage_qty_manual', color="black")
    p2 = plt.bar(index,y2, width=0.4,align = 'center', label='shortage_qty_AI', color="#87CEFA")
    # 设置横轴标签
    plt.xlabel('Date')
    # 设置纵轴标签
    plt.ylabel('qty')
    # 添加纵横轴的刻度
    end_inv = '%.2f' % simulation_shortages['shortage_qty_manual'].sum()
    end_inv_AI = '%.2f' % simulation_shortages['shortage_qty_AI'].sum()
    # mean = (simulation_shortages['shortage_qty_manual']+simulation_shortages['shortage_qty_AI']).mean()/3
    max = simulation_shortages['shortage_qty_AI'].max()
    t = "人工的缺货量总和为："\
        +str(end_inv)+',AI的缺货量为：'+ \
        str(end_inv_AI)
    plt.text(0, -(max/4), t, bbox=dict(facecolor='w', edgecolor='blue', alpha=0.65 ), verticalalignment='bottom',wrap=True)
    # 添加纵横轴的刻度
    plt.xticks(index, x, rotation=45, size=6)
    plt.xticks(index, x, rotation=45, size=6)
    A = math.floor(len(x) / 5)
    #存在真实数据为0的情况
    if A == 0:
        A = 1
    else:
        pass
    for label in ax.get_xticklabels():
        label.set_visible(False)
    for label in ax.get_xticklabels()[::A]:
        label.set_visible(True)
    # 添加图例
    plt.legend(loc="upper left")
    plt.title('AI & 人工 缺货量对比')
    # plt.show() #一定要把这行注释，否则保存下来的图标就是一张白色的图片什么内容都没有
    plt.savefig("D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/"
                            "v1---仿真/02_仿真输出/evaluation-17/"+str(file_document)+"/"+"shortages_comparison" + str(manufacture) + '.jpg', dpi=400,
        bbox_inches='tight')
    plt.close()

#以下是对比出库量
def delivery_comparison(simulation_Dataframe,manufacture,file_document):
    simulation_delivery = simulation_Dataframe.groupby(['account_date'],as_index=False).agg(sum)
    x = simulation_delivery['account_date']
    y1 = simulation_delivery['delivery_qty']
    y2 = simulation_delivery['delivery_qty_AI']
    fig, ax = plt.subplots(1, 1)
    # 包含每个柱子下标的序列
    index = np.arange(len(x))
    # 绘制柱状图, 每根柱子的颜色为紫罗兰色
    p1 = plt.bar(index+0.4,y1, width= 0.4,align = 'center', label='delivery_qty', color="black")
    p2 = plt.bar(index,y2, width=0.4,align = 'center', label='delivery_qty_AI', color="#87CEFA")
    # 设置横轴标签
    plt.xlabel('Date')
    # 设置纵轴标签
    plt.ylabel('qty')
    # 添加纵横轴的刻度
    end_inv = '%.2f' % simulation_delivery['delivery_qty'].sum()
    end_inv_AI = '%.2f' % simulation_delivery['delivery_qty_AI'].sum()
    # mean = (simulation_delivery['delivery_qty']+simulation_delivery['delivery_qty_AI']).mean()/3
    max = simulation_delivery['delivery_qty'].max()
    t = "人工的出库量总和为："\
        +str(end_inv)+',AI的出库量为：'+ \
        str(end_inv_AI)
    plt.text(0, -(max/4), t, bbox=dict(facecolor='w', edgecolor='blue', alpha=0.65 ), verticalalignment='bottom',wrap=True)
    plt.xticks(index, x, rotation=45, size=6)
    A = math.floor(len(x) / 5)
    #存在真实数据为0的情况
    if A == 0:
        A = 1
    else:
        pass
    for label in ax.get_xticklabels():
        label.set_visible(False)
    for label in ax.get_xticklabels()[::A]:
        label.set_visible(True)
    # 添加图例
    plt.legend(loc="upper left")
    plt.title('AI & 人工 出库量对比')
    # plt.show() #一定要把这行注释，否则保存下来的图标就是一张白色的图片什么内容都没有
    plt.savefig("D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/"
                            "v1---仿真/02_仿真输出/evaluation-17/"+str(file_document)+"/"+"delivery_comparison" + str(manufacture) + '.jpg', dpi=400,
        bbox_inches='tight')
    plt.close()

#定义函数来显示，AI和人工的成本数据
def ROCC_comparison(finance_Dataframe,manufacture,file_document):
    simulation_delivery = finance_Dataframe.groupby(['cnt_at'],as_index=False).agg(sum)
    x = simulation_delivery['cnt_at']
    y1 = simulation_delivery['ROCC_manual']
    y2 = simulation_delivery['ROCC_AI']
    fig, ax = plt.subplots(1, 1)
    # 包含每个柱子下标的序列
    index = np.arange(len(x))
    # 绘制柱状图, 每根柱子的颜色为紫罗兰色
    p1 = plt.bar(index+0.4,y1, width= 0.4,align = 'center', label='ROCC_manual', color="black")
    p2 = plt.bar(index,y2, width=0.4,align = 'center', label='ROCC_AI', color="#87CEFA")
    # 设置横轴标签
    plt.xlabel('Date')
    # 设置纵轴标签
    plt.ylabel('qty')
    # 添加纵横轴的刻度
    end_inv = '%.2f' % simulation_delivery['ROCC_manual'].sum()
    end_inv_AI = '%.2f' % simulation_delivery['ROCC_AI'].sum()
    # mean = (simulation_delivery['delivery_qty']+simulation_delivery['delivery_qty_AI']).mean()/3
    max = simulation_delivery['ROCC_manual'].max()
    t = "人工的ROCC总和为："\
        +str(end_inv)+',AI的ROCC为：'+ \
        str(end_inv_AI)
    plt.text(0, -(max/4), t, bbox=dict(facecolor='w', edgecolor='blue', alpha=0.65 ), verticalalignment='bottom',wrap=True)
    plt.xticks(index, x, rotation=45, size=6)
    A = math.floor(len(x) / 5)
    #存在真实数据为0的情况
    if A == 0:
        A = 1
    else:
        pass
    for label in ax.get_xticklabels():
        label.set_visible(False)
    for label in ax.get_xticklabels()[::A]:
        label.set_visible(True)
    # 添加图例
    plt.legend(loc="upper left")
    plt.title('AI & 人工 ROCC对比')
    # plt.show() #一定要把这行注释，否则保存下来的图标就是一张白色的图片什么内容都没有
    plt.savefig("D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/"
                            "v1---仿真/02_仿真输出/evaluation-17/"+str(file_document)+"/"+"ROCC_comparison" + str(manufacture) + '.jpg', dpi=400,
        bbox_inches='tight')
    plt.close()

#此函数是用来计算那些整体的仿真指标，包括总的（销售额/成本/ROI）制作成双轴，销售额和成本y1，ROIy2,包含两个维度，一个是整体的，一个是每天的
def overview_sales(finace_Dataframe,simulation_Dataframe,manufacture,file_document):
    #先把原来的所有的数据进行加和，观察所有AI和人工的差异
    Sale_AI= finace_Dataframe['Sales_AI'].sum()
    Sales_manual = finace_Dataframe['Sales_manual'].sum()
    TC_AI = finace_Dataframe['tc_AI'].sum()
    TC_Manual = finace_Dataframe['tc_manual'].sum()
    # ROI_Manual= (Sales_manual-TC_Manual)/TC_Manual
    # ROI_AI=  (Sale_AI-TC_AI)/TC_AI
    finace_Dataframe_sum = pd.DataFrame({
      '生产环境/指标':['现实人工环境','AI仿真环境','差值'],
    '总销售额（元）':['%.2f' % Sales_manual,'%.2f' % Sale_AI, '%.2f'%(Sales_manual-Sale_AI)],
    '总成本（元）':['%.2f' % TC_Manual, '%.2f' % TC_AI, '%.2f'%(TC_Manual-TC_AI)],
    # '整体_ROI':['%.4f%%' % (ROI_Manual * 100), '%.4f%%' % (ROI_AI * 100), '%.4f%%' % ((ROI_Manual-ROI_AI) * 100)]
    })
    print(finace_Dataframe_sum)
    stock_comparison(simulation_Dataframe,manufacture,file_document)
    shortages_comparison(simulation_Dataframe,manufacture,file_document)
    delivery_comparison(simulation_Dataframe,manufacture,file_document)
    interest_stock(simulation_Dataframe,manufacture,file_document)
    ROCC_comparison(finace_Dataframe,manufacture,file_document)



#定义函数来对SKU进行分析
#-----------------------------------------------------------------------

#对于ROI不如人工的SKU，计算统计他们的销量和销售额和利润,输入的是SKU的piecebarcode和自定义组织，和仿真表格,在select_good_SKU函数中会使用
def sales_sku(bad_sku_barcode,manufacture,simulation_Dataframe):
    evaluation_sku = simulation_Dataframe\
    [simulation_Dataframe['piece_bar_code'] == bad_sku_barcode]
    evaluation_sku = evaluation_sku[evaluation_sku['custom_business_num'] ==  manufacture]
    #记录有销售的天数
    day = len(evaluation_sku['account_date'])
    # print(day)
    #记录一共卖了多少件SKU
    qty = evaluation_sku['delivery_qty'].sum()
    # print(qty)
    #记录一共的销售额
    sales = (evaluation_sku['delivery_qty'] *
             (evaluation_sku['cost_untax'] + evaluation_sku['gross_profit_untax'])).sum()
    # print(sales)
    profit = (evaluation_sku['delivery_qty'] *evaluation_sku['gross_profit_untax']).sum()
    #新建一个空的Dataframe来记录每个sku销售情况
    sales_sku = pd.DataFrame({
        'piece_bar_code':bad_sku_barcode,
        'days' : day,
        'delivery_qty' : qty,
        'sales' : sales,
        'profit': profit
    },index = [0])
    return sales_sku


#先选择出那些总金额优于人工的比人工的好的SKU并列举出来,manufacture参数是可以根据不同的参数选择不同的生产商进行分析
#可以按照设定的路径存到相应的位置，同时也可以分别好的SKU和坏的SKU
def select_good_SKU(finace_Dataframe,simulation_Dataframe,manufacture):
    finace_Dataframe_select = finace_Dataframe[finace_Dataframe['custom_business_num'] == manufacture]
    # print(finace_Dataframe_select)
    SKU_sum = finace_Dataframe_select.groupby(['piece_bar_code'], as_index=False).agg(sum)
    # print(SKU_sum)
    SKU_sum['revenue_AI'] = (SKU_sum['Sales_AI'] - SKU_sum['tc_AI'])
    SKU_sum['revenue_manual'] = (SKU_sum['Sales_manual'] - SKU_sum['tc_manual'])

    SKU_sum['error'] = SKU_sum['revenue_AI'] - SKU_sum['revenue_manual']
    # 用于接受那些收益不比人差的SKU数据
    Dataframe_good = SKU_sum[SKU_sum['error'] >= 0]
    #对那些收益不比人差的SKU进行收益排序
    Dataframe_good =Dataframe_good.sort_values(by='error', ascending=False)
    k = len(Dataframe_good)
    ratio_sku = '%.4f%%' % ((k/len((SKU_sum))) *100)
    # 在那些比ROI高于人工的SKU进行继续分析，判断仿真时间内比人好的次数已经比人差的次数
    # ------------------------------------------------------------------------------------------------
    Barcode = set(Dataframe_good['piece_bar_code'])
    print('仿真共有' + str(len(SKU_sum)) + '款SKU,其中AI补货的收益不差于人工的SKU共有' + str(k) + '款。'+'自动化率为：'\
          +str(ratio_sku)+'。')
    print('优于人工补货的SKU是：'+str(Barcode))
    #以下是找到那些补货ROI差于人工的SKU
    Dataframe_bad = SKU_sum[SKU_sum['error'] < 0]
    #对那些收益不比人差的SKU进行收益排序
    Dataframe_bad =Dataframe_bad.sort_values(by='error', ascending=True)
    Bad_SKU_barcode = set(Dataframe_bad['piece_bar_code'])
    #以下的循环操作是针对那些不好的SKU，查看他们的销售出库的信息，看基本的统计信息
    sum_bad_SKU = pd.DataFrame()
    Sales_bad_sku = pd.DataFrame(columns=['piece_bar_code','days','delivery_qty','sales'])
    get_sku_sql = """select * from mid_cj_sales where custom_business_num = %s 
                and delivery_type_name = '销售'
                and custom_terminal_num = '1' """ % (manufacture)
    get_sku = Mysql_Data(get_sku_sql)
    for barcode in Bad_SKU_barcode:
        get_bad_sku = get_sku[get_sku['piece_bar_code'] ==barcode]
        df = pd.DataFrame({str(barcode):get_bad_sku['delivery_qty']})
        sum_bad_SKU = sum_bad_SKU.append(df, ignore_index=True)
        sales_sku_per_bad = sales_sku(barcode,manufacture,simulation_Dataframe)
        Sales_bad_sku = Sales_bad_sku.append(sales_sku_per_bad)
        # print(sales_sku)
    # print(sum_SKU)
    sum_bad_sku_describe = sum_bad_SKU.describe()
    #以下是在找出那些ROI好于人工的SKU，并分析他们的基本统计学信息
    good_SKU_barcode = set(Dataframe_good['piece_bar_code'])
    #以下的循环操作是针对那些不好的SKU，查看他们的销售出库的信息，看基本的统计信息
    sum_good_SKU = pd.DataFrame()
    Sales_good_sku = pd.DataFrame(columns=['piece_bar_code','days','delivery_qty','sales'])
    for barcode in good_SKU_barcode:
        get_good_sku = get_sku[get_sku['piece_bar_code'] == barcode]
        df = pd.DataFrame({str(barcode):get_good_sku['delivery_qty']})
        sum_good_SKU = sum_good_SKU.append(df, ignore_index=True)
        sales_sku_per_good = sales_sku(barcode,manufacture,simulation_Dataframe)
        Sales_good_sku = Sales_good_sku.append(sales_sku_per_good)
    sum_good_sku_describe = sum_good_SKU.describe()
    profit= Dataframe_good['profit_manual'].sum()
    sales = Dataframe_good['Sales_manual'].sum()
    total_sales = finace_Dataframe_select['Sales_manual'].sum()
    total_profit = finace_Dataframe_select['profit_manual'].sum()
    print('仿真中AI补货优于人工的SKU的销售总额为：' + str(sales) + '，占总的销售额的：' + str(('%.4f%%' %((sales/total_sales)*100)) + \
          ',总的利润为：'+ str(profit) +'，占到总利润的：'\
          +str('%.4f%%' % ((profit/total_profit)*100))+'。'))

    Sales_bad_sku.to_csv("D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/"
                            "v1---仿真/02_仿真输出/evaluation-17/Sales_bad_sku"+str(manufacture)+'.csv', encoding="utf_8_sig")
    Sales_good_sku.to_csv("D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/"
                            "v1---仿真/02_仿真输出/evaluation-17/Sales_good_sku"+str(manufacture)+'.csv',encoding="utf_8_sig")
    sum_good_sku_describe.to_csv("D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/"
                            "v1---仿真/02_仿真输出/evaluation-17/sum_good_sku_describe"+str(manufacture)+'.csv',encoding="utf_8_sig")
    sum_bad_sku_describe.to_csv("D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/"
                            "v1---仿真/02_仿真输出/evaluation-17/sum_bad_sku_describe"+str(manufacture)+'.csv', encoding="utf_8_sig")
    return Dataframe_good,Dataframe_bad


#定义一个函数用来看好于人工的SKU的一些财务指标
def P_G_good_sku(Dataframe_good,manufacture):
    sum_tc_AI_320 = Dataframe_good['tc_AI'].sum()
    print('生产商'+str(manufacture)+',AI好于人工的所有SKU，AI的总成本：' , sum_tc_AI_320)
    sum_tc_manual_320 = Dataframe_good['tc_manual'].sum()
    print('生产商'+str(manufacture)+',AI好于人工的所有SKU，人工的总成本:' , sum_tc_manual_320)
    sum_sales_AI_320 = Dataframe_good['Sales_AI'].sum()
    print('生产商'+str(manufacture)+',AI好于人工的所有SKU，AI的总销售额：' , sum_sales_AI_320)
    sum_sales_manual_320 = Dataframe_good['Sales_manual'].sum()
    print('生产商'+str(manufacture)+',AI好于人工的所有SKU，人工总销售额:',sum_sales_manual_320)
    print((sum_sales_AI_320 -sum_tc_AI_320))
    print((sum_sales_manual_320- sum_tc_manual_320))
    save_money_320 = (sum_sales_AI_320 -sum_tc_AI_320) - (sum_sales_manual_320- sum_tc_manual_320)
    print('生产商'+str(manufacture)+',从成本角度考虑共节省成本:' , save_money_320)


#选择补货AI好于人工的SKU的5款SKU进行画图和数据展示
def top_good_5(calculate_finance,simulation_Dataframe,finace_Dataframe,manufacture,file_document):
    #先对好的sku的dataframe进行error排序，从而选出排名前五的SKU的piece_bar_code
    finace_Dataframe_select = finace_Dataframe[finace_Dataframe['custom_business_num'] == manufacture]
    # print(finace_Dataframe_select)
    SKU_sum = finace_Dataframe_select.groupby(['piece_bar_code'], as_index=False).agg(sum)
    # print(SKU_sum)
    SKU_sum['revenue_AI'] = (SKU_sum['Sales_AI'] - SKU_sum['tc_AI'])
    SKU_sum['revenue_manual'] = (SKU_sum['Sales_manual'] - SKU_sum['tc_manual'])

    SKU_sum['error'] = SKU_sum['Sales_AI'] - SKU_sum['Sales_manual']
    SKU_sum_good = SKU_sum[SKU_sum['error'] >0]
    Dataframe_good_01 =SKU_sum_good.sort_values(by='error',ascending=False)
    Dataframe_good_01 = Dataframe_good_01.reset_index(drop=True)
    for i in range(0,5):
        piece_bar_code = Dataframe_good_01['piece_bar_code'].iloc[i]
        print(piece_bar_code)
        simulation_Dataframe_good = simulation_Dataframe[simulation_Dataframe['piece_bar_code'] == piece_bar_code]
        simulation_Dataframe_good = simulation_Dataframe_good[simulation_Dataframe_good['custom_business_num'] == manufacture]
        finace_Dataframe_good = finace_Dataframe[finace_Dataframe['piece_bar_code'] == piece_bar_code]
        finace_Dataframe_good = finace_Dataframe_good[finace_Dataframe_good['custom_business_num'] == manufacture]
        overview_sales(finace_Dataframe_good,simulation_Dataframe_good,piece_bar_code,file_document)

def AI_great_manual(Dataframe_good,simulation_Dataframe,finace_Dataframe,manufacture,file_document):
    #先对好的sku查看，那些收益优于人工，并且销售额大于人工的sku
    Dataframe_good['sales_error'] = Dataframe_good['Sales_AI'] - Dataframe_good['Sales_manual']
    Dataframe_good_01 = Dataframe_good[Dataframe_good['sales_error'] > 0]
    Dataframe_good_01 = Dataframe_good_01[Dataframe_good_01['Sales_manual'] != 0]
    #先对好的sku的dataframe进行error排序，从而选出排名前五的SKU的piece_bar_code
    Dataframe_good_01 = Dataframe_good_01.sort_values(by='sales_error',ascending=False)
    Dataframe_good_01 = Dataframe_good_01.reset_index(drop=True)
    for i in range(0,5):
        piece_bar_code = Dataframe_good_01['piece_bar_code'].iloc[i]
        print(piece_bar_code)
        simulation_Dataframe_good = simulation_Dataframe[simulation_Dataframe['piece_bar_code'] == piece_bar_code]
        simulation_Dataframe_good = simulation_Dataframe_good[simulation_Dataframe_good['custom_business_num'] == manufacture]
        finace_Dataframe_good = finace_Dataframe[finace_Dataframe['piece_bar_code'] == piece_bar_code]
        finace_Dataframe_good = finace_Dataframe_good[finace_Dataframe_good['custom_business_num'] == manufacture]
        overview_sales(finace_Dataframe_good,simulation_Dataframe_good,piece_bar_code,file_document)

#这是选择几个最差的SKU拿出来分析
def AI_bad_manual_worst(Dataframe_bad,simulation_Dataframe,finace_Dataframe,manufacture,file_document):
    #先对好的sku查看，那些收益优于人工，并且销售额大于人工的sku
    Dataframe_bad['sales_error'] = Dataframe_bad['Sales_AI'] - Dataframe_bad['Sales_manual']
    Dataframe_bad_01 = Dataframe_bad[Dataframe_bad['sales_error'] < 0]
    Dataframe_bad_01 = Dataframe_bad_01[Dataframe_bad_01['Sales_AI'] != 0]
    #先对好的sku的dataframe进行error排序，从而选出排名前五的SKU的piece_bar_code
    Dataframe_bad_01 = Dataframe_bad_01.sort_values(by='sales_error',ascending=True)
    Dataframe_bad_01 = Dataframe_bad_01.reset_index(drop=True)
    for i in range(0,5):
        piece_bar_code = Dataframe_bad_01['piece_bar_code'].iloc[i]
        print(piece_bar_code)
        simulation_Dataframe_good = simulation_Dataframe[simulation_Dataframe['piece_bar_code'] == piece_bar_code]
        simulation_Dataframe_good = simulation_Dataframe_good[simulation_Dataframe_good['custom_business_num'] == manufacture]
        finace_Dataframe_good = finace_Dataframe[finace_Dataframe['piece_bar_code'] == piece_bar_code]
        finace_Dataframe_good = finace_Dataframe_good[finace_Dataframe_good['custom_business_num'] == manufacture]
        overview_sales(finace_Dataframe_good,simulation_Dataframe_good,piece_bar_code,file_document)

#这是选择几个差的当中相对好的
def AI_bad_manual_relative_better(Dataframe_bad,simulation_Dataframe,finace_Dataframe,manufacture,file_document):
    #先对好的sku查看，那些收益优于人工，并且销售额大于人工的sku
    Dataframe_bad['sales_error'] = Dataframe_bad['Sales_AI'] - Dataframe_bad['Sales_manual']
    Dataframe_bad_01 = Dataframe_bad[Dataframe_bad['sales_error'] < 0]
    Dataframe_bad_01 = Dataframe_bad_01[Dataframe_bad_01['Sales_AI'] != 0]
    #先对好的sku的dataframe进行error排序，从而选出排名前五的SKU的piece_bar_code
    Dataframe_bad_01 = Dataframe_bad_01.sort_values(by='sales_error',ascending=False)
    Dataframe_bad_01 = Dataframe_bad_01.reset_index(drop=True)
    for i in range(0,5):
        piece_bar_code = Dataframe_bad_01['piece_bar_code'].iloc[i]
        print(piece_bar_code)
        simulation_Dataframe_good = simulation_Dataframe[simulation_Dataframe['piece_bar_code'] == piece_bar_code]
        simulation_Dataframe_good = simulation_Dataframe_good[simulation_Dataframe_good['custom_business_num'] == manufacture]
        finace_Dataframe_good = finace_Dataframe[finace_Dataframe['piece_bar_code'] == piece_bar_code]
        finace_Dataframe_good = finace_Dataframe_good[finace_Dataframe_good['custom_business_num'] == manufacture]
        overview_sales(finace_Dataframe_good,simulation_Dataframe_good,piece_bar_code,file_document)


#定一个函数查询销量值

def query_sales(manufacturer_num, custom_business_num, custom_stock_num, custom_terminal_num, start_sale_time,sku):
    sale_sql = """select piece_bar_code ,delivery_qty,account_date from mid_cj_sales
                                  where piece_bar_code =%s
                                  and manufacturer_num = %s
                                  and custom_business_num = %s 
                                  and custom_stock_num = %s
                                  and custom_terminal_num = %s
                                  and account_date > %s""" \
               % (sku,manufacturer_num, custom_business_num, custom_stock_num, custom_terminal_num, start_sale_time)
    get_sales = Mysql_Data(sale_sql)
    get_sales.columns = ['piece_bar_code' ,'delivery_qty','account_date']
    get_sales['account_date'] = get_sales['account_date'].apply(
        lambda x: datetime.strftime(x, '%Y-%m-%d'))
    # get_original_sales_002['account_date'] = datetime.date(get_original_sales_002['account_date'])
    # # print(barcode_01)
    x = get_sales['account_date']
    y = get_sales['delivery_qty']
    fig, ax = plt.subplots(1, 1)
    # 包含每个柱子下标的序列
    index = np.arange(len(x))
    # 绘制柱状图, 每根柱子的颜色为紫罗兰色
    p2 = plt.bar(index, y, width= 0.4, label=str(sku), color="#87CEFA")
    # 设置横轴标签
    plt.xlabel('Date')
    # 设置纵轴标签
    plt.ylabel('real_qty')
    # 添加标题
    plt.title(str(sku))
    # 添加纵横轴的刻度
    # print(x)
    plt.xticks(index, x, rotation=45, size=6)
    A = math.floor(len(x) / 5)
    for label in ax.get_xticklabels():
        label.set_visible(False)
    for label in ax.get_xticklabels()[::A]:
        label.set_visible(True)
    # 添加图例
    plt.show()
    # plt.legend(loc="upper right")
    # plt.savefig('D:/project/P&G/Code/补货代码/SKU数据形态/top_order/' + str(barcode) + '.jpg', dpi=400, bbox_inches='tight')
    plt.close()

def query_forecast(manufacturer_num, custom_business_num, custom_stock_num, custom_terminal_num,sku):
    forecast_sql = """select cnt_at,piece_bar_code,forecast_qty from dm_cj_forecast
                                  where piece_bar_code =%s
                                  and manufacturer_num = %s
                                  and custom_business_num = %s 
                                  and custom_stock_num = %s
                                  and custom_terminal_num = %s""" \
               % (sku,manufacturer_num, custom_business_num, custom_stock_num, custom_terminal_num)
    get_forecast = Mysql_Data(forecast_sql)
    get_forecast.columns = ['cnt_at','piece_bar_code','forecast_qty']
    get_forecast = get_forecast.groupby(['cnt_at'],as_index=False).mean()
    get_forecast['cnt_at'] = get_forecast['cnt_at'].apply(
        lambda x: datetime.strftime(x, '%Y-%m-%d'))
    # get_original_sales_002['account_date'] = datetime.date(get_original_sales_002['account_date'])
    # # print(barcode_01)
    x = get_forecast['cnt_at']
    y = get_forecast['forecast_qty']
    fig, ax = plt.subplots(1, 1)
    # 包含每个柱子下标的序列
    index = np.arange(len(x))
    # 绘制柱状图, 每根柱子的颜色为紫罗兰色
    p2 = plt.bar(index, y, width= 0.4, label=str(sku), color="#87CEFA")
    # 设置横轴标签
    plt.xlabel('Date')
    # 设置纵轴标签
    plt.ylabel('Forecast_qty')
    # 添加标题
    plt.title(str(sku))
    # 添加纵横轴的刻度
    # print(x)
    plt.xticks(index, x, rotation=45, size=6)
    A = math.floor(len(x) / 5)
    for label in ax.get_xticklabels():
        label.set_visible(False)
    for label in ax.get_xticklabels()[::A]:
        label.set_visible(True)
    # 添加图例
    plt.show()
    # plt.legend(loc="upper right")
    # plt.savefig('D:/project/P&G/Code/补货代码/SKU数据形态/top_order/' + str(barcode) + '.jpg', dpi=400, bbox_inches='tight')
    plt.close()

# query_sales('000320','3','1','1','20180101','6903148209103')
# query_forecast('000320','3','1','1','6903148209103')
#




#查看某个SKU下单情况的比较
def order_comparison(piece_bar_code,simulation_data,custom_business_num):
    data_frame_01 = simulation_data[simulation_data['custom_business_num'] == custom_business_num]
    data_frame = data_frame_01[data_frame_01['piece_bar_code'] ==piece_bar_code]
    x = data_frame['account_date']
    y1 = data_frame['purchase_order_manual']
    y2 = data_frame['purchase_order_qty_AI']
    fig, ax = plt.subplots(1, 1)
    # 包含每个柱子下标的序列
    index = np.arange(len(x))
    # 绘制柱状图, 每根柱子的颜色为紫罗兰色
    p1 = plt.bar(index + 0.4, y1, width=0.4, align='center', label='purchase_order_manual', color="black")
    p2 = plt.bar(index, y2, width=0.4, align='center', label='purchase_order_qty_AI', color="#87CEFA")
    # 设置横轴标签
    plt.xlabel('Date')
    # 设置纵轴标签
    plt.ylabel('qty')
    # 添加纵横轴的刻度
    purchase_order_manual = '%.2f' % data_frame['purchase_order_manual'].sum()
    purchase_order_qty_AI = '%.2f' % data_frame['purchase_order_qty_AI'].sum()
    # mean = (simulation_delivery['delivery_qty']+simulation_delivery['delivery_qty_AI']).mean()/3
    max = data_frame['purchase_order_qty_AI'].max()
    t = "人工的下单总量为：" \
        + str(purchase_order_manual) + ',AI的下单总量为：' + \
        str(purchase_order_qty_AI)
    plt.text(0, -(max / 4), t, bbox=dict(facecolor='w', edgecolor='blue', alpha=0.65), verticalalignment='bottom',
             wrap=True)
    plt.xticks(index, x, rotation=45, size=6)
    A = math.floor(len(x) / 5)
    # 存在真实数据为0的情况
    if A == 0:
        A = 1
    else:
        pass
    for label in ax.get_xticklabels():
        label.set_visible(False)
    for label in ax.get_xticklabels()[::A]:
        label.set_visible(True)
    # 添加图例
    # plt.legend(loc="upper left")
    plt.title('AI & 人工 下单量对比')
    plt.show() #一定要把这行注释，否则保存下来的图标就是一张白色的图片什么内容都没有
    # plt.savefig("D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/"
    #             "v1---仿真/02_仿真输出/evaluation-17/" + str(file_document) + "/" + "ROCC_comparison" + str(
    #     manufacture) + '.jpg', dpi=400,
    #             bbox_inches='tight')
    plt.close()


#在进行分析操作之前先需要算出simulation的财务参数等，先读取simulation的信息，设置的参数是0，代表不去缺货进行惩罚，再将得到的ROI存储到对应的表格中
simulation_Dataframe_01 = codecs.open("D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/"
                            "v1---仿真/02_仿真输出/evaluation-17/final_sheet20190306.csv",'rb')
simulation_Dataframe_02 = pd.read_csv(simulation_Dataframe_01,  keep_default_na=False)
# simulation_Dataframe_03 = convert_profit(simulation_Dataframe_02)
# simulation_Dataframe_04 = negative_inv(simulation_Dataframe_03)
# simulation_Dataframe = negative_cost(simulation_Dataframe_04)
# simulation_Dataframe.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/v1---仿真/02_仿真输出/evaluation-17/simulation_Dataframe.csv',encoding="utf_8_sig")
#
# calculate_finance = Calculate_finance(simulation_Dataframe,0)
# calculate_finance.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/v1---仿真/02_仿真输出/evaluation-17/calculate_finance.csv',encoding="utf_8_sig")
# overview_sales(calculate_finance,simulation_Dataframe,3,'over_view')
# Dataframe_good,Dataframe_bad= select_good_SKU(calculate_finance,simulation_Dataframe,3)
# Dataframe_good.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/v1---仿真/02_仿真输出/evaluation-17/Dataframe_good.csv',encoding="utf_8_sig")
# P_G_good_sku(Dataframe_good,320)
# top_good_5(calculate_finance,simulation_Dataframe,calculate_finance,3,'top_good_5')
# AI_great_manual(Dataframe_good,simulation_Dataframe,calculate_finance,3,'AI_great_manual')
# AI_bad_manual_relative_better(Dataframe_bad,simulation_Dataframe,calculate_finance,3,'AI_bad_manual_relative_better')
# AI_bad_manual_worst(Dataframe_bad,simulation_Dataframe,calculate_finance,3,'AI_bad_manual_relative_better')
order_comparison('6903148209103',simulation_Dataframe_02,'3')













































































































#定义一个函数，作为单次补货的建议值，看每天的补货的ROI，那些补货不如人工的天数的列举出来
def evaluation_perday(finace_Dataframe,simulation_Dataframe,manufacture):
    finace_Dataframe_select = finace_Dataframe[finace_Dataframe['custom_business_num'] == manufacture]
    finace_Dataframe_select_agg = finace_Dataframe_select.groupby(['cnt_at'], as_index=False).agg(sum)
    finace_Dataframe_select_agg['ROI_AI'] =  (finace_Dataframe_select_agg['Sales_AI'] - finace_Dataframe_select_agg['tc_AI'])\
                                         / finace_Dataframe_select_agg['tc_AI']
    finace_Dataframe_select_agg['ROI_manual'] = (finace_Dataframe_select_agg['Sales_manual'] - finace_Dataframe_select_agg['tc_manual'])\
                                         / finace_Dataframe_select_agg['tc_manual']
    finace_Dataframe_select_agg['error'] = finace_Dataframe_select_agg['ROI_manual'] - finace_Dataframe_select_agg['ROI_AI']
    # print(finace_Dataframe_select_agg)
    finace_Dataframe_select_agg = finace_Dataframe_select_agg[finace_Dataframe_select_agg['error'] > 0]
    account_date = set(finace_Dataframe_select_agg['cnt_at'])
    Failure_sku = pd.DataFrame(columns=['account_date','manual_sales','AI_sales','failure_count','total_count'])
    print(account_date)
    for days in account_date:
        # print(days)
        finace_Dataframe_days = finace_Dataframe_select[finace_Dataframe_select['cnt_at'] == str(days)]
        finace_Dataframe_days['error'] = finace_Dataframe_days['ROI_AI'] - finace_Dataframe_days['ROI_manual']
        failure_sku = pd.DataFrame({
            'account_date': days,
            'manual_sales': '%.2f' % finace_Dataframe_days['Sales_manual'].sum(),
            'AI_sales': '%.2f' % finace_Dataframe_days['Sales_AI'].sum(),
            'failure_count': len(finace_Dataframe_days[finace_Dataframe_days['error'] < 0]),
            'total_count': len(finace_Dataframe_days[finace_Dataframe_days['Sales_manual'] != 0])
        }, index=[0])
        # print(failure_sku)
        Failure_sku = Failure_sku.append(failure_sku)
    # print(Failure_sku)
    Failure_sku.to_csv\
        ('D:/project/P&G/Code/output/evaluation-15/result/Failure_sku'+str(manufacture)+'.csv', encoding="utf_8_sig")








#以下是对ROI好坏的SKU进行分类标签操作
def label_sku(good_sku_Dataframe,bad_sku_Dataframe,manufacture):
    bad_sku_Dataframe_01 = bad_sku_Dataframe.reset_index(drop=True)
    bad_sku_Dataframe_01.drop(['Unnamed: 0'], axis=1, inplace=True)
    bad_sku_Dataframe_02 = bad_sku_Dataframe_01.T
    bad_sku_Dataframe_02.columns = ['count', 'mean', 'std', 'min', '25%','50%','75%','max']
    bad_sku_Dataframe_02['label'] = 0
    good_sku_Dataframe_01 = good_sku_Dataframe.reset_index(drop=True)
    good_sku_Dataframe_01.drop(['Unnamed: 0'], axis=1, inplace=True)
    good_sku_Dataframe_02 = good_sku_Dataframe_01.T
    good_sku_Dataframe_02.columns = ['count', 'mean', 'std', 'min', '25%','50%','75%','max']
    good_sku_Dataframe_02['label'] = 1
    # print(good_sku_Dataframe_02)
    # good_sku_Dataframe_02.to_csv('D:/project/P&G/Code/output/evaluation-15/result/bad_sku_Dataframe_02.csv', encoding="utf_8_sig")
    total_sku= good_sku_Dataframe_02.append(bad_sku_Dataframe_02,ignore_index=False)
    total_sku = total_sku.fillna(0)
    #原来piece_bar_code的数据是索引列，因此以下的操作是将索引列进行更换，先进行了一个重置索引的操作，再对列名进行了修改
    total_sku = total_sku.reset_index()
    total_sku.rename(columns={'index':'piece_bar_code'},inplace=True)
    # print(total_sku)
    total_sku.to_csv('D:/project/P&G/Code/output/evaluation-15/result/total_sku.csv', encoding="utf_8_sig")
    #以下操作是取SKU的销售成本和销售毛利用于特征
    get_sale_sql = """select account_date,piece_bar_code,delivery_qty,cost_tax,transaction_unit_price_tax
    from mid_cj_sales where custom_business_num = %s and delivery_type_name = '销售'""" %(manufacture)
    get_sale = Mysql_Data(get_sale_sql)
    get_sale.columns = (['account_date','piece_bar_code','delivery_qty','cost_tax','transaction_unit_price_tax'])
    get_sale['cost_tax_per'] =  get_sale['cost_tax']/get_sale['delivery_qty']
    get_sale = get_sale.groupby(['piece_bar_code'],as_index=False).mean()
    get_sale.drop(['cost_tax','delivery_qty'], axis=1, inplace=True)
    # get_sale.to_csv('D:/project/P&G/Code/output/evaluation-15/result/get_sale.csv', encoding="utf_8_sig")
    final_SKU = pd.merge(total_sku,get_sale,on=['piece_bar_code'],how='left')
    final_SKU.fillna(0)
    final_SKU.drop(['25%','50%','75%'], axis=1, inplace=True)
    final_SKU.to_csv('D:/project/P&G/Code/output/evaluation-15/result/final_SKU.csv', encoding="utf_8_sig")
    return final_SKU

#以下是通过交叉报表来看下每两个变量之间的关系
def cross_sheet(data,manufacture):
    #以下是看count与label之间的关系
    cross_01 = pd.crosstab(pd.qcut(data['count'], [0,0.25,0.5,0.75]),data['label'])
    #将这个报表可视化
    mosaic(cross_01.stack())
    plt.savefig('D:/project/P&G/Code/output/evaluation-15/result/cross_tab_count' + str(manufacture) + '.jpg',
                dpi=400,
                bbox_inches='tight')
    plt.close()
    cross_02 = pd.crosstab(pd.qcut(data['mean'], [0,0.25,0.5,0.75]),data['label'])
    #将这个报表可视化
    mosaic(cross_02.stack())
    plt.savefig('D:/project/P&G/Code/output/evaluation-15/result/cross_tab_mean' + str(manufacture) + '.jpg',
                dpi=400,
                bbox_inches='tight')
    plt.close()
    cross_03 = pd.crosstab(pd.qcut(data['std'], [0,0.25,0.5,0.75]),data['label'])
    #将这个报表可视化
    mosaic(cross_03.stack())
    plt.savefig('D:/project/P&G/Code/output/evaluation-15/result/cross_tab_std' + str(manufacture) + '.jpg',
                dpi=400,
                bbox_inches='tight')
    plt.close()
    cross_04 = pd.crosstab(pd.qcut(data['transaction_unit_price_tax'], [0,0.25,0.5,0.75]),data['label'])
    #将这个报表可视化
    mosaic(cross_04.stack())
    plt.savefig('D:/project/P&G/Code/output/evaluation-15/result/cross_tab_transaction_unit_price_tax' + str(manufacture) + '.jpg',
                dpi=400,
                bbox_inches='tight')
    plt.close()

#以下是搭建LR回归模型,data就是一个dataframe
#--------------------------------------------------=-=========================================
#以下是搭建逻辑回归模型，并训练模型
def trainModel(data):
    formula = "label ~ count + mean + std + min + max + cost_tax_per + transaction_unit_price_tax"
    model = sm.Logit.from_formula(formula,data = data)
    re = model.fit()
    return re

#分析逻辑回归的模型的统计性质
def modelSummary(re):
    #以下是整体的统计结果的分析
    print(re.summary())
    #用f test看std，count，transaction_unit_price_tax三个系数是否是显著
    print('检验假设std的系数等于0：')
    print(re.f_test('std=0'))
    print('检验假设count的系数等于0：')
    print(re.f_test('count=0'))
    print('检验假设transaction_unit_price_tax的系数等于0：')
    print(re.f_test('transaction_unit_price_tax=0'))

def logitRegression(data):
    #以下是将数据分为训练集合测试集
    trainSet,testSet = train_test_split(data,test_size=0.2)
    #训练模型并分析模型效果
    re = trainModel(trainSet)
    modelSummary(re)
    return  re,testSet


#以下是理解和分析模型的结果,re代表训练好的LR模型
def interpretModel(re):
    conf = re.conf_int()
    #计算各个变量对事件发生比的影响
    #conf里面的3列，分别对应着估计值的下届，上届和估计值本身
    conf['OR'] = re.params
    conf.columns = ['2.5%','97.5%','OR']
    print('各个变量对事件发生比的影响:')
    print(np.exp(conf))
    print('各个变量的边际效应：')
    print(re.get_margeff(at='overall').summary())


#使用训练好的模型对测试数据做预测
def makePrediction(re, testSet, alpha = 0.4):
    #计算事件发生的概率
    testSet['prob'] = re.predict(testSet)
    print('事件发生概率（预测概率）大于0.6的数据个数：')
    print(testSet[testSet['prob'] > 0.6].shape[0])
    print('事件发生概率（预测概率）大于0.5的数据个数：')
    print(testSet[testSet['prob'] > 0.5].shape[0])
    #根据预测的概率，得出最终的预测
    testSet['pred'] = testSet.apply(lambda x: 1 if x['prob'] > alpha else 0, axis =1)
    return testSet


# re,testset = logitRegression(data)
# predict = makePrediction(re,testset)
# print(predict)


#计算预测结果的precision 和recall
def evaluation(re):
    bins =np.array([0,0.4,1])
    label = re['label']
    pred = re['pred']
    tp,fp,fn,tn = np.histogram2d(label,pred,bins=bins)[0].flatten()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 * precision *recall / (precision + recall)
    print('查准率: %3f,查全率： %3f，f1 ：%3f' %(precision,recall,f1))



#定义一个主函数，来调用那些整体的函数
def main_overview(Simulation_file,finance_file,manufacture):
    Simulation,finance = distinguish_manufacture(Simulation_file,finance_file,manufacture)
    overview_sales(finance, Simulation, manufacture)
    evaluation_perday(finance, Simulation, manufacture)
    select_good_SKU(finance, Simulation,manufacture)
    # good_sku_Dataframe = pd.read_csv('D:/project/P&G/Code/output/evaluation-15/result/sum_good_sku_describe'+str(manufacture)+'.csv', skipinitialspace=True)
    # bad_sku_Dataframe = pd.read_csv('D:/project/P&G/Code/output/evaluation-15/result/sum_bad_sku_describe'+str(manufacture)+'.csv', skipinitialspace=True)
    # label_sku(good_sku_Dataframe, bad_sku_Dataframe, 1)
    # final_SKU = label_sku(good_sku_Dataframe, bad_sku_Dataframe,manufacture)
    # cross_sheet(final_SKU,manufacture)
    # #以下是调用LR函数
    # re,testset = logitRegression(final_SKU)
    # predict = makePrediction(re,testset)
    # print(predict)
    # evaluation(predict)
# main_overview('D:/project/P&G/Code/output/evaluation-15/result/final_sheet_revised.csv',
#               'D:/project/P&G/Code/output/evaluation-15/result/calculate_finance.csv',1)

















