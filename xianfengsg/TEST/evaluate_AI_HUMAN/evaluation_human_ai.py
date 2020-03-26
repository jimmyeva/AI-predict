# -*- coding: utf-8 -*-
# @Time    : 2019/11/7 9:40
# @Author  : Ye Jinyu__jimmy
# @File    : evaluation_human_ai

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import *
import itertools
import datetime
import os
import pymysql
import copy
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
import math
import warnings
import cx_Oracle


import importlib,sys
importlib.reload(sys)
LANG="en_US.UTF-8"
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

'''该脚本是为了对比人工下单和AI下单的实际对比,时间开始的时间是2019年10月21日'''

#设置函数用来生成和保存每个的对应的日志信息
def mkdir():
    folder = os.path.exists('D:/AI/xianfengsg/TEST/evaluate_AI_HUMAN')
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs('D:/AI/xianfengsg/TEST/evaluate_AI_HUMAN')  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("----生成新的文件目录----")
    else:
        print(
            "当前文件夹已经存在")


def print_in_log(string):
    print(string)
    file = open('D:/AI/xianfengsg/TEST/evaluate_AI_HUMAN/log.txt', 'a')
    file.write(str(string) + '\n')

def print_in_text(string):
    print(string)
    file = open('D:/AI/xianfengsg/TEST/evaluate_AI_HUMAN/text.txt', 'a+')
    file.write(str(string) + '\n')


#--------------------------获取oracle主库里面实际下单的情况
def get_real_orders(yes_date,today):
    dbconn = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com", database="purchare_sys",
                                 user="purchare_sys",password="purchare_sys@123",port = 3306,
                                 charset='utf8')
    print_in_log('采购2.0，MYSQL数据库连接成功,正在获取人工的补货数据')
    get_orders = """SELECT DATE_FORMAT(e.create_time,'%%Y-%%m-%%d') Account_date,e.received_warehouse_name Warehouse_name ,
                    a.group_name ,a.goods_code Code,
                    case when e.is_urgent=0 then '不是' when e.is_urgent=1 then '是' end urgent,sum(a.amount) real_orders
                    from p_warehouse_order_dtl a 
                    left join p_warehouse_order e on e.id=a.warehouse_order_id
                    LEFT JOIN (
                    select b.warehouse_order_id,b.warehouse_order_dtl_id,a.plan_order_id id,b.id dtlid from p_purchase_plan_order a,p_purchase_plan_order_dtl b
                    where a.plan_order_id=b.p_purchase_plan_order_id
                    ) b on a.warehouse_order_id=b.warehouse_order_id and a.id=b.warehouse_order_dtl_id
                    where a.amount<>0 AND e.create_time > date ('%s') AND e.create_time < date ('%s') AND 
                    e.received_warehouse_name='杭州配送商品仓'
                    group by DATE_FORMAT(e.create_time,'%%Y-%%m-%%d'),e.received_warehouse_name,a.group_name,
                    a.goods_code,a.goods_name,e.is_urgent""" \
                 %(yes_date,today)
    orders= pd.read_sql(get_orders,dbconn)
    orders['Code'] = orders['Code'].astype(str)
    # def polishing(x):
    #     x['Code'].rjust(5, '0')
    # orders['Code'] = orders.apply(lambda x: polishing(x), axis = 1)
    dbconn.close()
    print_in_log('人工的补货数据读取完成，并关闭了服务器的连接')
    return orders


#-----------------------------------------该函数是用来获取AI的补货数据
def get_AI_replenishment(today_date):
    print_in_log('连接到mysql服务器...')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123", port=3306, charset='utf8')
    print_in_log('连接成功,正在获取AI的补货建议')
    replenishment_sql = """SELECT Sku_id,Code,Sku_name ,rate 装箱规格,Forecast_box 预测的箱数,
    SS 安全库存,Stock 现有库存箱数,Suggestion_qty 建议补货箱数
     FROM dc_replenishment where Account_date = DATE('%s')""" % (today_date)
    db.cursor()
    read_replenishment_sql = pd.read_sql(replenishment_sql, db)
    read_replenishment_sql['Code'] = read_replenishment_sql['Code'].astype(str)
    read_replenishment_sql['Sku_id'] = read_replenishment_sql['Sku_id'].astype(str)
    def polishing(x):
        return x['Code'].rjust(5, '0')
    read_replenishment_sql['Code'] = read_replenishment_sql.apply(lambda x: polishing(x), axis=1)
    read_replenishment_sql['AI_建议补货'] = read_replenishment_sql['建议补货箱数'] * read_replenishment_sql['装箱规格']
    read_replenishment_sql['现有库存最小单位'] = read_replenishment_sql['现有库存箱数'] * read_replenishment_sql['装箱规格']
    read_replenishment_sql['AI补货后的库存'] = read_replenishment_sql['AI_建议补货'] + read_replenishment_sql['现有库存最小单位']
    db.close()
    print_in_log('AI的补货建议读取完成，并关闭了服务器的连接')
    return read_replenishment_sql


#设置程序读取设定日期的门店实际的销售量
def get_store_sales_qty(end_day,TDAT_day):
    print_in_log('连接到mysql服务器...')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123", port=3306, charset='utf8')
    print_in_log('连接成功,正在获取门店的实际销售数据')
    store_sales_sql = """SELECT sh.GDGID Sku_id,sh.Sku_name,sh.qty 门店销售最小规格数量
    FROM sales_his sh WHERE sh.Ocrdate= DATE('%s')""" % (TDAT_day)
    db.cursor()
    store_sales = pd.read_sql(store_sales_sql, db)
    store_sales['Sku_id'] = store_sales['Sku_id'].astype(str)
    # store_sales['Ocrdate'] = pd.to_datetime(end_day)
    db.close()
    print_in_log('AI的补货建议读取完成，并关闭了服务器的连接')
    return store_sales




#====================================================设置主程序用于读取设置自定义的时间进行
def main_function(repl_day,tomorrow,TDAT_day):
    data_AI = get_AI_replenishment(repl_day)
    data_human = get_real_orders(repl_day,tomorrow)
    store_sales = get_store_sales_qty(repl_day,TDAT_day)
    result_01 = pd.merge(data_AI,store_sales,on=['Sku_id','Sku_name'],how='right')
    result = pd.merge(result_01,data_human,on=['Code'],how='left')
    result['real_orders'] = result['real_orders'].fillna(0)
    result['人工补货数量'] = result['real_orders'] * result['装箱规格']
    result['人工补货后的总库存'] = result['人工补货数量'] + result['现有库存最小单位']
    result['AI_剩余库存'] = result['AI补货后的库存'] - result['门店销售最小规格数量']
    result['人工_剩余库存'] = result['人工补货后的总库存'] - result['门店销售最小规格数量']
    result.to_csv('D:/AI/xianfengsg/TEST/evaluate_AI_HUMAN/result'+str(repl_day)+'.csv',
                         encoding='utf_8_sig',index=False)



#===============================用于循环2019年10月21日2019年11月7日的预测和决策的数据
# def start_end_date(start, end):
#     for days in tqdm(pd.date_range(start, end)):
#         today = days.strftime('%Y%m%d')
#         tomorrow = (days + datetime.timedelta(1)).strftime('%Y%m%d')
#         TDAT_day = (days + datetime.timedelta(2)).strftime('%Y%m%d')
#         main_function(today,tomorrow,TDAT_day)
#         print('正在进行forecast和decision的日期是：'+str(today))
#
#
# start_end_date('20191021','20191107')


#-----------------------------------------需要再加入商品的价格和毛利情况,分为获取最小规格的进价和最小规格的售价
def get_pp_sp(code_7th):
    print('连接到mysql服务器...,正在读取进价数据')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123", port=3306, charset='utf8')
    get_pp_sql = """SELECT pcr.box_gauge MUNIT,pcr.price PRICE FROM p_call_record pcr WHERE
     pcr.goods_id='%s' AND pcr.warehouse_id ='1000255'""" % (code_7th)
    db.cursor()
    get_pp = pd.read_sql(get_pp_sql, db)
    db.close()
    get_pp['PRICE'] = get_pp['PRICE'].astype(float)
    get_pp['MUNIT'] = get_pp['MUNIT'].astype(float)
    if get_pp.empty == False:
        get_pp['purchase_price'] = get_pp['PRICE'] / get_pp['MUNIT']
        pp = get_pp['purchase_price'].iloc[0]
    else:
        pp = 1
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    # 读取的数据包括销量的时间序列，天气和活动信息
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", parameters)
    get_sp = """SELECT a.IMPPRC MBRPRC FROM (SELECT * FROM RPGGD r WHERE r.gdgid =
     '%s' ORDER BY r.LSTUPDTIME DESC)a where ROWNUM=1 """ %(code_7th)

    sp = pd.read_sql(get_sp,conn)
    if sp.empty==False:
        sales_price = sp['MBRPRC'].iloc[0]
    else:
        sales_price = 1
    print(code_7th,pp,sales_price)
    data = pd.DataFrame({'Sku_id':[code_7th],'pp':[pp],'sp':[sales_price]})
    return data




#------------------------定义计算实际的销售数量
def sales_calculate(data):
    def real_sales_AI(x):
        if x['门店销售最小规格数量'] <= x['AI补货后的库存']:
            return x['门店销售最小规格数量']
        else:
            return x['AI补货后的库存']
    def real_sales_huaman(x):
        if x['门店销售最小规格数量'] <= x['人工补货后的总库存']:
            return x['门店销售最小规格数量']
        else:
            return x['人工补货后的总库存']
    data['AI_实际销售'] = data.apply\
        (lambda x: real_sales_AI(x),axis=1)
    data['人工_实际销售'] = data.apply\
        (lambda x: real_sales_huaman(x),axis=1)
    data['AI_实际库存剩余'] = data['AI补货后的库存'] - data['AI_实际销售']
    data['人工_实际库存剩余'] = data['人工补货后的总库存'] - data['人工_实际销售']
    return data

#-------------------为了保证计算顺利，将所有需要计算的列进行格式转换
def data_formative(data):
    data['现有库存最小单位'].astype(float)
    data['AI_建议补货'].astype(float)
    data['AI_实际销售'].astype(float)
    data['pp'].astype(float)
    data['人工补货数量'].astype(float)
    data['人工_实际销售'].astype(float)
    data['AI_实际库存剩余'].astype(float)
    data['人工_实际库存剩余'].astype(float)
    data['sp'].astype(float)
    data['人工补货数量'].astype(float)
    data['人工补货后的总库存'].astype(float)
    data['AI补货后的库存'].astype(float)
    data['AI_剩余库存'].astype(float)
    data['人工_剩余库存'].astype(float)
    return data


#---------------------------定义计算相应的财务指标
def finance_calculate(data):
    data['ROCC_AI'] = (data['现有库存最小单位'] + data['AI_建议补货'] - data['AI_实际销售']) * data['pp'] * 0.008 \
                      + data['AI_实际销售']*  data['pp']
    data['ROCC_人工'] = (data['现有库存最小单位'] + data['人工补货数量'] - data['人工_实际销售']) * data['pp'] * 0.008 \
                      + data['人工_实际销售']*  data['pp']
    data['stock_AI'] = ((data['现有库存最小单位'] + data['AI_实际库存剩余']) / 2) * 0.1
    data['stock_人工'] = ((data['现有库存最小单位'] + data['人工_实际库存剩余']) / 2) * 0.1
    def AI_SC_judge(x):
        if x['AI_剩余库存'] >= 0:
            return 0
        else:
            SC_AI = (x['AI_剩余库存'] * x['sp']) * (-1.07)
            return SC_AI

    def HUMAN_SC_judge(x):
        if x['人工_剩余库存'] >= 0:
            return 0
        else:
            SC_AI = (x['人工_剩余库存'] * x['sp']) * (-1.07)
            return SC_AI
    data['SC_AI'] = data.apply(lambda x: AI_SC_judge(x),axis=1)
    data['SC_人工'] = data.apply(lambda x: HUMAN_SC_judge(x), axis=1)
    data['VL_AI'] = data['AI_实际库存剩余'] * 1.07 * data['pp']
    data['VL_人工'] = data['人工_实际库存剩余'] * 1.07 * data['pp']
    data['Profit_AI'] = data['AI_实际销售'] * (data['sp'] - data['pp'])
    data['Profit_人工'] = data['人工_实际销售'] * (data['sp'] - data['pp'])
    data['成本_AI'] = data['ROCC_AI'] + data['stock_AI'] + data['SC_AI'] + data['VL_AI']
    data['成本_人工'] = data['ROCC_人工'] + data['stock_人工'] + data['SC_人工'] + data['VL_人工']
    data['成本_人工'] = data['ROCC_人工'] + data['stock_人工'] + data['SC_人工'] + data['VL_人工']
    data_final = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)].dropna()
    return data_final

#------------------------------------用一个函数用来封装，实际销售和财务指标的计算
def merge_function(data):
    data = data.fillna(0)
    data_mid = sales_calculate(data)
    data_mid = data_formative(data_mid)
    print(data_mid)
    result = finance_calculate(data_mid)
    return result


#----------------------------设置函数，拿到每天的直观数据并保存到日志的文件中，，'%.2f' %(a/b)
def indicator_finance(data):
    cost_AI = round((data['成本_AI'].sum()),4)
    cost_human = round((data['成本_人工'].sum()),4)
    Profit_AI = round((data['Profit_AI'].sum()),4)
    Profit_human = round((data['Profit_人工'].sum()),4)

    ROI_AI = round((Profit_AI/cost_AI),4)
    ROI_human = round((Profit_human/cost_human),4)
    ITO_AI = round((2 * data['AI_实际销售'].sum())/(data['现有库存最小单位'].sum() +data['AI_实际库存剩余'].sum()),4)
    ITO_human = round((2 * data['人工_实际销售'].sum())/(data['现有库存最小单位'].sum() + data['人工_实际库存剩余'].sum()),4)
    ROI_AI_BETTER_HUMAN = round(((ROI_AI-ROI_human)/ROI_human),4)
    ITO_AI_BETTER_HUMAN = round(((ITO_AI-ITO_human)/ITO_human),4)
    cost_AI_BETTER_HUMAN = round((cost_human - cost_AI),4)
    profit_AI_BETTER_HUAM = round((Profit_AI - Profit_human),4)
    return cost_AI,cost_human,Profit_AI,Profit_human,ROI_AI,ROI_human,ITO_AI,ITO_human,\
           ROI_AI_BETTER_HUMAN,ITO_AI_BETTER_HUMAN,cost_AI_BETTER_HUMAN,profit_AI_BETTER_HUAM






def start_end_date_finance(start, end):
    data_record = pd.DataFrame()
    for days in tqdm(pd.date_range(start, end)):
        today = days.strftime('%Y%m%d')
        result = pd.read_csv('D:/AI/xianfengsg/TEST/evaluate_AI_HUMAN/result' + str(today) + '.csv',
                      encoding='utf_8_sig')
        sp_pp_data = pd.DataFrame()
        code_7th = set(result['Sku_id'])
        for code in tqdm(code_7th):
            data = get_pp_sp(code)
            sp_pp_data = sp_pp_data.append(data)

        final_data = pd.merge(result,sp_pp_data,on='Sku_id',how='left')
        final = merge_function(final_data)
        cost_AI, cost_human, Profit_AI, Profit_human, ROI_AI, ROI_human, ITO_AI, \
        ITO_human,ROI_AI_BETTER_HUMAN,ITO_AI_BETTER_HUMAN,cost_AI_BETTER_HUMAN,profit_AI_BETTER_HUAM\
            = indicator_finance(final)
        print_in_text(str(today)+'的评估指标信息如下：1. AI补货的总利润是：'+ str(Profit_AI) + ';人工补货的总利润是：'+str(Profit_human)+
                     '\n'
                     + ';AI的ROI是：'+str(ROI_AI)+'；人工的ROI是：'+str(ROI_human)+';AI的ROI比人工高'+str(ROI_AI_BETTER_HUMAN)+
                     '\n'
                     +';AI的利润比人工高'+str(profit_AI_BETTER_HUAM)+';AI的成本节省：'+str(cost_AI_BETTER_HUMAN)+'\n'+
                     '元;AI的库存周转率比人工提高：'+str(ITO_AI_BETTER_HUMAN)+'。')
        data_record_mid = pd.DataFrame({'date':[today],'cost_AI':[cost_AI],'cost_human':[cost_human],'Profit_AI':[Profit_AI],
                                        'Profit_human':[Profit_human],'ROI_AI':[ROI_AI],'ROI_human':[ROI_human],
                                        'ITO_AI':[ITO_AI],'ITO_human':[ITO_human],'ROI_AI_BETTER_HUMAN':[ROI_AI_BETTER_HUMAN],
                                        'ITO_AI_BETTER_HUMAN':[ITO_AI_BETTER_HUMAN],'cost_AI_BETTER_HUMAN':[cost_AI_BETTER_HUMAN],
                                        'profit_AI_BETTER_HUAM':[profit_AI_BETTER_HUAM]})
        data_record = data_record.append(data_record_mid)
        final.to_csv('D:/AI/xianfengsg/TEST/evaluate_AI_HUMAN/final' + str(today) + '.csv',
                      encoding='utf_8_sig',index=False)
    data_record.to_csv('D:/AI/xianfengsg/TEST/evaluate_AI_HUMAN/data_record.csv',
             encoding='utf_8_sig',index=False)


# start_end_date_finance('20191021','20191107')




#================================================================第三步骤，获取那些AI完全好于人工的的SKU进行计算
#-------------------------先定义函数拿到那些只有AI比人工好的SKU
def retain_AI_good(data):
    data['good'] = data['Profit_AI'] - data['Profit_人工']
    result = data[data['good'] >= 0]
    # data['better'] = data[''] - data['SC_人工']
    result = result.drop(columns='good',axis=1,inplace=False)
    return result



#
def get_better_than_human(start, end):
    data_good_record = pd.DataFrame()
    for days in tqdm(pd.date_range(start, end)):
        today = days.strftime('%Y%m%d')
        final = pd.read_csv('D:/AI/xianfengsg/TEST/evaluate_AI_HUMAN/final' + str(today) + '.csv',
                      encoding='utf_8_sig')
        result = retain_AI_good(final)
        cost_AI, cost_human, Profit_AI, Profit_human, ROT_AI, ROT_human, ITO_AI, \
        ITO_human,ROI_AI_BETTER_HUMAN,ITO_AI_BETTER_HUMAN,cost_AI_BETTER_HUMAN,profit_AI_BETTER_HUAM\
            = indicator_finance(result)
        print_in_text(str(today)+'的评估指标信息如下：1. AI补货的总利润是：'+ str(Profit_AI) + ';人工补货的总利润是：'+str(Profit_human)+
                     '\n'
                     + ';AI的ROI是：'+str(ROT_AI)+'；人工的ROI是：'+str(ROT_human)+';AI的ROI比人工高'+str(ROI_AI_BETTER_HUMAN)+
                     '\n'
                     +';AI的利润比人工高'+str(profit_AI_BETTER_HUAM)+';AI的成本节省：'+str(cost_AI_BETTER_HUMAN)+'\n'+
                     '元;AI的库存周转率比人工提高：'+str(ITO_AI_BETTER_HUMAN)+'。')
        data_record_mid = pd.DataFrame({'date':[today],'cost_AI':[cost_AI],'cost_human':[cost_human],'Profit_AI':[Profit_AI],
                                        'Profit_human':[Profit_human],'ROT_AI':[ROT_AI],'ROT_human':[ROT_human],
                                        'ITO_AI':[ITO_AI],'ITO_human':[ITO_human],'ROI_AI_BETTER_HUMAN':[ROI_AI_BETTER_HUMAN],
                                        'ITO_AI_BETTER_HUMAN':[ITO_AI_BETTER_HUMAN],'cost_AI_BETTER_HUMAN':[cost_AI_BETTER_HUMAN],
                                        'profit_AI_BETTER_HUAM':[profit_AI_BETTER_HUAM]})
        data_good_record = data_good_record.append(data_record_mid)
        result.to_csv('D:/AI/xianfengsg/TEST/evaluate_AI_HUMAN/result_good' + str(today) + '.csv',
                      encoding='utf_8_sig',index=False)

    data_good_record.to_csv('D:/AI/xianfengsg/TEST/evaluate_AI_HUMAN/data_good_record.csv',
             encoding='utf_8_sig',index=False)


# get_better_than_human('20191021','20191107')



#===================================================定义函数获取那些每日补货都优于人工的SKU
#---------------------------------------定义函数用来补货每个sku的名称
def get_sku_name(code_7th):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    # 读取的数据包括销量的时间序列，天气和活动信息
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", parameters)
    get_name_sql = """SELECT G.GID Sku_id,G.NAME FROM GOODS g WHERE g.GID='%s'""" %(code_7th)
    get_name = pd.read_sql(get_name_sql,conn)
    sku_name = get_name['NAME'].iloc[0]
    return sku_name


def get_better_sku(start, end):
    good_sku = pd.DataFrame()
    for days in tqdm(pd.date_range(start, end)):
        print(days)
        today = days.strftime('%Y%m%d')
        result_good = pd.read_csv('D:/AI/xianfengsg/TEST/evaluate_AI_HUMAN/result_good' + str(today) + '.csv',
                      encoding='utf_8_sig')
        if today == start:
            sku_data = result_good[['Sku_id']]
            good_sku = good_sku.append(sku_data)
        else:
            sku_data = result_good['Sku_id']
            good_sku = pd.merge(sku_data,good_sku,on='Sku_id',how='inner')
    good_sku = good_sku.drop_duplicates(keep='first',inplace=False)
    good_sku = good_sku.reset_index(drop=True)
    good_sku['name'] = '无数据'
    for i in tqdm(range(len(good_sku))):
        sku_gid = good_sku['Sku_id'].iloc[i]
        sku_name = get_sku_name(sku_gid)
        good_sku['name'].iloc[i] = sku_name
    good_sku.to_csv('D:/AI/xianfengsg/TEST/evaluate_AI_HUMAN/good_sku.csv',
             encoding='utf_8_sig',index=False)

get_better_sku('20191021','20191107')

#===========================================================第四维度分析，针对每个sku分别的成本降低与其他财务指标
def get_each_sku_value(start, end):
    good_sku = pd.read_csv('D:/AI/xianfengsg/TEST/evaluate_AI_HUMAN/good_sku.csv',
             encoding='utf_8_sig')
    sku_list = good_sku['Sku_id'].to_list()
    sku_name = good_sku['name'].to_list()
    good_sku_finance = pd.DataFrame()
    for i in tqdm(range(len(good_sku['Sku_id']))):
        id = sku_list[i]
        name = sku_name[i]
        data_sku = pd.DataFrame()
        for days in tqdm(pd.date_range(start, end)):
            print(days)
            today = days.strftime('%Y%m%d')
            result_good = pd.read_csv('D:/AI/xianfengsg/TEST/evaluate_AI_HUMAN/result_good' + str(today) + '.csv',
                                      encoding='utf_8_sig')
            good_data = result_good[result_good['Sku_id'] == id]
            data_sku = data_sku.append(good_data)
        cost_AI, cost_human, Profit_AI, Profit_human, ROT_AI, ROT_human, ITO_AI, \
        ITO_human,ROI_AI_BETTER_HUMAN,ITO_AI_BETTER_HUMAN,cost_AI_BETTER_HUMAN,profit_AI_BETTER_HUAM\
            = indicator_finance(data_sku)
        #记录每个sku，每日的基本数据
        data_sku.to_csv('D:/AI/xianfengsg/TEST/evaluate_AI_HUMAN/result_good' + str(name) + '.csv',
                                      encoding='utf_8_sig',index=False)
        data_good_record = pd.DataFrame(
            {'sku_id':[id],'name':[name],'cost_AI': [cost_AI], 'cost_human': [cost_human], 'Profit_AI': [Profit_AI],
             'Profit_human': [Profit_human], 'ROT_AI': [ROT_AI], 'ROT_human': [ROT_human],
             'ITO_AI': [ITO_AI], 'ITO_human': [ITO_human], 'ROI_AI_BETTER_HUMAN': [ROI_AI_BETTER_HUMAN],
             'ITO_AI_BETTER_HUMAN': [ITO_AI_BETTER_HUMAN], 'cost_AI_BETTER_HUMAN': [cost_AI_BETTER_HUMAN],
             'profit_AI_BETTER_HUAM': [profit_AI_BETTER_HUAM]})
        good_sku_finance = good_sku_finance.append(data_good_record)
        #用于记录每个sku的基本指标

        data_good_record.to_csv('D:/AI/xianfengsg/TEST/evaluate_AI_HUMAN/good_sku'+str(name)+'_'+ str(id) +'.csv',
                                encoding='utf_8_sig')
    good_sku_finance.to_csv('D:/AI/xianfengsg/TEST/evaluate_AI_HUMAN/good_sku_total.csv',
                            encoding='utf_8_sig')



get_each_sku_value('20191021','20191107')













#用于简单的测试，先将历史的仓库的出库数据从本地读取并存入到mysql的数据库中

def connectdb():
    print_in_log('连接到mysql服务器...')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123",port=3306, charset='utf8')
    print_in_log('连接成功')
    return db

#《-------------------------------------------------------------------------------------删除重复日期数据
def drop_data(db,yes_date):
    cursor = db.cursor()
    sql = """delete from sales_his where Ocrdate = DATE ('%s')"""%(yes_date)
    cursor.execute(sql)
    print('历史重复数据删除成功')


def insert_sales_db(db,data):
    cursor = db.cursor()
    data_list = data.values.tolist()
    print(data_list)
    print_in_log(str(len(data_list)))
    sql = """INSERT INTO sales_his (SENDER,DC_NAME,GID,
    WAREHOUSE_NAME,NUM,GDGID,SKU_NAME,OCRDATE,CRTOTAL,MUNIT
    QTY,QTYSTR,TOTAL,PRICE,QPC,RTOTAL)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    try:
        cursor.executemany(sql, data_list)
        print_in_log("所有历史数据的sku数据插入数据库成功")
        db.commit()
    except OSError as reason:
        print_in_log('出错原因是%s' % str(reason))
        db.rollback()


#<=============================================================================关闭
def closedb(db):
    db.close()

def main():
    data = pd.read_excel('D:/jimmy-ye/AI/AI_supply_chain/V1.0/result_start_up_10.21/sales_DC.xlsx',
                         encoding='utf_8_sig',converters={u'Ocrdate': str})
    print(data)
    db = connectdb()
    # print_in_log('历史销售数据的总长度'+str(len(sales_his)))
    drop_data(db)
    if data.empty:
        print_in_log("The data frame is empty")
        print_in_log("result:1")
        closedb(db)
    else:
        insert_sales_db(db,data)
        closedb(db)
        print_in_log("result:1")
        print_in_log("result:所有数据插入mysql数据库成功")







