# -*- coding = utf-8 -*-
'''
@Time: 2018/11/21 19:21
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


def missing_value( data_frame):
    data_frame_01 = data_frame.fillna(0)
    return data_frame_01

global parame_mysql

parame_mysql ={'host':"192.168.6.122",
                 'port': 3306,
                 'user':"root",
                 'password':"Rpt@123456",
                 'database':"data_cjgm"}

def Mysql_Data(sql_name):
    conn = pymysql.connect(**parame_mysql)
    conn.cursor()
    data_mysql = pd.read_sql(sql_name, conn)
    return data_mysql

#计算物单个SKU的物流费用,order cost = OC
def per_sku_oc():
    sale_sql = """select account_date,piece_bar_code,total_num from cj_sales
                  where DATE_FORMAT(account_date,'%Y%m%d') between ('20180601') and ('20180930')
                    and stock_num = '11A' """
    sale_sql_read = Mysql_Data(sale_sql)
    sale_sql_read.columns = ['account_date','piece_bar_code','total_num']
    sale_sql_read['account_date'] = pd.to_datetime(sale_sql_read['account_date']).dt.normalize()
    sale_sql_read_group = sale_sql_read.groupby(['account_date'],as_index=False)
    sale_sql_read_group = sale_sql_read_group.agg(sum)
    sale_sql_read_group = sale_sql_read_group['total_num'].sum()
    #根据配送费用表手工算出6-9月，四个月的物流总成本为1388345元
    oc_total_fee = 1388345
    per_sku_oc = oc_total_fee /sale_sql_read_group
    return per_sku_oc



#计算每个SKU每日的库存存储和服务成本，先计算2018年的11A仓库的总的数量
#-------------------------------------------------------------------------------------------------
def per_day_sku_sc():
    total_stock = """select cnt_at,manufacturer_num,piece_bar_code,sum_total from cj_inv_history
                    where stock_num = '11A'
                    and DATE_FORMAT(cnt_at,'%Y%m%d') between ('20180501') and ('20180731')"""
    total_stock_read =Mysql_Data(total_stock)
    total_stock_read.columns = ['cnt_at','manufacturer_num','piece_bar_code','sum_total']
    total_stock_read['cnt_at'] = pd.to_datetime(total_stock_read['cnt_at']).dt.normalize()
    total_stock_read_group =total_stock_read.groupby(['cnt_at'],as_index=False).sum()
    total_stock_mean = total_stock_read_group['sum_total'].mean()
    #根据配送费用表手工算出平均每日的仓储和服务成本用是18613元，取2-7月的仓储费用的均值
    per_day_total_fee = 13000
    per_day_sku_sc_fee = per_day_total_fee / total_stock_mean
    return per_day_sku_sc_fee



#设定可以用于mysql日期查询格式的时间
#-----------------------------------------------------------------------------------
def pm_pbc(Data_Frame_01,Data_Frame_02,in_dex):
    if Data_Frame_02.empty == True:
        Data_Frame_02['piece_bar_code'] = Data_Frame_01['piece_bar_code']
        Data_Frame_02 = Data_Frame_02.fillna(0)
        Merge_02 = pd.merge(Data_Frame_01, Data_Frame_02, on=['piece_bar_code'], how=in_dex)
        Merge_02 = Merge_02.fillna(0)
    else:
        Merge_02 = pd.merge(Data_Frame_01, Data_Frame_02,on=['piece_bar_code'], how=in_dex)
        Merge_02 = Merge_02.fillna(0)
    return Merge_02

def pm_pbc_po(Data_Frame_01,Data_Frame_02,in_dex):
    if Data_Frame_02.empty == True:
        Data_Frame_02['piece_bar_code'] = Data_Frame_01['piece_bar_code']
        Data_Frame_02['po_bill_type'] = Data_Frame_01['po_bill_type']
        Data_Frame_02 = Data_Frame_02.fillna(0)
        Merge_02 = pd.merge(Data_Frame_01, Data_Frame_02, on=['piece_bar_code','po_bill_type'], how=in_dex)
    else:
        Data_Frame_02 = Data_Frame_02.fillna(0)
        Merge_02 = pd.merge(Data_Frame_01, Data_Frame_02,on=['piece_bar_code','po_bill_type'], how=in_dex)
    return Merge_02

def pm_pbc_interval(Data_Frame_01,Data_Frame_02,in_dex):
    if Data_Frame_02.empty == True:
        Data_Frame_02['piece_bar_code'] = Data_Frame_01['piece_bar_code']
        Data_Frame_02['interval_time'] = Data_Frame_01['interval_time']
        Data_Frame_02 = Data_Frame_02.fillna(0)
        Merge_02 = pd.merge(Data_Frame_01, Data_Frame_02, on=['piece_bar_code', 'interval_time'], how=in_dex)
    else:
        Data_Frame_02 = Data_Frame_02.fillna(0)
        Merge_02 = pd.merge(Data_Frame_01, Data_Frame_02,on=['piece_bar_code', 'interval_time'], how=in_dex)

    return Merge_02

#此函数，是为了解决创洁实际下单的量，在数据库中会存在一天有很多次的下单行为，这里是进行了一次合并的操作
def process_order(Data_frame):
    if Data_frame.empty != True:
        Data_frame_01 = Data_frame.groupby(['piece_bar_code','po_bill_type'],as_index=False).agg(sum)
    else:
        Data_frame_01 = Data_frame
    return Data_frame_01

def process_order_brfore(Data_frame):
    if Data_frame.empty != True:
        Data_frame_01 = Data_frame.groupby(['piece_bar_code','po_bill_type_before'],as_index=False).agg(sum)
    else:
        Data_frame_01 =Data_frame
    return Data_frame_01