# -*- coding = utf-8 -*-
'''
@Time: 2019/5/13 18:49
@Author: Ye Jinyu
'''


import pandas as pd
import numpy as np
import warnings
import math
from datetime import timedelta
from datetime import date
warnings.filterwarnings("ignore")
import pymysql
import traceback
import sys
import time
import cx_Oracle as cx
import seaborn as sns
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from get_batabase import *
import codecs
import datetime
import time
from datetime import datetime,timedelta
from dateutil.tz import gettz,tzlocal
import math
import multiprocessing
from numba import jit, prange, vectorize
from numba import cuda
from numba import njit
from multiprocessing import Value , Array
import os
import shutil


# 先获取get_bd这个类
get_bd = get_bd()
date_parameter = sys.argv[1]
get_bd.mkdir(date_parameter)


#需要让该py文件的所有参数都加入统一公用的参数date_parameter

date_parameter = sys.argv[1]


def print_in_log(string):
    print(string)
    date_1 = datetime.now()
    str_10 = datetime.strftime(date_1,'%Y-%m-%d')
    file = open('./' + str(date_parameter) + '/' + 'log' + str(str_10) + '.txt', 'a')
    file.write(str(string) + '\n')

#定义函数多线程读取交易节点数据
def multi_read(node_relationship_tran,date_parameter):
    today = str(date_parameter)
    Today_date = (pd.to_datetime(datetime.strptime(today, '%Y%m%d'))).strftime(
        '%Y-%m-%d %H:%M:%S')
    Tomorrow = (pd.to_datetime(datetime.strptime(today, '%Y%m%d')) + timedelta(1) + timedelta(seconds=-1)).strftime(
        '%Y-%m-%d %H:%M:%S')

    pool = multiprocessing.Pool(processes=4)  # 创建4个进程
    results = []
    data = pd.DataFrame(columns=['belong_date','cnt_at','current_status','dem_node_id','nom_resource',
                                 'parent_status','resource_id','response_time','service_level',
                                 'sup_node_id','unit'])
    for j in range(35):
        today = datetime.date(datetime.strptime(date_parameter, '%Y%m%d')) + timedelta(j)
        tomorrow = datetime.date(datetime.strptime(date_parameter, '%Y%m%d')) + timedelta(j+1)+ timedelta(seconds = -1)
        results.append(pool.apply_async(node_relationship_tran,
                                        args=(date_parameter,today, tomorrow, Today_date, Tomorrow,j)))
    pool.close()  # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
    pool.join()  # 等待进程池中的所有进程执行完毕

    for i in results:
        if i.get().empty == True:
            pass
        else:
            a = i.get()
            data = data.append(a, ignore_index=True)
    return data

# #定义函数多线程读取节点需求数据
def multi_read_node_demand(node_demand,date_parameter):
    today = str(date_parameter)
    Today_date = (pd.to_datetime(datetime.strptime(today, '%Y%m%d'))).strftime(
        '%Y-%m-%d %H:%M:%S')
    Tomorrow = (pd.to_datetime(datetime.strptime(today, '%Y%m%d')) + timedelta(1) + timedelta(seconds=-1)).strftime(
        '%Y-%m-%d %H:%M:%S')


    pool = multiprocessing.Pool(processes=4)  # 创建4个进程
    results = []
    data = pd.DataFrame(columns=['dem_cirl_node_id','sup_cirl_node_id','forecast_HY','forecast_season',
                                 'forecast_month','duration_time','unit','demand_qty','resource_id',
                                 'cnt_at','dem_node_id','sup_node_id','demand_date'])
    for j in range(35):
        today = datetime.date(datetime.strptime(date_parameter, '%Y%m%d')) + timedelta(j)
        tomorrow = datetime.date(datetime.strptime(date_parameter, '%Y%m%d')) + timedelta(j+1)+ timedelta(seconds = -1)
        results.append(pool.apply_async(node_demand,
                                        args=(date_parameter,today, tomorrow, Today_date, Tomorrow,j)))

    pool.close()  # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
    pool.join()  # 等待进程池中的所有进程执行完毕
    for i in results:
        if i.get().empty == True:
            pass
        else:
            a = i.get()
            data = data.append(a, ignore_index=True)
    return data

def multi_read_node_demand_without(node_demand_without,date_parameter):
    today = str(date_parameter)
    Today_date = (pd.to_datetime(datetime.strptime(today, '%Y%m%d'))).strftime(
        '%Y-%m-%d %H:%M:%S')
    Tomorrow = (pd.to_datetime(datetime.strptime(today, '%Y%m%d')) + timedelta(1) + timedelta(seconds=-1)).strftime(
        '%Y-%m-%d %H:%M:%S')

    pool = multiprocessing.Pool(processes=4)  # 创建4个进程
    results = []
    data = pd.DataFrame(columns=['dem_cirl_node_id','sup_cirl_node_id','forecast_HY','forecast_season',
                                 'forecast_month','duration_time','unit','demand_qty','resource_id',
                                 'cnt_at','dem_node_id','sup_node_id','demand_date'])
    for j in range(35):
        today = datetime.date(datetime.strptime(date_parameter, '%Y%m%d')) + timedelta(j)
        tomorrow = datetime.date(datetime.strptime(date_parameter, '%Y%m%d')) + timedelta(j+1)+ timedelta(seconds = -1)
        results.append(pool.apply_async(node_demand_without,
                                        args=(date_parameter,today, tomorrow, Today_date, Tomorrow,j)))
    pool.close()  # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
    pool.join()  # 等待进程池中的所有进程执行完毕
    for i in results:
        if i.get().empty == True:
            pass
        else:
            a = i.get()
            data = data.append(a, ignore_index=True)
    return data



#定义函数多线程读取数据
def single_read(node_relationship_tran,date_parameter):
    print_in_log('进入单线程')
    tomorrow = datetime.date(datetime.strptime(date_parameter, '%Y%m%d')) + timedelta(1)
    Today_date = datetime.date(datetime.strptime(date_parameter, '%Y%m%d'))

    data = pd.DataFrame(columns=['belong_date','cnt_at','current_status','dem_node_id','nom_resource',
                                 'parent_status','resource_id','response_time','service_level',
                                 'sup_node_id','unit'])

    print_in_log(str(Today_date)+str(tomorrow)+str(Today_date)+str(Today_date))
    a = node_relationship_tran(Today_date, Today_date, Today_date, Today_date,1)

    return data

'''先是从接口读取数据，将读取接口的数据'''
def get_data_save(date_parameter_transform):
    today = str(date_parameter_transform)
    Today_date = (pd.to_datetime(datetime.strptime(today, '%Y%m%d'))).strftime(
        '%Y-%m-%d %H:%M:%S')
    tomorrow = (pd.to_datetime(datetime.strptime(today, '%Y%m%d')) + timedelta(1) + timedelta(seconds=-1)).strftime(
        '%Y-%m-%d %H:%M:%S')

    print_in_log('当前输入日期为：' + str(Today_date))
    print_in_log('输入有效期的截至日期为：' + str(tomorrow))
    # today_1 = (today + timedelta(days=-40)).strftime('%Y-%m-%d %H:%M:%S')
    # 得到类中的函数，然后再取类中的函数获取节点需求
    '''加入新的函数用于将get_database里面的数据加入创建目录的函数,先创建一个用于存储日志的文件目录'''
    get_bd.mkdir(date_parameter_transform)

    t0 = time.time()
    node_relationship_tran = multi_read(get_bd.node_relationship_tran,date_parameter_transform)
    node_relationship_tran.to_csv('./' + str(date_parameter) + '/' + 'node_relationship_tran'+str(date_parameter_transform)+'.csv',encoding="utf_8_sig")
    del node_relationship_tran
    t1 = time.time()

    print_in_log('node_relationship_tran:'+str(t1 - t0))

    node_demand_without = multi_read_node_demand_without(get_bd.node_demand_without, date_parameter_transform)
    node_demand_without.to_csv('./' + str(date_parameter) + '/' + 'node_demand_without' + str(date_parameter_transform) + '.csv', encoding="utf_8_sig")
    del node_demand_without
    t2 = time.time()
    print_in_log('node_demand_without:'+str(t2 - t1))
    # 得到类中的函数，然后再取类中的函数获取交易节点的节点关系

    node_demand = multi_read_node_demand(get_bd.node_demand,date_parameter_transform)
    node_demand.to_csv('./' + str(date_parameter) + '/' + 'node_demand'+str(date_parameter_transform)+'.csv', encoding="utf_8_sig")
    del node_demand
    # node_relationship_tran = pd.read_csv('node_relationship_tran.csv')
    t3 = time.time()
    print_in_log('node_demand耗时:'+str(t3 - t2))
    # 得到类中的函数，然后再取类中的函数获取流转节点的节点关系
    node_relationship_cirl = get_bd.node_relationship_cirl(date_parameter_transform,Today_date,tomorrow)
    node_relationship_cirl.to_csv('./' + str(date_parameter) + '/' + 'node_relationship_cirl'+str(date_parameter_transform)+'.csv', encoding="utf_8_sig")
    del node_relationship_cirl
    # node_relationship_cirl = pd.read_csv('node_relationship_cirl.csv')
    t4 = time.time()
    print_in_log('node_relationship_cirl耗时:'+str(t4 - t3))

    # 得到类中的函数，然后再取类中的函数获取节点描述
    node_description = get_bd.node_description(date_parameter_transform)
    node_description.to_csv('./' + str(date_parameter) + '/' + 'node_description'+str(date_parameter_transform)+'.csv', encoding="utf_8_sig")
    del node_description
    # node_description = pd.read_csv('node_description.csv')
    t5 = time.time()
    print_in_log('node_description耗时:'+str(t5 - t4))
    #得到类中的函数，然后再取类中的函数获取节点资源（可下单）
    node_resource = get_bd.get_node_resource(date_parameter_transform,Today_date,tomorrow)
    node_resource.to_csv('./' + str(date_parameter) + '/' + 'node_resource'+str(date_parameter_transform)+'.csv', encoding="utf_8_sig")
    del node_resource
    # node_resource = pd.read_csv('node_resource.csv')
    t6 = time.time()
    print_in_log('get_node_resource耗时:'+str(t6 - t5))
    # #得到类中的函数，然后再取类中的函数获取参数调节
    parameter_revised = get_bd.parameter_revised(date_parameter_transform,Today_date,tomorrow)
    parameter_revised.to_csv('./' + str(date_parameter) + '/' + 'parameter_revised'+str(date_parameter_transform)+'.csv', encoding="utf_8_sig")
    del parameter_revised
    # parameter_revised = pd.read_csv('parameter_revised.csv')
    t7 = time.time()
    print_in_log('parameter_revised耗时:'+str(t7 - t6))
    # #业务数据
    business_data = get_bd.business_data(date_parameter_transform,Today_date,tomorrow)
    business_data.to_csv('./' + str(date_parameter) + '/' + 'business_data'+str(date_parameter_transform)+'.csv', encoding="utf_8_sig")
    del business_data
    # business_data = pd.read_csv('business_data.csv')
    t8 = time.time()
    print_in_log('business_data耗时:'+str(t8 - t7))
    # #资源转化
    resource_convert = get_bd.resource_convert(date_parameter_transform)
    resource_convert.to_csv('./' + str(date_parameter) + '/' + 'resource_convert'+str(date_parameter_transform)+'.csv', encoding="utf_8_sig")
    del resource_convert
    # resource_convert = pd.read_csv('resource_convert.csv')
    t9 = time.time()
    print_in_log('resource_convert耗时:'+str(t9 - t8))
    # return node_relationship_tran,node_relationship_cirl,node_demand,\
    #        node_description,node_resource,parameter_revised,business_data,resource_convert

def date_parameter_read():
    date_parameter = sys.argv[1]
    date_parameter_intercept = date_parameter[0:8]
    # date_parameter_intercept = '20190307'
    return date_parameter_intercept


#《============================================================================主函数入口
if __name__ == '__main__':
    try:
        st = time.time()
        date_parameter = date_parameter_read()
        # date_parameter = '20190423'
        get_data_save(date_parameter)
        ed = time.time()
        print_in_log('总程序耗时：'+str(ed - st))
        print("result:1")
    except OSError as reason:
        print_in_log('出错原因是%s'+str(reason))
        print("result:0")
