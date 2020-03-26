# -*- coding = utf-8 -*-
'''
@Time: 2019/5/20 17:25
@Author: Ye Jinyu
'''
# -*- coding = utf-8 -*-


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

# 先获取get_bd这个类
get_bd = get_bd()

date_parameter = sys.argv[1]
get_bd.mkdir(date_parameter)


def print_in_log(string):
    print(string)
    date_1 = datetime.now()
    str_10 = datetime.strftime(date_1, '%Y-%m-%d')
    file = open('./' + str(date_parameter) + '/' + 'log' + str(str_10) + '.txt', 'a')
    file.write(str(string) + '\n')


'''先是从接口读取数据，将读取接口的数据'''
def get_data(date_parameter_transform):
    tomorrow = datetime.date(datetime.strptime(date_parameter_transform, '%Y%m%d')) + timedelta(hours=24)
    Today_date = datetime.date(datetime.strptime(date_parameter_transform, '%Y%m%d')) + timedelta(hours=0)
    print_in_log('当前输入日期为：' + str(Today_date))
    print_in_log('输入有效期的截至日期为：' + str(tomorrow))
    # today_1 = (today + timedelta(days=-40)).strftime('%Y-%m-%d %H:%M:%S')

    t0 = time.time()
    # 得到类中的函数，然后再取类中的函数获取交易节点的节点关系
    # node_relationship_tran = multi_read(get_bd.node_relationship_tran,date_parameter_transform)
    # node_relationship_tran.to_csv('node_relationship_tran.csv',encoding="utf_8_sig")
    node_relationship_tran = pd.read_csv('./' + str(date_parameter) + '/' +
                                         'node_relationship_tran'+str(date_parameter_transform)+'.csv')
    t1 = time.time()
    print_in_log('multi_read耗时:'+ str(t1 - t0))
    # 得到类中的函数，然后再取类中的函数获取流转节点的节点关系
    # node_relationship_cirl = get_bd.node_relationship_cirl(Today_date,tomorrow)
    # node_relationship_cirl.to_csv('node_relationship_cirl.csv', encoding="utf_8_sig")
    node_relationship_cirl = pd.read_csv('./' + str(date_parameter) + '/' +
                                         'node_relationship_cirl'+str(date_parameter_transform)+'.csv')
    t2 = time.time()
    print_in_log('node_relationship_cirl耗时:'+ str(t2 - t1))
    # 得到类中的函数，然后再取类中的函数获取节点需求
    # node_demand = multi_read_node_demand(get_bd.node_demand,date_parameter_transform)
    # node_demand.to_csv('node_demand.csv', encoding="utf_8_sig")
    node_demand = pd.read_csv('./' + str(date_parameter) + '/' +
                              'node_demand'+str(date_parameter_transform)+'.csv')
    t3 = time.time()
    print_in_log('node_demand耗时:'+ str(t3 - t2))
    # 得到类中的函数，然后再取类中的函数获取节点描述
    # node_description = get_bd.node_description()
    # node_description.to_csv('node_description.csv', encoding="utf_8_sig")
    node_description = pd.read_csv('./' + str(date_parameter) + '/' +
                                   'node_description'+str(date_parameter_transform)+'.csv')
    t4 = time.time()
    print_in_log('node_description耗时:'+ str(t4 - t3))
    #得到类中的函数，然后再取类中的函数获取节点资源（可下单）
    # node_resource = get_bd.get_node_resource(Today_date,tomorrow)
    # node_resource.to_csv('node_resource.csv', encoding="utf_8_sig")
    node_resource = pd.read_csv('./' + str(date_parameter) + '/' +
                                'node_resource'+str(date_parameter_transform)+'.csv')
    t5 = time.time()
    print_in_log('get_node_resource耗时:'+ str(t5 - t4))
    # #得到类中的函数，然后再取类中的函数获取参数调节
    # parameter_revised = get_bd.parameter_revised(Today_date,tomorrow)
    # parameter_revised.to_csv('parameter_revised.csv', encoding="utf_8_sig")
    parameter_revised = pd.read_csv('./' + str(date_parameter) + '/' +
                                    'parameter_revised'+str(date_parameter_transform)+'.csv')
    t6 = time.time()
    print_in_log('parameter_revised耗时:'+ str(t6 - t5))
    # #业务数据
    # business_data = get_bd.business_data(Today_date,tomorrow)
    # business_data.to_csv('business_data.csv', encoding="utf_8_sig")
    business_data = pd.read_csv('./' + str(date_parameter) + '/' +
                                'business_data'+str(date_parameter_transform)+'.csv')
    t7 = time.time()
    print_in_log('business_data耗时:'+ str(t7 - t6))
    # #资源转化
    # resource_convert = get_bd.resource_convert()
    # resource_convert.to_csv('resource_convert.csv', encoding="utf_8_sig")
    resource_convert = pd.read_csv('./' + str(date_parameter) + '/' +
                                   'resource_convert'+str(date_parameter_transform)+'.csv')
    t8 = time.time()
    print_in_log('resource_convert耗时:' + str(t8 - t7))
    node_demand_without = pd.read_csv('./' + str(date_parameter) + '/' +
                                      'node_demand_without'+str(date_parameter_transform)+'.csv')
    t3 = time.time()
    print_in_log('node_demand耗时:'+ str(t3 - t2))
    return node_relationship_tran,node_relationship_cirl,node_demand,\
           node_description,node_resource,parameter_revised,business_data,\
           resource_convert,node_demand_without



'''#先对节点的需求进行处理操作
#------------------------------------------------------------------------------------
#需要输入节点需求，节点关系（交易）/（流转），用于筛掉那些因为存续期问题而不用考虑的需求
'''

#对最后给的输出需要再进行一次修正，即对于那些没有预测信息的数据需要再进行一次合并
def without_prediction(dem_node,node_demand,node_resource,node_relationship_tran,
                       node_relationship_cirl,resource_convert,business_data):
    #对于那些节点需求是负值的资源，这里的是未给出需求的
    node_demand_without = node_demand[node_demand['demand_qty'] < 0]
    node_demand_without['sup_node_id'] = node_demand_without['sup_node_id'].apply(int)
    node_demand_without = node_demand_without[node_demand_without['sup_node_id'] == dem_node]
    node_demand_without = node_demand_without[['resource_id']]
    #对节点需求里面的资源id进行去重
    node_demand_without = node_demand_without.drop_duplicates()
    #这是时候再找节点资源，找到对应的资源id，并计算可用库存等
    node_resource_1 = node_resource[['tran_node_id','resource_id','available_resource_qty','cirl_node_id']]
    node_resource_1 = node_resource_1.rename(index=str, columns={"tran_node_id": "dem_tran_node",
                                                             "cirl_node_id": "dem_cirl_node"})
    node_resource_1['resource_id'] = node_resource_1['resource_id'].apply(int)
    node_demand_without['resource_id'] = node_demand_without['resource_id'].apply(int)
    #进行资源和合并
    node_resource_2 = pd.merge(node_resource_1,node_demand_without,on=['resource_id'],how='inner')
    node_resource_mid = node_resource_2.groupby(['resource_id','dem_tran_node']).head(1)
    #找到对应为进行预测的供需
    print_in_log('此dataframe是未给出节点需求的，资源id，包含了节点资源')
    print_in_log('此dataframe的长度为' + str(len(node_resource_mid)))
    # print_in_log(node_resource_mid)
    #接下来是需要找到对应的供方交易节点
    node_relationship_tran_mid = node_relationship_tran[['sup_node_id','dem_node_id','resource_id']]
    node_relationship_tran_mid = node_relationship_tran_mid.drop_duplicates()
    node_relationship_tran_mid = node_relationship_tran_mid.rename(index=str,
                                                                   columns={"sup_node_id": "sup_tran_node",
                                                                            'dem_node_id':'dem_tran_node'})
    print_in_log('此dataframe是交易节点的预处理')
    print_in_log('此dataframe的长度为' + str(len(node_relationship_tran_mid)))
    # print_in_log(node_relationship_tran_mid)

    node_resource_mid[['resource_id', 'dem_tran_node']] = node_resource_mid[['resource_id', 'dem_tran_node']].astype(int)
    node_relationship_tran_mid[['resource_id', 'dem_tran_node']] = node_relationship_tran_mid[['resource_id', 'dem_tran_node']].astype(int)

    node_resource_mid = pd.merge(node_resource_mid,node_relationship_tran_mid,on=['dem_tran_node','resource_id'],how='inner')
    print_in_log('此dataframe是未给出节点需求的，资源id，包含了交易节点的合并')
    print_in_log('此dataframe的长度为' + str(len(node_resource_mid)))
    # print_in_log(node_resource_mid)
    #同理通道对应的流转节点的
    node_relationship_cirl_mid = node_relationship_cirl[['sup_node_id','dem_node_id']]
    node_relationship_cirl_mid = node_relationship_cirl_mid.rename(index=str, columns={"sup_node_id": "sup_cirl_node",
                                                                                       'dem_node_id':'dem_cirl_node'})
    node_relationship_cirl_mid = node_relationship_cirl_mid.drop_duplicates()
    print_in_log('此dataframe是流转节点的预处理')
    print_in_log('此dataframe的长度为' + str(len(node_relationship_cirl_mid)))
    node_relationship_cirl_mid['dem_cirl_node'] = node_relationship_cirl_mid['dem_cirl_node'].apply(int)
    node_resource_mid['dem_cirl_node'] = node_resource_mid['dem_cirl_node'].apply(int)
    # print_in_log(node_relationship_cirl_mid)
    node_resource_2 = pd.merge(node_resource_mid,node_relationship_cirl_mid,on=['dem_cirl_node'],how='inner')
    print_in_log('此dataframe是未给出节点需求的，资源id，包含了交易和流转节点的合并')
    print_in_log('此dataframe的长度为' + str(len(node_resource_2)))
    # print_in_log(node_resource_2)
    #接下来是要进行资源id的转换，单位，并找到供方教育节点
    resource_convert_1 = resource_convert[['resource_id','package_id','package_unit','package_convert_rate']]
    resource_convert_1['resource_id'] = resource_convert_1['resource_id'].apply(int)
    node_resource_convert =pd.merge(node_resource_2,resource_convert_1,on=['resource_id'],how='inner')
    node_resource_convert =  node_resource_convert.drop(['resource_id'],axis = 1)
    node_resource_convert = node_resource_convert.rename(index=str, columns={"package_id": "resource_id",
                                                                             'available_resource_qty':'resource_available_qty',
                                                                             'package_unit':'unit'})
    print_in_log('此dataframe是未给出节点需求的，资源id，包含了交易和流转节点的合并并处理后的结果')
    print_in_log('此dataframe的长度为' + str(len(node_resource_convert)))
    # print_in_log(node_resource_convert)
    node_resource_convert[['resource_available_qty']] = node_resource_convert[['resource_available_qty']].astype(float)
    node_resource_convert[['package_convert_rate']] = node_resource_convert[['package_convert_rate']].astype(float)
    node_resource_convert['resource_available_qty'] = node_resource_convert['resource_available_qty']/\
                                                      node_resource_convert['package_convert_rate']
    node_resource_convert =  node_resource_convert.drop(['package_convert_rate'],axis = 1)
    node_resource_convert['replenish_qty'] = -1
    node_resource_convert['plan_qty'] = -1
    business_data = business_data[['dem_node_id', 'sup_node_id', 'pre_date']]
    business_data = business_data.rename(index=str, columns={"dem_node_id": "dem_tran_node",
                                                             "sup_node_id": "sup_tran_node",
                                                             "pre_date": "aging"})
    print(business_data)
    node_resource_convert = pd.merge(node_resource_convert, business_data, on=['dem_tran_node', 'sup_tran_node'], how='left')
    node_resource_convert = node_resource_convert[node_resource_convert['dem_tran_node'] == dem_node]

    print_in_log('未给补货建议的最后输出')
    print_in_log('此dataframe的长度为' + str(len(node_resource_convert)))
    # print_in_log(node_resource_convert)
    return node_resource_convert


def demand_final(dem_node,node_demand, node_relationship_tran,node_relationship_cirl):
    # 先删除那些未进行预测的数据资源
    node_demand_first = node_demand[node_demand['demand_qty'] >= 0]
    print_in_log(str(type(node_demand_first['sup_node_id'].iloc[11])))
    node_demand_first = node_demand_first[node_demand_first['sup_node_id'] == dem_node]
    # node_demand_first.to_csv('./' + str(date_parameter) + '/' +
    #                        'node_demand_first.csv', encoding="utf_8_sig")
    print_in_log('node_demand_first'+str(len(node_demand_first)))
    # 对应响应时间，先计算一个上游交易节点的响应时间
    node_relationship_tran_01 = node_relationship_tran[['cnt_at', 'dem_node_id', 'resource_id', 'response_time']]
    # node_relationship_tran_01.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v2---补货/'
    #                        'v2---输入-输出/人工造的数据/test/node_relationship_tran_01.csv', encoding="utf_8_sig")
    print_in_log('node_relationship_tran_01'+str(len(node_relationship_tran_01)))
    # 为了进行方便进行合并操作，需要对对其进行列名更改
    node_relationship_tran_01 = node_relationship_tran_01.rename(index=str, columns={"cnt_at": "demand_date"})
    node_relationship_tran_01 = node_relationship_tran_01.rename(index=str, columns={"dem_node_id": "sup_node_id"})
    node_relationship_tran_01 = node_relationship_tran_01.rename(index=str, columns={"response_time": "up_tran_resp_time"})

    # 对这个上游交易节点进行响应时间的合并
    node_relationship_tran_01[['resource_id', 'sup_node_id']] = node_relationship_tran_01[['resource_id', 'sup_node_id']].astype(int)
    node_demand_first[['resource_id', 'sup_node_id']] = node_demand_first[['resource_id', 'sup_node_id']].astype(int)

    node_demand_first = pd.merge(node_demand_first, node_relationship_tran_01,
                                 on=['demand_date', 'sup_node_id', 'resource_id'], how='left')
    # node_demand_first.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v2---补货/'
    #                        'v2---输入-输出/人工造的数据/test/node_demand_first.csv', encoding="utf_8_sig")
    print_in_log('node_demand_first'+str(len(node_demand_first)))
    # 接下来是要看上游流转节点的响应时间
    node_relationship_cirl_01 = node_relationship_cirl[['cnt_at', 'dem_node_id', 'response_time']]
    # node_relationship_cirl_01.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v2---补货/'
    #                        'v2---输入-输出/人工造的数据/test/node_relationship_cirl_01.csv', encoding="utf_8_sig")
    print_in_log('node_relationship_cirl_01'+str(len(node_relationship_cirl_01)))
    # 为了进行方便进行合并操作，需要对对其进行列名更改
    node_relationship_cirl_01 = node_relationship_cirl_01.rename(index=str,
                                                                 columns={"cnt_at": "demand_date"})
    node_relationship_cirl_01 = node_relationship_cirl_01.rename(index=str,
                                                                 columns={"dem_node_id": "sup_cirl_node_id"})
    node_relationship_cirl_01 = node_relationship_cirl_01.rename(index=str,
                                                                 columns={"response_time": "up_cirl_resp_time"})

    node_demand_first['sup_cirl_node_id'] = \
        node_demand_first['sup_cirl_node_id'].apply(int)
    node_relationship_cirl_01['sup_cirl_node_id'] = \
        node_relationship_cirl_01['sup_cirl_node_id'].apply(int)
    node_demand_first = pd.merge(node_demand_first, node_relationship_cirl_01, on=['demand_date', 'sup_cirl_node_id'],
                                 how='left')
    node_demand_first = node_demand_first.drop_duplicates()
    # node_demand_first.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v2---补货/'
    #                        'v2---输入-输出/人工造的数据/test/node_demand_first.csv', encoding="utf_8_sig")
    print_in_log('node_demand_first'+str(len(node_demand_first)))

    # 同理计算下游交易节点和流转节点的响应时间
    node_relationship_tran_02 = node_relationship_tran[['belong_date', 'dem_node_id', 'resource_id', 'response_time']]
    node_relationship_tran_02 = node_relationship_tran_02.drop_duplicates()
    # node_relationship_tran_02.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v2---补货/'
    #                        'v2---输入-输出/人工造的数据/test/node_relationship_tran_02.csv', encoding="utf_8_sig")
    print_in_log('node_relationship_tran_02'+str(len(node_relationship_tran_02)))
    # 为了进行方便进行合并操作，需要对对其进行列名更改
    node_relationship_tran_02 = node_relationship_tran_02.rename(index=str, columns={"belong_date": "cnt_at"})
    node_relationship_tran_02 = node_relationship_tran_02.rename(index=str,
                                                                 columns={"response_time": "down_tran_resp_time"})

    node_relationship_tran_02[['resource_id', 'dem_node_id']] = \
        node_relationship_tran_02[['resource_id', 'dem_node_id']].astype(int)
    node_demand_first[['resource_id', 'dem_node_id']] = \
        node_demand_first[['resource_id', 'dem_node_id']].astype(int)
    # 对这个上游交易节点进行响应时间的合并
    node_demand_02 = pd.merge(node_demand_first, node_relationship_tran_02, on=['cnt_at', 'dem_node_id', 'resource_id'],
                              how='inner')
    node_demand_02 = node_demand_02.drop_duplicates()
    # node_demand_02.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v2---补货/'
    #                        'v2---输入-输出/人工造的数据/test/node_demand_02.csv', encoding="utf_8_sig")
    print_in_log('node_demand_02'+str(len(node_demand_02)))

    # 接下来是下游的流转节点的响应时间合并操作
    node_relationship_cirl_02 = node_relationship_cirl[['cnt_at', 'dem_node_id', 'response_time']]
    # node_relationship_cirl_02.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v2---补货/'
    #                        'v2---输入-输出/人工造的数据/test/node_relationship_cirl_02.csv', encoding="utf_8_sig")
    print_in_log('node_relationship_cirl_02'+str(len(node_relationship_cirl_02)))
    # 为了进行方便进行合并操作，需要对对其进行列名更改
    node_relationship_cirl_02 = node_relationship_cirl_02.rename(index=str, columns={"cnt_at": "demand_date"})
    node_relationship_cirl_02 = node_relationship_cirl_02.rename(index=str, columns={"dem_node_id": "dem_cirl_node_id"})
    node_relationship_cirl_02 = node_relationship_cirl_02.rename(index=str,
                                                                 columns={"response_time": "down_cirl_resp_time"})

    node_demand_02[['dem_cirl_node_id']] = node_demand_02[['dem_cirl_node_id']].astype(int)
    node_relationship_cirl_02[['dem_cirl_node_id']] = node_relationship_cirl_02[['dem_cirl_node_id']].astype(int)

    # 对这个上游流转节点进行响应时间的合并
    node_demand_final = pd.merge(node_demand_02, node_relationship_cirl_02, on=['demand_date', 'dem_cirl_node_id'],
                                 how='left')
    node_demand_final = node_demand_final.drop_duplicates()
    node_demand_final = node_demand_final.fillna(0)
    # node_demand_final.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v2---补货/'
    #                        'v2---输入-输出/人工造的数据/test/node_demand_final.csv', encoding="utf_8_sig")
    print_in_log('node_demand_final'+str(len(node_demand_final)))
    print_in_log('这是对节点需求进行判断后的结果，是否为空'+str(node_demand_final.empty))
    return node_demand_final


'''#以下测操作需要加入参数调节的内容。把预测的不确定度和响应的不确定度加进去,需要注意的是，forecast_date在从接口读取数据的时候
#就进行筛选,这里需要加的交易节点和流转节点的响应时间的误差，用于与存续期这个进行匹配
#-----------------------------------------------------------------------------------------------------------------------
#设置一个需要满足的服务水平指标，先选择type，然后再匹配预测日期，再匹配其他的指标
#对于参数调节，有三种不确定度的类型，需要进行分开操作,想先将预测的 不确定度加入预测值的计算
'''
def service_level(dem_node,alpha,sigma,gamma,node_demand_final,parameter_revised):
#对于参数需要设置五个参数，一个是交易节点预测的服务水平的参数，基座alpha，一个是交易节点的响应时间的服务水平记做sigma，一个是流转节点的响应时间的
#服务水平记做gamma，还有一个参数是节点需求的dataframe，还有一个是参数调节的dataframe
    #先进行节点需求的在一定服务水平下的误差
    parameter_forecast = parameter_revised[parameter_revised['type']=='tran_forecast']
    parameter_forecast = parameter_forecast[parameter_forecast['sup_node_id'] == dem_node]
    parameter_forecast = parameter_forecast[parameter_forecast['service_level'] == alpha]

    parameter_forecast = parameter_forecast.rename(index=str, columns={"ABS_error": "forecast_error"})
    parameter_forecast = parameter_forecast.drop(['type','service_level','relative_error','forecast_date'],axis=1)
    demand_forecast = pd.merge(node_demand_final,parameter_forecast,on=['resource_id','dem_node_id','sup_node_id'],how='left')
    demand_forecast = demand_forecast.fillna(0)
    print_in_log('这是先对预测不确定度进行合并后的结果，是否为空:'+str(demand_forecast.empty))
    #在对上游交易节点的响应时间的不确定度进行计算
    parameter_up_tran_response = parameter_revised[parameter_revised['type']=='tran_response']
    parameter_up_tran_response = parameter_up_tran_response[parameter_up_tran_response['service_level'] == sigma]
    parameter_up_tran_response = parameter_up_tran_response.drop(['type','service_level','relative_error','sup_node_id','forecast_date'],axis=1)
    parameter_up_tran_response = parameter_up_tran_response.rename(index=str, columns={"ABS_error": "up_tran_error_response"})
    #因为是上游交易的节点的响应不确定度，所以，参数调节内的需求节点，对应的应该是节点需求内的供应节点
    parameter_up_tran_response = parameter_up_tran_response.rename(index=str, columns={"dem_node_id": "sup_node_id"})
    demand_up_response =pd.merge(demand_forecast,parameter_up_tran_response,on=['resource_id','sup_node_id'],how='left')
    demand_up_response = demand_up_response.fillna(0)
    print_in_log('这是先对上游交易节点响应时间合并的结果，是否为空:'+str(demand_up_response.empty))
    #在对下游交易节点的响应时间的不确定度进行计算
    parameter_down_tran_response = parameter_revised[parameter_revised['type']=='tran_response']
    parameter_down_tran_response = parameter_down_tran_response[parameter_down_tran_response['service_level'] == sigma]
    parameter_down_tran_response = parameter_down_tran_response.drop(['type','service_level','relative_error','forecast_date'],axis=1)
    parameter_down_tran_response = parameter_down_tran_response.rename(index=str, columns={"ABS_error": "down_tran_error_response"})
    demand_tran_response =pd.merge(demand_up_response,parameter_down_tran_response,on=['resource_id','sup_node_id','dem_node_id'],how='left')
    demand_tran_response = demand_tran_response.fillna(0)
    print_in_log('这是先对下游交易节点响应时间合并的结果，是否为空:'+str(demand_tran_response.empty))

    #计算上游流转节点响应时间的误差值
    parameter_up_cirl = parameter_revised[parameter_revised['type']=='cirl_response']
    parameter_up_cirl = parameter_up_cirl[parameter_up_cirl['service_level'] == gamma]
    parameter_up_cirl = parameter_up_cirl.drop(['type','service_level','relative_error','forecast_date','sup_node_id'],axis=1)
    parameter_up_cirl = parameter_up_cirl.rename(index=str, columns={"ABS_error": "up_cirl_error_response",'dem_node_id':'sup_cirl_node_id'})
    demand_cirl =pd.merge(demand_tran_response,parameter_up_cirl,on=['resource_id','sup_cirl_node_id'],how='left')
    demand_cirl = demand_cirl.fillna(0)
    print_in_log('这是先对上游流转节点响应时间合并的结果，是否为空:'+str(demand_cirl.empty))
    #计算下游流转节点的响应时间的误差
    parameter_down_cirl = parameter_revised[parameter_revised['type']=='cirl_response']
    parameter_down_cirl = parameter_down_cirl[parameter_down_cirl['service_level'] == gamma]
    parameter_down_cirl = parameter_down_cirl.drop(['type','service_level','relative_error','forecast_date'],axis=1)
    parameter_down_cirl = parameter_down_cirl.rename(index=str, columns={"ABS_error": "down_cirl_error_response",'dem_node_id':'dem_cirl_node_id',
                                                                         "sup_node_id":"sup_cirl_node_id"})
    parameter_down_cirl = parameter_down_cirl.fillna(0)
    print_in_log('这是先对下游流转节点响应时间合并的结果，是否为空:'+str(parameter_down_cirl.empty))
    demand_response =pd.merge(demand_cirl,parameter_down_cirl,on=['resource_id','sup_cirl_node_id','dem_cirl_node_id'],how='left')
    demand_response = demand_response.fillna(0)
    print_in_log('将满足一定服务水平下的误差加入评估模型中，最后的结果是否为空'+str(demand_response.empty))
    return demand_response



'''#接下来是要对节点需求进行最后一步的判断，节点的响应时间，预测的不确定度的计算等
#-------------------------------------------------------------------------------------------------------------
'''
def demand_last(dem_node,alpha,sigma,gamma,node_demand,node_relationship_tran,node_relationship_cirl,parameter_revised):
    t0 = time.time()
    node_demand_final = demand_final(dem_node,node_demand,node_relationship_tran, node_relationship_cirl)
    t1 = time.time()
    print_in_log('demand_final耗时：'+str(t1 - t0))
    t2 = time.time()
    demand_response = service_level(dem_node,alpha, sigma, gamma, node_demand_final, parameter_revised)
    t3 = time.time()
    print_in_log('service_level耗时：'+str(t3 - t2))
    demand_response.to_csv('./' + str(date_parameter) + '/' +
                           'demand_response.csv', encoding="utf_8_sig")
    # print_in_log(demand_dataframe)
    # 获得了所有节点的响应时间后，需要对每对供需节点的节点需求进行重新计算
    t4 = time.time()
    demand_response['response_time'] = demand_response['up_tran_resp_time'] + \
                                       demand_response['up_cirl_resp_time'] + \
                                       demand_response['up_cirl_error_response'] + \
                                       demand_response['up_tran_error_response'] + \
                                       demand_response['down_cirl_resp_time'] + \
                                       demand_response['down_tran_resp_time']+  \
                                       demand_response['down_tran_error_response'] + \
                                       demand_response['down_cirl_error_response']
    demand_response['duration_time'] = pd.to_numeric(demand_response['duration_time'])
    #定义一个函数用于来评估存续期与响应时间，得到最后的真实的需求
    def compare_max_min(a, b):
        if a < b:
            return 0
        else:
            return 1

    demand_response['demand_qty'] = demand_response.apply\
        (lambda row: row['demand_qty'] * compare_max_min(row['response_time'], row['duration_time']), axis=1)

    t5 = time.time()
    print_in_log('循环耗时：'+str(t5 - t4))
    print_in_log('正在对节点需求进行存续期的判断，这是最后的节点需求的输出，是否为空'+
                 str(demand_response.empty))
    return demand_response

'''#这里需要新增三列，可卖库存（new），期末库存和计划资源数量，粒度是每个资源id和每天
#--------------------------------------------------------------------------------------------------------------------
#对节点资源和节点需求做交集，因为对于一些未给出节点需求的资源，将在最后输出的时候再进行合并
'''

#此函数是在多线程内进行计算的补货计划的数量
def plan_supply_inside(resource_id,plan,process):
    print_in_log('进程：'+str(process)+'正在进行plan_supply的运算')
    # 这个需要针对每个资源ID
    # 同时这里需要新建一个空的dataframe，用来存储所有的每个资源id计算后的结果
    plan_resource = pd.DataFrame(columns=['cnt_at', 'tran_node_id', 'cirl_node_id', 'resource_id',
                                          'vendible_resource_qty', 'available_resource_qty', 'update_date',
                                          'forecast_error', 'demand_qty', 'VRQ_NEW', 'VRQ_end', 'plan_qty'
                                            ,'up_tran_resp_time', 'up_cirl_resp_time', 'down_tran_resp_time',
                                          'down_cirl_resp_time', 'up_cirl_error_response',
                                          'up_tran_error_response', 'dem_cirl_node_id'])
    #存在一共供方节点多个需方节点的关系，需要需要按照比例进行比例拆分
    total_demand = plan['demand_qty'].sum()

    # 对每一个的资源id进行供应计划的计算
    for res_id in resource_id:
        plan_resource_mid = pd.DataFrame(columns=['cnt_at', 'tran_node_id', 'cirl_node_id', 'resource_id',
                                              'vendible_resource_qty', 'available_resource_qty', 'update_date',
                                              'forecast_error', 'demand_qty', 'VRQ_NEW', 'VRQ_end', 'plan_qty'
            , 'up_tran_resp_time', 'up_cirl_resp_time', 'down_tran_resp_time',
                                              'down_cirl_resp_time', 'up_cirl_error_response',
                                              'up_tran_error_response', 'dem_cirl_node_id'])
        new_plan = plan[plan['resource_id'] == res_id]
        k = 0
        new_plan_01 = new_plan[['dem_cirl_node_id', 'cirl_node_id']]
        new_plan_01 = new_plan_01.drop_duplicates()
        dem_cirl_node_id = new_plan_01['dem_cirl_node_id'].tolist()
        cirl_node_id = new_plan_01['cirl_node_id'].tolist()
        while k < len(dem_cirl_node_id):
            dem_cirl_node_id_x = dem_cirl_node_id[k]
            # 计算当前需方节点对应能够分配到多少的节点资源
            mid_plan = (new_plan[new_plan['dem_cirl_node_id'] == dem_cirl_node_id_x])
            ratio = int((mid_plan['demand_qty'].sum())/total_demand)
            cirl_node_id_x = cirl_node_id[k]
            # print_in_log(dem_cirl_node_id_x)
            new_plan_mid = new_plan[new_plan['dem_cirl_node_id'] == dem_cirl_node_id_x]
            new_plan_mid = new_plan_mid[new_plan_mid['cirl_node_id'] == cirl_node_id_x]
            new_plan_mid = new_plan_mid.reset_index()
            #对可卖库存进行更新
            new_plan_mid['VRQ_NEW'] = new_plan_mid['VRQ_NEW']*ratio
            new_plan_mid_VRQ_NEW = new_plan_mid['VRQ_NEW'].tolist()
            new_plan_mid_vendible_resource_qty = new_plan_mid['VRQ_NEW'].tolist()
            new_plan_mid_demand_qty = new_plan_mid['demand_qty'].tolist()
            new_plan_mid_forecast_error = new_plan_mid['forecast_error'].tolist()
            new_plan_mid_plan_qty = new_plan_mid['plan_qty'].tolist()
            # print_in_log(new_plan_mid)
            new_plan_mid_VRQ_end = new_plan_mid['VRQ_end'].tolist()
            i = 0
            while i < len(new_plan_mid_VRQ_NEW):
                if i == 0:
                    new_plan_mid_VRQ_NEW[0] = new_plan_mid_vendible_resource_qty[0]
                    x1 = new_plan_mid_VRQ_NEW[0]
                    s1 = new_plan_mid_demand_qty[0]
                    ss = new_plan_mid_forecast_error[0]
                    if x1 - s1 > ss:
                        new_plan_mid_plan_qty[i] = 0
                        new_plan_mid_VRQ_end[i] = x1 - s1
                    else:
                        new_plan_mid_plan_qty[i] = ss - x1 + s1
                        new_plan_mid_VRQ_end[i] = ss
                else:
                    xi_old = new_plan_mid_vendible_resource_qty[i]
                    x1 = new_plan_mid_VRQ_NEW[0]
                    delta = xi_old - x1
                    new_plan_mid_VRQ_NEW[i] = new_plan_mid_VRQ_end[i - 1] + delta
                    # 第二部分的操作计算除第一行的期末库存和plan的数据
                    xi_new = new_plan_mid_VRQ_NEW[i]
                    si = new_plan_mid_demand_qty[i]
                    ss = new_plan_mid_forecast_error[0]
                    if xi_new - si > ss:
                        new_plan_mid_plan_qty[i] = 0
                        new_plan_mid_VRQ_end[i] = xi_new - si
                    else:
                        new_plan_mid_plan_qty[i] = ss - xi_new + si
                        new_plan_mid_VRQ_end[i] = ss
                i += 1
            new_plan_mid['plan_qty'] = pd.Series(new_plan_mid_plan_qty)
            new_plan_mid['VRQ_end'] = pd.Series(new_plan_mid_VRQ_end)
            # print_in_log(new_plan_mid_plan_qty)
            plan_resource_mid = plan_resource_mid.append(new_plan_mid)
            k += 1
        # plan_resource_mid.to_csv('plan_resource_mid'+str(res_id)+'.csv', encoding="utf_8_sig")
        plan_resource = plan_resource.append(plan_resource_mid)
    return plan_resource

#定义多进程函数，用来进行分进程进行数据处理
def plan_supply_multi(resource_id,plan):
    length_res = len(resource_id)
    n = 12  #这里计划设置60个线程进行计算
    step = int(math.ceil(length_res / n))
    lists = {}
    for i in range(1, n + 1):
        lists[(i - 1) * step + 1] = i * step + 1

    pool = multiprocessing.Pool(processes=12)  # 创建60个进程
    results = []
    data = pd.DataFrame(columns=['cnt_at', 'tran_node_id', 'cirl_node_id', 'resource_id',
                                      'vendible_resource_qty', 'available_resource_qty', 'update_date',
                                      'forecast_error', 'demand_qty', 'VRQ_NEW', 'VRQ_end', 'plan_qty'
                                      ,'up_tran_resp_time', 'up_cirl_resp_time', 'down_tran_resp_time',
                                      'down_cirl_resp_time', 'up_cirl_error_response',
                                      'up_tran_error_response', 'dem_cirl_node_id'])
    for start_user, end_user in lists.items():
        print_in_log('进行这段资源id的供应计划的计算' + str(start_user) +',' + str(end_user))
        resource_id_multi_process = resource_id[(start_user-1):(end_user-1)]
        print_in_log(str(start_user)+str(end_user))
        results.append(pool.apply_async(plan_supply_inside, args=(resource_id_multi_process, plan,start_user)))
    pool.close()  # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
    pool.join()  # 等待进程池中的所有进程执行完毕

    for i in results:
        a = i.get()
        data = data.append(a, ignore_index=True)
    return data

#这是每个资源id计划资源的主函数
def plan_supply(demand_response,node_resource):
#首先需要将节点资源的数据和节点的需求数据进行合并，对于节点需求的数据，只需要保留节点需求，预测误差供徐关系和资源id
    demand_merge = demand_response[['forecast_error','sup_node_id','sup_cirl_node_id','resource_id',
                                    'demand_date','demand_qty','up_tran_resp_time','up_cirl_resp_time',
                                    'down_tran_resp_time','down_cirl_resp_time','up_cirl_error_response',
                                    'up_tran_error_response','dem_cirl_node_id']]
    demand_merge = demand_merge.rename(index= str,columns={"sup_node_id": "tran_node_id",
                                                           'sup_cirl_node_id':'cirl_node_id',
                                                           'demand_date':'cnt_at'
                                                           })
    demand_merge[['resource_id']] = demand_merge[['resource_id']].astype(float)
    plan = pd.merge(node_resource,demand_merge,on=['resource_id','tran_node_id',
                                                   'cirl_node_id','cnt_at'],how='inner')
    plan[['resource_id']] = plan[['resource_id']].astype(float)

    #将两个dataframe合并后需要新增三列，并进行初始化负值
    plan['VRQ_NEW'] = plan['vendible_resource_qty']
    plan['VRQ_end'] = 0
    plan['plan_qty'] = 0
    plan = plan.fillna(0)
    plan.to_csv('./' + str(date_parameter) + '/' + 'plan.csv',encoding="utf_8_sig")
    resource_id = list(set(plan['resource_id']))
    print_in_log(str(resource_id))
    #接下来的操作是需要计算新的期初库存和，期末库存和资源补货计划
    plan_resource = plan_supply_multi(resource_id,plan)
    #以防多进程重复计算，需要去重操作
    plan_resource = plan_resource.drop_duplicates()
    # 这一步的操作是为了用一个单独的列来表示，当前节点与上游节点的所有响应时间和误差值的和
    plan_resource['response'] = plan_resource['up_tran_resp_time']+ plan_resource['up_cirl_resp_time']+\
                                plan_resource['up_cirl_error_response']+ plan_resource['up_tran_error_response']
    plan_resource[['resource_id','tran_node_id']] = plan_resource[['resource_id','tran_node_id']].astype(int)
    print_in_log('这是供应计划的输出')
    return plan_resource


'''#以下需要结合业务数据，
#----------------------------------------------------------------------------------------------
#给出建议的补货数量，首先要对业务数据的时效性进行分析
#定义一个函数用于处理业务数据和计划资源
'''
def business_plan(node_relationship_tran,plan_resource,business_data):
#这部操作是需要将计划资源的dataframe中加入服务水平的内容，因此需要输入计划dataframe和交易节点的关系
#首先需要将节点关系只保留需方节点，日期和服务水平和资源分布与状态的一些列，再与计划资源的dataframe进行merge
    mid = node_relationship_tran[['cnt_at','sup_node_id','dem_node_id','resource_id','unit',
                                  'nom_resource','response_time','service_level',
                                  'current_status','parent_status']]
    mid = mid.rename(index=str, columns={"dem_node_id": "tran_node_id"})

    #然后将计划资源的dataframe与处理过的节点关系的数据进行合并

    mid[['tran_node_id']] = mid[['tran_node_id']].astype(int)
    mid[['resource_id']] = mid[['resource_id']].astype(int)

    merge_plan = pd.merge(plan_resource, mid, on=['resource_id','tran_node_id','cnt_at'],how='inner')
    #这时候还需要对业务数据进行处理，这是因为，业务数据中存在着不同的预告时间
    business_effective = business_data[['dem_node_id','sup_node_id','repl_cycle','pre_date',
                                        'min_order','min_unit']]
    business_effective = business_effective.drop_duplicates()
    #t同时还需要计算有几种供需关系，因此需要计算因此需要对业务数据进行处理
    mid_business = business_data[['dem_node_id','sup_node_id']]
    mid_business = mid_business.drop_duplicates()
    print_in_log('这是加入了业务数据处理后的，供应计划是否为空'+str(mid_business.empty))
    return mid_business,merge_plan,business_effective


#设置此函数是用来定义计算一个约定到货时间，和需要为其准备的补货天数，该函数内包含了，对于重叠天数的处理
#这里需要输入的是一个响应时间，预告时间的列表和补货周期的列表
#返回的是一个约定到货时间喝
def process_repl_date(response_time_int,list_pre_date,list_repl_cycle):
    '''# 因此约定到货日期=预告时间+响应时间'''
    deal_arrive = [int(i + response_time_int) for i in list_pre_date]
    '''# 接下来是要加入补货周期的参数，用来计算需要补货的天数'''
    mid_repl = [int(list_repl_cycle[i] + deal_arrive[i]) for i in range(0, len(list_repl_cycle))]
    '''# 以下是执行补货的天数的计算，防止重叠的天数产生'''
    for i in range(1, len(mid_repl)):
        b = deal_arrive[i]
        a = mid_repl[i - 1]
        if a >= b:
            mid_repl[i - 1] = deal_arrive[i] - 1
        else:
            pass
    return deal_arrive,mid_repl

#这里需要再定义一个函数用于将补货中间表的merge_plan的流转节点的供应节点找到
def up_cirl_merge(merge_plan,node_relationship_cirl):
    node_relationship_cirl_01 = node_relationship_cirl[['dem_node_id','sup_node_id']]
    node_relationship_cirl_mid_default = node_relationship_cirl_01.drop_duplicates()

    node_relationship_cirl_mid = pd.DataFrame(columns=['cirl_node_id','sup_cirl_node'])
    node_relationship_cirl_mid['cirl_node_id'] =node_relationship_cirl_mid_default['dem_node_id']
    node_relationship_cirl_mid['sup_cirl_node'] = node_relationship_cirl_mid_default['sup_node_id']
    node_relationship_cirl_mid['cirl_node_id'] = node_relationship_cirl_mid['cirl_node_id'].apply(int)
    merge_plan['cirl_node_id'] = merge_plan['cirl_node_id'].apply(int)
    print_in_log(str(type(merge_plan['cirl_node_id'])))
    # print_in_log(merge_plan['cirl_node_id'])

    up_merge = pd.merge(merge_plan,node_relationship_cirl_mid,on=['cirl_node_id'],how='inner')
    up_merge = up_merge.drop_duplicates()
    return up_merge



#这里需要定义一个函数，用来看应该用什么样的补货规格
#需要输入计划补货的数量（最小单位，对应的资源id，资源关系转换表
def package_convert_rate(repl_qty,res_id,resource_convert,available_resource_qty):
    resource_convert_id = resource_convert[['resource_id','min_unit','package_id','package_unit',
                                            'package_convert_rate','box_id','box_unit','box_convert_rate'
                                            ]]
    #采用的逻辑是先转换成box_unit看四舍五入的计算方法是否为0，如果不为0，则按照box_unit进行补货，如果为0则按照package进行补货
    resource_convert_id = resource_convert_id[resource_convert_id['resource_id'] == res_id]
    resource_convert_id['package_convert_rate'] = pd.to_numeric(resource_convert_id['package_convert_rate'])
    #repl_package是建议补货数量按照四舍五入的方式进行取整计算
    repl_package = round(repl_qty/(resource_convert_id['package_convert_rate'].iloc[0]))
    package_id = resource_convert_id['package_id'].iloc[0]
    package_unit = resource_convert_id['package_unit'].iloc[0]
    #输出的所有数量都是按照package——id对应的数量来的，所以需要对第一步决策后的计划资源和可卖库存进行计算
    #repl_qty_package这是未经任何修改的建议补货的数量进行的单位转换
    repl_qty_package = float('%.2f' % (repl_qty/(resource_convert_id['package_convert_rate'].iloc[0])))
    available_resource_qty_package = float('%.2f' % (available_resource_qty/(resource_convert_id['package_convert_rate'].iloc[0])))
    return package_id,package_unit,repl_package,repl_qty_package,available_resource_qty_package,package_unit

def main_function(dem_node,date_parameter_transform,alpha,sigma,gamma,node_relationship_tran,
                  node_relationship_cirl, node_demand,node_description,
                  node_resource, parameter_revised, business_data,
                  resource_convert,node_demand_without):
#再开始程序就选定一个一个计算开始的时间
    now = time.time()
    local_time = time.localtime(now)
    today = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
    print_in_log('当前决策的时间为' +str(today))
    t0 = time.time()
    # #这一步是依据服务水平和相应时间等对节点需求进行修正
    demand_response = demand_last(dem_node,alpha,sigma,gamma, node_demand,
                                  node_relationship_tran, node_relationship_cirl,
                                  parameter_revised)
    demand_response.to_csv('./' + str(date_parameter) + '/' +
                           'demand_response'+str(date_parameter_transform)+str(dem_node)+'.csv', encoding="utf_8_sig")
    # demand_response = pd.read_csv("demand_response.csv", keep_default_na=False)
    t1 = time.time()
    print_in_log('demand_last耗时：'+str(t1 - t0))

    #这是不会补货计划进行第一步的计算
    plan_resource = plan_supply(demand_response, node_resource)
    # plan_resource.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v2---补货/'
    #                        'v2---输入-输出/人工造的数据/test/plan_resource.csv', encoding="utf_8_sig")
    plan_resource.to_csv('./' + str(date_parameter) + '/' +
                         'plan_resource'+str(date_parameter_transform)+str(dem_node)+'.csv', encoding="utf_8_sig")
    # plan_resource = pd.read_csv("plan_resource.csv", keep_default_na=False)
    t2 = time.time()
    print_in_log('plan_supply耗时：'+str(t2 - t1))

    mid_business, merge_plan_01, business_effective = business_plan(node_relationship_tran, plan_resource, business_data)

    # mid_business.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v2---补货/'
    #                      'v2---输入-输出/人工造的数据/test/mid_business.csv', encoding="utf_8_sig")

    merge_plan_01.to_csv('./' + str(date_parameter) + '/' +
                         'merge_plan_01'+str(date_parameter_transform)+str(dem_node)+'.csv', encoding="utf_8_sig")
    t3 = time.time()
    print_in_log('business_plan耗时：'+str(t3 - t2))
    merge_plan = up_cirl_merge(merge_plan_01, node_relationship_cirl)
    merge_plan.to_csv('./' + str(date_parameter) + '/' +
                      'merge_plan'+str(date_parameter_transform)+str(dem_node)+'.csv', encoding="utf_8_sig")
    # merge_plan = pd.read_csv("merge_plan.csv", keep_default_na=False)
    # merge_plan.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v2---补货/'
    #               'v2---输入-输出/人工造的数据/test/merge_plan.csv', encoding="utf_8_sig")
    t4 = time.time()
    print_in_log('up_cirl_merge耗时：'+str(t4 - t3))

    without_prediction_data = without_prediction(dem_node,node_demand_without, node_resource, node_relationship_tran,
                                                 node_relationship_cirl, resource_convert,business_data)

    without_prediction_data.to_csv('./' + str(date_parameter) + '/' +
                                   'without_prediction_data'+str(date_parameter_transform)+str(dem_node)+'.csv', encoding="utf_8_sig")
    t5 = time.time()
    print_in_log('without_prediction耗时：'+str(t5 - t4))
#<-----------------------------------------------------------------------------读取外部参数
def date_parameter_read():

    date_parameter_intercept = sys.argv[1]
    dem_node_01 = sys.argv[2]
    dem_node = int(dem_node_01)
    sup_node_01 = sys.argv[3]
    sup_node = int(sup_node_01)
    print_in_log(dem_node)
    print_in_log(sup_node)
    # date_parameter_intercept = '20190307'
    return date_parameter_intercept, dem_node, sup_node

if __name__ == '__main__':
    try:
        st = time.time()
        begin = datetime.now()
        date_parameter,dem_node, sup_node = date_parameter_read()
        node_relationship_tran, node_relationship_cirl, node_demand, \
        node_description, node_resource, parameter_revised, business_data,\
        resource_convert,node_demand_without = get_data(date_parameter)
        main_function(dem_node,date_parameter, 0.95, 0.95, 0.95, node_relationship_tran, node_relationship_cirl, node_demand,
         node_description, node_resource, parameter_revised, business_data, resource_convert,node_demand_without)
        ed = time.time()
        print('总程序耗时：'+ str(ed - st))
        print("result:1")
    except OSError as reason:
        print_in_log('出错原因是%s'+str(reason))
        print ("result:0")