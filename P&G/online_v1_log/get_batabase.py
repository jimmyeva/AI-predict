# -*- coding = utf-8 -*-
'''
@Time: 2019/1/30 13:24
@Author: Ye Jinyu
'''

import pandas as pd
import numpy as np
import warnings
import math
from datetime import datetime,timedelta
from datetime import date
warnings.filterwarnings("ignore")
import pymysql
import traceback
import sys
import cx_Oracle as cx
from hessian_util import *
#此函数是让pandas显示所有的行和列
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import time
from datetime import datetime,timedelta
import time
import os


pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

#----------------------------------------------------------使用http的协议去读取接口的数据
#172.16.4.134:9106
urllib = HessianUtil(
    "http://192.168.6.165:9106/com.wwwarehouse.xdw.supplychainBi.service.BiService")


class get_bd(object):#定义一个类，在这个py文件里面存在编辑好的函数

#设立一个函数用于得到节点信息的数据
#------------------------------------------------------------------------------------
#此函数是用于获取节点关系（交易节点）的接口数据

    def mkdir(self,path):
        folder = os.path.exists(path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
            print(
                "----生成新的文件目录----")
        else:
            print(
                "当前文件夹已经存在")

    # 192.168.6.165:9106
    def print_in_log(self,date_parameter,string):
        print(string)
        date_1 = datetime.now()
        str_10 = datetime.strftime(date_1, '%Y-%m-%d')
        file = open('./' + str(date_parameter) + '/' + 'log' + str(str_10) + '.txt', 'a')
        file.write(str(string) + '\n')

    def node_relationship_tran(self,date_parameter,b, e, s, w, process):
        self.print_in_log(date_parameter,'进程' + str(process) + '：开始读数据'+str(b)+str(e)+str(s)+str(w))
        get_bd = urllib.request("getBiDataParam",
                                "{"
                                "\"selects\": [\"cnt_at\",\"sup_node_id\",\"dem_node_id\""
                                ",\"resource_id\",\"unit\",\"nom_resource\",\"response_time\",\"service_level\","
                                "\"current_status\",\"parent_status\",\"belong_date\"],"
                                "\"dataSet\": \"node_relationship_tran\","
                                "\"filters\":["
                                "{\"column\":\"cnt_at\",\"type\":\"between\",\"values\":[\"%s\", \"%s\"]},"
                                "{\"column\":\"belong_date\",\"type\":\"between\",\"values\":[\"%s\", \"%s\"]}]}"
                                % (b, e, s, w))

        node_relationship = get_bd['data']['data']['dataList']

        len_node_relationship = len(node_relationship)
        if len_node_relationship == 0:
            nod_relationship_tran = pd.DataFrame()

        else:
            nod_relationship_tran = pd.DataFrame(node_relationship)
            nod_relationship_tran['resource_id'] = nod_relationship_tran['resource_id'].apply(int)
            nod_relationship_tran['sup_node_id'] = nod_relationship_tran['sup_node_id'].apply(int)
            nod_relationship_tran['dem_node_id'] = nod_relationship_tran['dem_node_id'].apply(int)
            nod_relationship_tran['belong_date'] = nod_relationship_tran.apply(lambda row: row['belong_date']['value'],
                                                                               axis=1)
            nod_relationship_tran['cnt_at'] = nod_relationship_tran.apply(lambda row: row['cnt_at']['value'], axis=1)
            nod_relationship_tran['service_level'] = nod_relationship_tran.apply(lambda row: row['service_level']['value'],
                                                                                 axis=1)

        self.print_in_log(date_parameter,'进程' + str(process) + '：数据长度' +
                     str(len(nod_relationship_tran))+'数据大小:'+
                     str(sys.getsizeof(nod_relationship_tran)))
        return nod_relationship_tran

    #此函数是用于获取节点关系（流转节点）的接口数据
    def node_relationship_cirl(self,date_parameter,s,w):
        get_bd = urllib.request("getBiDataParam",
                                "{\"selects\": [\"cnt_at\",\"belong_date\",\"sup_node_id\""
                                ",\"dem_node_id\",\"unit\",\"response_time\"],\"dataSet\": \"node_relationship_cirl\","
                                "\"filters\":["
                                "{\"column\":\"belong_date\",\"type\":\"between\",\"values\":[\"%s\", \"%s\"]}]}"
                                % (s, w))
        node_relationship_cirl = get_bd['data']['data']['dataList']
        len_node_relationship_cirl = len(node_relationship_cirl)
        # print(get_bd['data']['data']['dataList'])
        self.print_in_log(date_parameter,'读取流转节点的节点关系，此项输入共有行数：'+str(len_node_relationship_cirl))
        if len_node_relationship_cirl == 0:
            self.print_in_log(date_parameter,"读取数据失败，数据为0")
        else:

            node_relationship_cirl = pd.DataFrame(node_relationship_cirl)
            node_relationship_cirl['sup_node_id'] = node_relationship_cirl['sup_node_id'].apply(int)
            node_relationship_cirl['dem_node_id'] = node_relationship_cirl['dem_node_id'].apply(int)
            node_relationship_cirl['cnt_at'] = node_relationship_cirl.apply(lambda row: row['cnt_at']['value'],axis=1)
            node_relationship_cirl['belong_date'] = node_relationship_cirl.apply(lambda row: row['belong_date']['value'],axis=1)
            return node_relationship_cirl



    #设立一个函数用于得到节点资源(可卖资源）信息的数据
    #------------------------------------------------------------------------------------
    def get_node_resource(self,date_parameter,s, w):
        get_bd = urllib.request("getBiDataParam",
                            "{\"selects\": [\"cnt_at\",\"tran_node_id\",\"cirl_node_id\""
                            ",\"resource_id\",\"vendible_resource_qty\",\"available_resource_qty\","
                            "\"update_date\"],\"dataSet\": \"node_resource\","
                                "\"filters\":["
                                # "{\"column\":\"resource_id\",\"type\":\"in\",\"values\":[\"3861\",\"3121\","
                                # "\"3187\",\"3246\",\"3188\",\"3124\",\"3268\",\"3856\",\"3344\",\"3269\","
                                # "\"3340\",\"3126\",\"3855\",\"3842\",\"3157\"]},"
                                
                                # "{\"column\":\"resource_id\",\"type\":\"in\",\"values\":[\"8112\",\"8234\","
                                # "\"8182\",\"8153\",\"8142\",\"8178\",\"8177\",\"8097\",\"8110\",\"8154\","
                                # "\"2854\",\"2855\",\"2917\",\"2443\",\"2616\", \"2398\",\"2725\",\"2440\",\"2762\",\"2438\"]}"
                                
                                # "{\"column\":\"resource_id\",\"type\":\"between\",\"values\":[\"8100\",\"8300\"]},"
                                "{\"column\":\"update_date\",\"type\":\"between\",\"values\":[\"%s\", \"%s\"]}]}"
                                % (s, w))
        get_node_resource = get_bd['data']['data']['dataList']
        len_get_node_resource = len(get_node_resource)
        # print_in_log(get_bd['data']['data']['dataList'])
        self.print_in_log(date_parameter,'读取节点资源，此项输入共有行数：'+str(len_get_node_resource))
        if len_get_node_resource == 0:
            get_node_resource = pd.DataFrame()
            self.print_in_log(date_parameter,"读取数据失败，数据为0")
        else:
            get_node_resource = pd.DataFrame(get_node_resource)
            get_node_resource['tran_node_id'] = get_node_resource['tran_node_id'].apply(int)
            get_node_resource['cirl_node_id'] = get_node_resource['cirl_node_id'].apply(int)
            get_node_resource['resource_id'] = get_node_resource['resource_id'].apply(int)
            get_node_resource['cnt_at'] = get_node_resource.apply(lambda row: row['cnt_at']['value'],axis=1)
            self.print_in_log(date_parameter,str(type(get_node_resource['cnt_at'].iloc[11])))
            get_node_resource['update_date'] = get_node_resource.apply(lambda row: row['update_date']['value'],axis=1)
            # get_node_resource['cnt_at'] = get_node_resource['cnt_at'].dt.strftime('%Y-%m-%d')
            # get_node_resource['update_date'] = get_node_resource['update_date'].dt.strftime('%Y-%m-%d')

        return get_node_resource

    #设立一个函数用于得到节点需求的数据
    #------------------------------------------------------------------------------------
    def node_demand(self,date_parameter,b,e,s,w,proecess):
        self.print_in_log(date_parameter,'正在进行进程：'+str(proecess)+str(b)+str(e)+str(s)+str(w))
        get_bd = urllib.request("getBiDataParam",
                            "{\"selects\": [\"dem_cirl_node_id\",\"sup_cirl_node_id\",\"forecast_HY\""
                            ",\"forecast_season\",\"forecast_month\",\"duration_time\",\"unit\","
                            "\"demand_qty\",\"resource_id\",\"cnt_at\",\"dem_node_id\","
                            "\"sup_node_id\",\"demand_date\"],"
                            "\"dataSet\": \"node_demand\","
                                "\"filters\":["
                                "{\"column\":\"demand_qty\",\"type\":\">\",\"values\":[\"-1\"]},"
                                "{\"column\":\"demand_date\",\"type\":\"between\",\"values\":[\"%s\", \"%s\"]},"
                                "{\"column\":\"cnt_at\",\"type\":\"between\",\"values\":[\"%s\", \"%s\"]}]}"
                                % (b,e,s,w))
        # print_in_log(get_bd)
        get_node_demand = get_bd['data']['data']['dataList']
        len_get_node_demand = len(get_node_demand)
        # print_in_log(get_bd['data']['data']['dataList'])

        self.print_in_log(date_parameter,'筛选条件：demand_date,cnt_at'+str(b)+str(e)+str(s)+str(w))
        if len_get_node_demand == 0:
            get_node_demand = pd.DataFrame()
            self.print_in_log(date_parameter,"读取数据失败，数据为0")
        else:
            get_node_demand = pd.DataFrame(get_node_demand)
            get_node_demand['dem_cirl_node_id'] = get_node_demand['dem_cirl_node_id'].apply(int)
            get_node_demand['sup_cirl_node_id'] = get_node_demand['sup_cirl_node_id'].apply(int)
            get_node_demand['sup_node_id'] = get_node_demand['sup_node_id'].apply(int)
            get_node_demand['resource_id'] = get_node_demand['resource_id'].apply(int)
            get_node_demand['dem_node_id'] = get_node_demand['dem_node_id'].apply(int)
            get_node_demand['duration_time'] = get_node_demand.apply(lambda row: row['duration_time']['value'],axis=1)
            get_node_demand['demand_date'] = get_node_demand.apply(lambda row: row['demand_date']['value'],axis=1)
            get_node_demand['cnt_at'] = get_node_demand.apply(lambda row: row['cnt_at']['value'], axis=1)
            # get_node_demand['cnt_at'] = get_node_demand['cnt_at'].dt.strftime('%Y-%m-%d')
            # get_node_demand['demand_date'] = get_node_demand['demand_date'].dt.strftime('%Y-%m-%d')
        self.print_in_log(date_parameter,'读取节点需求的数据，此项输入共有行数:'+str(len_get_node_demand)+
                     '数据大小:' + str(sys.getsizeof(get_node_demand)))
        return get_node_demand



    def node_demand_without(self,date_parameter,b,e,s,w,proecess):
        self.print_in_log(date_parameter,'正在进行进程：'+str(proecess)+str(b)+str(e)+str(s)+str(w))
        get_bd = urllib.request("getBiDataParam",
                            "{\"selects\": [\"dem_cirl_node_id\",\"sup_cirl_node_id\",\"forecast_HY\""
                            ",\"forecast_season\",\"forecast_month\",\"duration_time\",\"unit\","
                            "\"demand_qty\",\"resource_id\",\"cnt_at\",\"dem_node_id\","
                            "\"sup_node_id\",\"demand_date\"],"
                            "\"dataSet\": \"node_demand\","
                                "\"filters\":["
                                "{\"column\":\"demand_qty\",\"type\":\"in\",\"values\":[\"-1\"]},"
                                "{\"column\":\"demand_date\",\"type\":\"between\",\"values\":[\"%s\", \"%s\"]},"
                                "{\"column\":\"cnt_at\",\"type\":\"between\",\"values\":[\"%s\", \"%s\"]}]}"
                                % (b,e,s,w))
        get_node_demand = get_bd['data']['data']['dataList']
        len_get_node_demand = len(get_node_demand)
        # print_in_log(get_bd['data']['data']['dataList'])
        self.print_in_log(date_parameter,'筛选条件：demand_date,cnt_at'+str(b)+str(e)+str(s)+str(w))
        if len_get_node_demand == 0:
            get_node_demand = pd.DataFrame()
            self.print_in_log(date_parameter,"读取数据失败，数据为0")
        else:
            get_node_demand = pd.DataFrame(get_node_demand)
            get_node_demand['dem_cirl_node_id'] = get_node_demand['dem_cirl_node_id'].apply(int)
            get_node_demand['sup_cirl_node_id'] = get_node_demand['sup_cirl_node_id'].apply(int)
            get_node_demand['sup_node_id'] = get_node_demand['sup_node_id'].apply(int)
            get_node_demand['resource_id'] = get_node_demand['resource_id'].apply(int)
            get_node_demand['dem_node_id'] = get_node_demand['dem_node_id'].apply(int)
            get_node_demand['duration_time'] = get_node_demand.apply(lambda row: row['duration_time']['value'],axis=1)
            get_node_demand['demand_date'] = get_node_demand.apply(lambda row: row['demand_date']['value'],axis=1)
            get_node_demand['cnt_at'] = get_node_demand.apply(lambda row: row['cnt_at']['value'], axis=1)
            # get_node_demand['cnt_at'] = get_node_demand['cnt_at'].dt.strftime('%Y-%m-%d')
            # get_node_demand['demand_date'] = get_node_demand['demand_date'].dt.strftime('%Y-%m-%d')
        self.print_in_log(date_parameter,'读取节点需求的数据，此项输入共有行数：'+str(len_get_node_demand)+'数据大小:'
                     + str(sys.getsizeof(get_node_demand)))
        return get_node_demand
    #获取节点描述的信息
    def node_description(self,date_parameter):
        get_bd = urllib.request("getBiDataParam",
                                "{\"selects\": [\"parent_node\",\"node_id\",\"effective_end_date\""
                                ",\"effective_start_date\"],"
                                "\"dataSet\": \"node_description\"}")
        get_node_description = get_bd['data']['data']['dataList']
        len_get_node_description = len(get_node_description)
        # print(get_bd['data']['data']['dataList'])
        self.print_in_log(date_parameter,'读取节点描述，此项输入共有行数：'+str(len_get_node_description))
        if len_get_node_description == 0:
            self.print_in_log(date_parameter,"读取数据失败，数据为0")
        else:
            get_node_description = pd.DataFrame(get_node_description)
            get_node_description['effective_end_date'] = get_node_description.apply\
                (lambda row: row['effective_end_date']['value'],axis=1)
            get_node_description['effective_start_date'] = get_node_description.apply\
                (lambda row: row['effective_start_date']['value'],axis=1)

            return get_node_description


    #需要查看业务数据的内容,包含，最小起订量，补货周期和预告时间
    #--------------------------------------------------------------------------------------------
    def business_data(self,date_parameter,s, w):
        get_bd = urllib.request("getBiDataParam",
                            "{\"selects\": [\"min_unit\",\"min_order\",\"unit\""
                            ",\"pre_date\",\"repl_cycle\",\"sup_node_id\",\"dem_node_id\",\"cnt_at\"],"
                            "\"dataSet\": \"business_data\","
                                "\"filters\":["
                                "{\"column\":\"cnt_at\",\"type\":\"between\",\"values\":[\"%s\", \"%s\"]}]}"
                                % (s, w))
        get_business_data = get_bd['data']['data']['dataList']
        len_get_business_data = len(get_business_data)
        # print(get_bd['data']['data']['dataList'])
        self.print_in_log(date_parameter,'读取业务数据，此项输入共有行数：'+str(len_get_business_data))
        if len_get_business_data == 0:
            self.print_in_log(date_parameter,"读取数据失败，数据为0")
        else:
            get_business_data = pd.DataFrame(get_business_data)
            get_business_data['cnt_at'] = get_business_data.apply(lambda row: row['cnt_at']['value'],axis=1)
            get_business_data['sup_node_id'] = get_business_data['sup_node_id'].apply(int)
            get_business_data['dem_node_id'] = get_business_data['dem_node_id'].apply(int)

            # get_business_data['cnt_at'] = get_business_data['cnt_at'].dt.strftime('%Y-%m-%d')
            return get_business_data


    #需要查看参数调节的参数
    #--------------------------------------------------------------------------------------------
    def parameter_revised(self,date_parameter,s, w):
        get_bd = urllib.request("getBiDataParam",
                            "{\"selects\": [\"forecast_date\",\"sup_node_id\",\"dem_node_id\",\"service_level\","
                            "\"relative_error\",\"ABS_error\",\"type\",\"resource_id\"],"
                            "\"dataSet\": \"parameter_revised\"}")
                            # "\"dataSet\": \"parameter_revised\","
                                # "\"filters\":["
                                # "{\"column\":\"resource_id\",\"type\":\"in\",\"values\":[\"3861\",\"3121\","
                                # "\"3187\",\"3246\",\"3188\",\"3124\",\"3268\",\"3856\",\"3344\",\"3269\","
                                # "\"3340\",\"3126\",\"3855\",\"3842\",\"3157\"]}"
                            
                                # "{\"column\":\"resource_id\",\"type\":\"in\",\"values\":[\"8112\",\"8234\","
                                # "\"8182\",\"8153\",\"8142\",\"8178\",\"8177\",\"8097\",\"8110\",\"8154\","
                                # "\"2854\",\"2855\",\"2917\",\"2443\",\"2616\", \"2398\",\"2725\",\"2440\",\"2762\",\"2438\"]}"

                            # "{\"column\":\"resource_id\",\"type\":\"between\",\"values\":[\"8100\",\"8300\"]},"
                            # "]}")

        get_parameter_revised = get_bd['data']['data']['dataList']

        len_get_parameter_revised = len(get_parameter_revised)
        # print(get_bd['data']['data']['dataList'])
        self.print_in_log(date_parameter,'读取参数调节的数据，此项输入共有行数：'+str(len_get_parameter_revised))
        if len_get_parameter_revised == 0:
            self.print_in_log(date_parameter,"读取数据失败，数据为0")
        else:
            get_parameter_revised = pd.DataFrame(get_parameter_revised)
            get_parameter_revised['forecast_date'] = get_parameter_revised.apply(lambda row: row['forecast_date']['value'], axis=1)
            # get_parameter_revised['forecast_date'] = get_parameter_revised['forecast_date'].dt.strftime('%Y-%m-%d')
            get_parameter_revised['sup_node_id'] = get_parameter_revised['sup_node_id'].apply(int)
            get_parameter_revised['dem_node_id'] = get_parameter_revised['dem_node_id'].apply(int)
            get_parameter_revised['resource_id'] = get_parameter_revised['resource_id'].apply(int)

            return get_parameter_revised

    #需要查看资源转换表，每个资源ID限制了其包装规格，从这个表取数据是用作取资源的转换和业务数据里面的最小补货量的对应单位
    #--------------------------------------------------------------------------------------------
    def resource_convert(self,date_parameter):
        get_bd = urllib.request("getBiDataParam",
                            "{\"selects\": [\"box_convert_rate\",\"box_unit\",\"box_id\",\"package_convert_rate\","
                            "\"package_unit\",\"package_id\",\"min_unit\",\"resource_id\",\"effective_end_date\",\"effective_start_date\"],"
                            "\"dataSet\": \"resource_convert\"}")
        get_resource_convert = get_bd['data']['data']['dataList']
        len_get_resource_convert = len(get_resource_convert)
        # print(get_bd['data']['data']['dataList'])
        self.print_in_log(date_parameter,'读取资源转换的数据，此项输入共有行数：'+str(len_get_resource_convert))
        if len_get_resource_convert == 0:
            self.print_in_log(date_parameter,"读取数据失败，数据为0")
        else:
            get_resource_convert = pd.DataFrame(get_resource_convert)
            get_resource_convert['effective_end_date'] = get_resource_convert.apply\
                (lambda row: row['effective_end_date']['value'],axis=1)
            get_resource_convert['effective_start_date'] = get_resource_convert.apply(
                lambda row: row['effective_start_date']['value'], axis=1)

            return get_resource_convert



if __name__ == '__main__':
    get_bd = get_bd()
    today = '20190515'
    today_01 = (pd.to_datetime(datetime.strptime(today, '%Y%m%d')) + timedelta(hours=0)).strftime(
        '%Y-%m-%d %H:%M:%S')
    tomorrow = (pd.to_datetime(datetime.strptime(today, '%Y%m%d')) + timedelta(hours=+24)).strftime(
        '%Y-%m-%d %H:%M:%S')
    # begin = datetime.now()
    # node_relationship_tran = get_bd.node_relationship_tran(today_01,tomorrow,today_01,tomorrow,1)
    # print('node_relationship_tran')
    # print(node_relationship_tran)
    # end = datetime.now()
    # print(end-begin)
    # node_relationship_cirl = get_bd.node_relationship_cirl(today_01,tomorrow)
    # # print('node_relationship_cirl')
    # # print(node_relationship_cirl)
    # node_demand = get_bd.node_demand(today_01,tomorrow,today_01,tomorrow,1)
    # print('node_demand')
    # print(node_demand)







