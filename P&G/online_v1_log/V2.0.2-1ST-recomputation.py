# -*- coding = utf-8 -*-
'''
@Time: 2019/5/16 21:21
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
                                         'node_relationship_tran'+str(date_parameter_transform)+'.csv', encoding="utf_8_sig")
    t1 = time.time()
    print_in_log('multi_read耗时:'+str(t1 - t0))
    # 得到类中的函数，然后再取类中的函数获取流转节点的节点关系
    # node_relationship_cirl = get_bd.node_relationship_cirl(Today_date,tomorrow)
    # node_relationship_cirl.to_csv('node_relationship_cirl.csv', encoding="utf_8_sig")
    node_relationship_cirl = pd.read_csv('./' + str(date_parameter) + '/' +
                                         'node_relationship_cirl'+str(date_parameter_transform)+'.csv', encoding="utf_8_sig")
    t2 = time.time()
    print_in_log('node_relationship_cirl耗时:'+str(t2 - t1))
    # 得到类中的函数，然后再取类中的函数获取节点需求
    # node_demand = multi_read_node_demand(get_bd.node_demand,date_parameter_transform)
    # node_demand.to_csv('node_demand.csv', encoding="utf_8_sig")
    node_demand = pd.read_csv('./' + str(date_parameter) + '/' +
                              'node_demand'+str(date_parameter_transform)+'.csv', encoding="utf_8_sig")
    t3 = time.time()
    print_in_log('node_demand耗时:'+str(t3 - t2))
    # 得到类中的函数，然后再取类中的函数获取节点描述
    # node_description = get_bd.node_description()
    # node_description.to_csv('node_description.csv', encoding="utf_8_sig")
    node_description = pd.read_csv('./' + str(date_parameter) + '/' +
                                   'node_description'+str(date_parameter_transform)+'.csv', encoding="utf_8_sig")
    t4 = time.time()
    print_in_log('node_description耗时:'+str(t4 - t3))
    #得到类中的函数，然后再取类中的函数获取节点资源（可下单）
    # node_resource = get_bd.get_node_resource(Today_date,tomorrow)
    # node_resource.to_csv('node_resource.csv', encoding="utf_8_sig")
    node_resource = pd.read_csv('./' + str(date_parameter) + '/' +
                                'node_resource'+str(date_parameter_transform)+'.csv', encoding="utf_8_sig")
    t5 = time.time()
    print_in_log('get_node_resource耗时:'+str(t5 - t4))
    # #得到类中的函数，然后再取类中的函数获取参数调节
    # parameter_revised = get_bd.parameter_revised(Today_date,tomorrow)
    # parameter_revised.to_csv('parameter_revised.csv', encoding="utf_8_sig")
    parameter_revised = pd.read_csv('./' + str(date_parameter) + '/' +
                                    'parameter_revised'+str(date_parameter_transform)+'.csv', encoding="utf_8_sig")
    t6 = time.time()
    print_in_log('parameter_revised耗时:'+str(t6 - t5))
    # #业务数据
    # business_data = get_bd.business_data(Today_date,tomorrow)
    # business_data.to_csv('business_data.csv', encoding="utf_8_sig")
    business_data = pd.read_csv('./' + str(date_parameter) + '/' +
                                'business_data'+str(date_parameter_transform)+'.csv', encoding="utf_8_sig")
    t7 = time.time()
    print_in_log('business_data耗时:'+str(t7 - t6))
    # #资源转化
    # resource_convert = get_bd.resource_convert()
    # resource_convert.to_csv('resource_convert.csv', encoding="utf_8_sig")
    resource_convert = pd.read_csv('./' + str(date_parameter) + '/' +
                                   'resource_convert'+str(date_parameter_transform)+'.csv', encoding="utf_8_sig")
    t8 = time.time()
    print_in_log('resource_convert耗时:'+str(t8 - t7))
    return node_relationship_tran,node_relationship_cirl,node_demand,\
           node_description,node_resource,parameter_revised,business_data,resource_convert




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

#定义一个函数，用来表示进行第一此补货逻辑的补货量的计算，对于不涉及抢货的补货，只需进行一次函数的计算即可，对于涉及抢货行为的补货，此函数可以用作
#第一次补货量的计算
def first_repl(process,pre_date,deal_arrive,mid_repl,res_id,response,resource_convert,today):
    ''' # 依据预告天数，依次计算出每个预告天数的条件下应该补货的数量,这里需要新建一个空的列表用于接受每个资源的补货信息'''
    repl_resource_mid = pd.DataFrame(
                    columns=['resource_id','dem_tran_node','sup_tran_node','dem_cirl_node',
                             'sup_cirl_node','resource_available_qty','replenish_qty','plan_qty',
                             'aging','unit','decision_date','update_date'])
    print_in_log('进程：'+str(process)+'，资源id：'+str(res_id)+'进行第一次的补货计算：')
    response_tran_node_id = response['tran_node_id'].tolist()
    response_sup_node_id = response['sup_node_id'].tolist()
    response_cirl_node_id = response['cirl_node_id'].tolist()
    response_sup_cirl_node = response['sup_cirl_node'].tolist()
    def time_convert(a):
        b = datetime.strptime(a, '%Y-%m-%d %H:%M:%S') + timedelta(hours=8)
        return b
    response['update_date'] = response.apply \
    (lambda row: time_convert(row['update_date']), axis=1)
    response_update_date = response['update_date'].tolist()

    for i in range(0, len(pre_date)):
        '''# 需要查看deal_arrive抵达日期 和 mid_repl需要补货满足的日期，因此自定义两个变量，用于dataframe的切片操作'''
        start = deal_arrive[i] - 1  # 减1的操作是解决在切片过程中，存在第一个数的索引是0的问题

        end = mid_repl[i] - 1

        '''# 这里对于装箱因子先采取四舍五入的方式'''
        repl_qty = response['plan_qty'].iloc[start:end].sum()
        available_resource_qty = response['available_resource_qty'].iloc[0]
        package_id,package_unit,repl_package,repl_qty_package,available_resource_qty_package,package_unit\
            = package_convert_rate(repl_qty,res_id,resource_convert,available_resource_qty)
        print_in_log('进程：'+ str(process)+ '，资源id：'+ str(res_id)+ '第'+str(i)+'个预告时间计算，进行资源转换后的结果')
        repl_mid_resource = pd.DataFrame(
            columns=['resource_id', 'dem_tran_node', 'sup_tran_node', 'dem_cirl_node',
                     'sup_cirl_node', 'resource_available_qty', 'replenish_qty', 'plan_qty',
                     'aging', 'unit', 'decision_date', 'update_date'])

        '''# 接下来将每个供需节点中的每个资源id对应的不同种的预告时间一起放入补货计划表中'''
        #这里的id使用装箱的ID，单位，可用库存，原始计划补货数量和建议的补货数量都是按照装箱ID的单位走
        repl_mid_resource['resource_id']       = pd.Series(package_id)
        repl_mid_resource['dem_tran_node']     = pd.Series(response_tran_node_id[i])
        repl_mid_resource['sup_tran_node']     = pd.Series(response_sup_node_id[i])
        repl_mid_resource['dem_cirl_node']     = pd.Series(response_cirl_node_id[i])
        repl_mid_resource['sup_cirl_node']     = pd.Series(response_sup_cirl_node[i])
        repl_mid_resource['resource_available_qty']  = pd.Series(available_resource_qty_package)
        #replenish_qty是原始补货数量的补货数量的单位转换
        repl_mid_resource['replenish_qty']     = pd.Series(repl_qty_package)
        #plan_qty建议补货数量按照四舍五入计算后的结果
        repl_mid_resource['plan_qty']          = pd.Series(repl_package)
        repl_mid_resource['aging']             = pd.Series(pre_date[i])
        '''# 这里unit 后面有空格，有可能是数据处理上面的问题，真正涉及到接口数据的时候需要再定义'''
        repl_mid_resource['unit']              = pd.Series(package_unit)
        repl_mid_resource['decision_date']     = pd.Series(today)
        repl_mid_resource['update_date']       = pd.Series(response_update_date[i])
        repl_resource_mid = repl_resource_mid.append(repl_mid_resource)
    print_in_log('进程：'+str(process)+'，资源id：'+str(res_id)+
          '这里是不涉及抢货逻辑下的供应计划，该dataframe是否为空'+str(repl_resource_mid.empty))
    return repl_resource_mid

#这里需要再定义一个抢货逻辑内部具体如何抢货的计算逻辑
def grab_the_goods(pre_date,deal_arrive,mid_repl,res_id,response,resource_convert,
                   count_repl,response_time,repl_cycle,today):
    ''' # 依据预告天数，依次计算出每个预告天数的条件下应该补货的数量,这里需要新建一个空的列表用于接受每个资源的补货信息'''
    repl_resource_mid = pd.DataFrame(
                    columns=['resource_id','dem_tran_node','sup_tran_node','dem_cirl_node',
                             'sup_cirl_node','resource_available_qty','replenish_qty','plan_qty',
                             'aging','unit','decision_date','update_date'])

    response_tran_node_id = response['tran_node_id'].tolist()
    response_sup_node_id = response['sup_node_id'].tolist()
    response_cirl_node_id = response['cirl_node_id'].tolist()
    response_sup_cirl_node = response['sup_cirl_node'].tolist()
    def time_convert(a):
        b = datetime.strptime(a, '%Y-%m-%d %H:%M:%S') + timedelta(hours=8)
        return b
    response['update_date'] = response.apply \
    (lambda row: time_convert(row['update_date']), axis=1)
    response_update_date = response['update_date'].tolist()

    for x in range(0, len(pre_date)):
        '''# 需要查看deal_arrive抵达日期 和 mid_repl需要补货满足的日期，因此自定义两个变量，用于dataframe的切片操作'''
        start = deal_arrive[x] - 1  # 减1的操作是解决在切片过程中，存在第一个数的索引是0的问题
        end = mid_repl[x] - 1

        #这里是计算了每一种预告时间下最初的一个补货数量
        repl_qty = response['plan_qty'].iloc[start:end].sum()
        #目前先暂定了一个多进行三次的额外补货
        # 设置一个空的列表，来表示在上次补货中一共准备了多少天的补货天数，目的是啊配合本次补货的，去重计算的时候使用，
        # 但是这个列表需要不要的更新，因为存在每次补货后对补货的天数和数量都会有变化
        repl_date = []
        for k in range(0, len(deal_arrive)):
            for v in range(deal_arrive[k], mid_repl[k] + 1):
                repl_date.append(v)
        # 这里是会发生几次额外的补货，抢货逻辑中的定义，每一次循环都意味着将要发生多发生一次额外的补货
        for i in range(1, count_repl):
            # t 用i * 是表示在下几次补货之后的情况
            t = [x * i for x in repl_cycle]
            # z是对预告天数进行了一次更新，及触发了下一次的补货将会对预告天数的大小发生改变
            z = [t[i] + pre_date[i] for i in range(0, len(t))]
            # 接下来是要看下一次补货将会准备多少天数的货，mid_deal_arrive，mid_mid_repl分别是第i补货应该备货处理的天数
            mid_deal_arrive, mid_mid_repl = process_repl_date(response_time, z, repl_cycle)
            # 得到当前的补货天数之后，还需要与之前的补货天数进行匹配，不要出现重合的天数
            # -------------------------------------------------------------------------------
            # 先对当前补货的天数进行计算，看看一共准备了多少天的补货天数
            # current_date当前抢货周期的内应该补货的天数
            current_date = []
            for k in range(0, len(mid_deal_arrive)):
                for v in range(mid_deal_arrive[k], mid_mid_repl[k] + 1):
                    current_date.append(v)
            # 将当前补货天数与之前的补货天数去重,即当前补货天数的日期不在之前补货日期内出现即可
            for i in range(0, len(repl_date)):
                while repl_date[i] in current_date:
                    current_date.remove(repl_date[i])
            # 在计算了每次额外的补货天数之后需要对 repl_date总补货的天数进行更新

            # 以上操作后，得到的补货天数是没有含有重复补货天数的数据，下面要做的操作就是把没有重合的日期，放入对应的补货周期内
            # 这里的思路是针对不同的补货提前期，存在不同种的补货策略，需要将多补货的数量与补货提前期一一对应，然后再看需要补货的天数，与几种补货周期的交集
            # k代表有几个不同预告时间，针对每个预告时间进行计算
            start_date = mid_deal_arrive[x]

            end_date = mid_mid_repl[x]
            # 设置一个空的列表，用来表示在这个预告时间下，需要多补的天数
            pre_date_grab = []
            for v in range(mid_deal_arrive[x], mid_mid_repl[x] + 1):
                pre_date_grab.append(v)

            # 然后再进行去重的操作,在历史的补货天数中已经存在了的补货天数将会被去重
            for i in range(0, len(repl_date)):
                while repl_date[i] in pre_date_grab:
                    pre_date_grab.remove(repl_date[i])

            # 需要定两个数值来保存再抢货逻辑的情况下，当前补货预告时间下的起始时间
            start_grab = min(pre_date_grab)
            end_grab = max(pre_date_grab)
            #从接口读取的这列数据是str类型因此需要对该列进行转数值操作
            response['service_level'] = pd.to_numeric(response['service_level'])
            #同时还要记录当前补货时间内的服务水平
            service_grab = response['service_level'].iloc[end_grab]

            repl_qty_grab = (response['plan_qty'].iloc[start_grab:end_grab].sum()) * service_grab
            #最后一步需要对一开始的计划补货的数量进行更新
            repl_qty += repl_qty_grab
            repl_date += current_date
            repl_date = list(set(repl_date))

        available_resource_qty = response['available_resource_qty'].iloc[0]
        package_id,package_unit,repl_package,repl_qty_package,available_resource_qty_package,package_unit\
            = package_convert_rate(repl_qty,res_id,resource_convert,available_resource_qty)
        repl_mid_resource = pd.DataFrame(
            columns=['resource_id', 'dem_tran_node', 'sup_tran_node', 'dem_cirl_node',
                     'sup_cirl_node', 'resource_available_qty', 'replenish_qty', 'plan_qty',
                     'aging', 'unit', 'decision_date', 'update_date'])
        # print_in_log(response)
        '''# 接下来将每个供需节点中的每个资源id对应的不同种的预告时间一起放入补货计划表中'''
        #这里的id使用装箱的ID，单位，可用库存，原始计划补货数量和建议的补货数量都是按照装箱ID的单位走
        repl_mid_resource['resource_id']       = pd.Series(package_id)
        repl_mid_resource['dem_tran_node']     = pd.Series(response_tran_node_id[x])
        repl_mid_resource['sup_tran_node']     = pd.Series(response_sup_node_id[x])
        repl_mid_resource['dem_cirl_node']     = pd.Series(response_cirl_node_id[x])
        repl_mid_resource['sup_cirl_node']     = pd.Series(response_sup_cirl_node[x])
        repl_mid_resource['resource_available_qty']  = pd.Series(available_resource_qty_package)
        repl_mid_resource['replenish_qty']     = pd.Series(repl_qty_package)
        repl_mid_resource['plan_qty']          = pd.Series(repl_package)
        repl_mid_resource['aging']             = pd.Series(pre_date[x])
        '''# 这里unit 后面有空格，有可能是数据处理上面的问题，真正涉及到接口数据的时候需要再定义'''
        repl_mid_resource['unit']              = pd.Series(package_unit)
        repl_mid_resource['decision_date']     = pd.Series(today)
        repl_mid_resource['update_date']       = pd.Series(response_update_date[x])
        repl_resource_mid = repl_resource_mid.append(repl_mid_resource)
    # print_in_log('经过抢货逻辑后的补货数据,该dataframe是否为空')
    return repl_resource_mid

# 这里需要再进行配额的计算，考虑抢货逻辑之后使用该函数，因为在供需关系的资源发布里面有是用最小粒度的节点资源来表示，因此在进行
# 可用配额的计算中需要，先将完成抢货逻辑后的最后的输出，先按照资源转换表转成，最小的资源单位
def nominal_algorithm(node_relationship_tran, resource_convert):
    # 思路是这样，需要先把节点关系（交易）的资源id，和可用配额进行一次计算，向下取整
    node_relationship_tran_noimal = node_relationship_tran[
        ['cnt_at', 'resource_id', 'sup_node_id', 'dem_node_id', 'nom_resource', 'current_status',
         'parent_status','belong_date']]
    node_relationship_tran_noimal = node_relationship_tran_noimal.reset_index(drop=True)
    node_relationship_tran_noimal.to_csv('./' + str(date_parameter) + '/' +
                                         'node_relationship_tran_noimal.csv',encoding="utf_8_sig")
    resource_convert['package_convert_rate'] = pd.to_numeric(resource_convert['package_convert_rate'])
    nominal_resource_convert = resource_convert[['resource_id', 'min_unit', 'package_id', 'package_unit',
                                                 'package_convert_rate']]
    nominal_resource_convert = nominal_resource_convert.reset_index(drop=True)
    nominal_resource_convert.to_csv('./' + str(date_parameter) + '/' +
                                    'nominal_resource_convert.csv', encoding="utf_8_sig")
    # 对资源进行匹配和计算可用配额

    nominal_resource_convert['resource_id'] = nominal_resource_convert['resource_id'].astype(int)
    node_relationship_tran_noimal['resource_id'] = node_relationship_tran_noimal['resource_id'].astype(int)
    merge_noimal = pd.merge(node_relationship_tran_noimal, nominal_resource_convert, on='resource_id',
                            how='left')
    merge_noimal['nom_resource'] = merge_noimal['nom_resource'] // merge_noimal['package_convert_rate']
    merge_noimal['resource_id'] = merge_noimal['package_id']
    merge_noimal = merge_noimal.drop(['package_id'], axis=1)
    # 为了能够进行匹配需要将名字进行更改
    merge_noimal = merge_noimal.rename(index=str,
                                       columns={'dem_node_id': 'dem_tran_node', 'sup_node_id': 'sup_tran_node'})
    print_in_log('')
    print_in_log(str(merge_noimal.empty))
    return merge_noimal

# merge_noimal这个dataframe是对配额进行了计算之后的结果
# 接下来就是对两个进行判断，对于比较的思路是这样：对于一对供需关系的预告时间和数量是一致的，因此分为三步走
# 1. 先根据给出的计划补货，来讲预告时间进行预告时间的从低到高的排序
# 2. 再依次去除预告时间，将预告时间转换成天数加到belong_date 上取cnt_at是那天的可用配额
# 3. 将配额的信息和计划补货进行合并，以资源id（package_id,供需节点和，并保留当前状态与父级节点的状态
# 首先需要对aging的字段进行变成int的处理
def nominal_compare(res_id,resource_convert,repl_resource, merge_noimal, tran_node_id,node_description):
    print_in_log('资源id：'+str(res_id)+ '正在进行配额的计算........................')
    #为了减少数据量在一开始计算的时候就对dataframe进行减少操作
    #先将原始的资源id换成已经是装箱因子的资源id
    resource_convert_mid = resource_convert[resource_convert['resource_id']== res_id]
    resource_convert_mid = resource_convert_mid.reset_index(drop=True)
    package_id = resource_convert_mid['package_id'].iloc[0]
    merge_noimal_01 = merge_noimal[merge_noimal['resource_id'] == package_id]
    merge_noimal_01 = merge_noimal_01.reset_index(drop=True)
    repl_resource_final = repl_resource
    repl_resource_final['aging'] = pd.to_numeric(repl_resource_final['aging'])
    # aging_counting这个函数只是看对应了几个预告时间并进行操作
    aging_counting = repl_resource.sort_values(by='aging', ascending=True)
    aging_counting = aging_counting.groupby(['aging'], as_index=False).agg(sum)
    aging_counting = aging_counting.reset_index(drop=True)
    i = 0
    while i < len(aging_counting):
        # 每进行一次循环判断就是有几中的预告时间
        aging_day = (aging_counting['aging'].iloc[i]).astype(float)
        print_in_log('资源id：'+str(res_id)+"这是第" + str(aging_day) + '预告时间的配额计算')
        counting_day = merge_noimal_01['belong_date'].iloc[0]
        print_in_log(str(type(aging_day)))
        if type(counting_day) == datetime:
            counting_day = counting_day.dt.strftime('%Y-%m-%d')
        else:
            pass
        counting_day = datetime.date(datetime.strptime(counting_day, '%Y-%m-%d %H:%M:%S'))
        print_in_log(str(counting_day))
        days_mid = counting_day + timedelta(days=aging_day)
        print_in_log(str(type(days_mid)))
        days = pd.datetime.strftime(days_mid, '%Y-%m-%d')
        merge_noimal_compare = merge_noimal_01[merge_noimal_01['cnt_at'] == days]
        # 这里拿到了每个aging对应的配额需要进行判断和父级节点的确认
        mid_repl_resource = repl_resource_final[repl_resource_final['aging'] == aging_day]
        #因为在循环的过程中，会将里面的值的格式进行了修改，因此在进行merge操作的时候，需要对格式进行修改
        mid_repl_resource[['resource_id', 'sup_tran_node', 'dem_tran_node']] = \
            mid_repl_resource[['resource_id', 'sup_tran_node', 'dem_tran_node']].astype(int)
        merge_noimal_compare[['resource_id', 'sup_tran_node', 'dem_tran_node']] = \
            merge_noimal_compare[['resource_id', 'sup_tran_node', 'dem_tran_node']].astype(int)
        nomial_compare = pd.merge(mid_repl_resource, merge_noimal_compare,
                                  on=['resource_id', 'sup_tran_node', 'dem_tran_node'], how='inner')
        # 每一次循环都会需要进行配额的判断并与父级节点的匹配
        nomial_compare_current_status = nomial_compare['current_status'].tolist()
        nomial_compare_parent_status = nomial_compare['parent_status'].tolist()
        nomial_compare_plan_qty = nomial_compare['plan_qty'].tolist()
        nomial_compare_nom_resource = nomial_compare['nom_resource'].tolist()
        k = 0
        while k < len(nomial_compare):
            print_in_log('资源id：'+str(res_id)+ '进行配额计算------------')
            current_status = nomial_compare_current_status[k]
            parent_status = nomial_compare_parent_status[k]
            plan_qty = nomial_compare_plan_qty[k]
            nomial_qty = nomial_compare_nom_resource[k]
            if current_status != 0:
                if plan_qty <= nomial_qty:
                    print_in_log('资源id：'+str(res_id)+'当前节点可用，并且节点资源满足补货量')
                    pass
                else:
                    # 如果计划的资源比可用配额的数量要大的话，需要找到父级节点的配额
                    # 找到父级节点
                    if parent_status == 0:  # 如果父级节点不可用
                        print_in_log('资源id：'+str(res_id)+'当前节点可用，但父级节点资源不可用')
                        repl_resource_final['plan_qty'].iloc[k] = nomial_qty
                    else:  # 如果父级节点可用
                        # 首先找到父级节点
                        parent_node = node_description[node_description['node_id'] == tran_node_id]
                        nomial_compare_parent = mid_repl_resource[mid_repl_resource['dem_node_id'] == parent_node]
                        # 上面的这部操作只是先找到父级节点对应的可用库存，下面的这部重命名操作是为了更好的做merge
                        nomial_compare_parent['dem_node_id'] = tran_node_id
                        nomial_compare_parent_total = pd.merge(mid_repl_resource, merge_noimal_compare,
                                                               on=['resource_id', 'sup_tran_node', 'dem_tran_node'],
                                                               how='inner')
                        # parent_nomial代表父级节点的可用配额
                        parent_nomial = nomial_compare_parent_total['nom_resource'].iloc[k]
                        print_in_log('资源id：'+str(res_id)+'当前节点可用，父级节点资源可用')
                        if plan_qty <= (nomial_qty + parent_nomial):
                            pass
                        else:
                            repl_resource_final['plan_qty'].iloc[k] = nomial_qty + parent_nomial
            else:
                print_in_log('资源id：'+str(res_id)+'当前节点不可用，补货量为0')
                repl_resource_final['plan_qty'].iloc[k] = 0
            k += 1
        i += 1
    print_in_log('-----------------------------------------------------------------')
    print_in_log('资源id：'+str(res_id)+'配额计算完成.......')
    return repl_resource_final

#这里是主函数供需计划的，内部计算的方法
def sup_dem_plan_inside(resource_id,schedule_df, pre_date, repl_cycle,
                        resource_convert, today, tran_node_id, node_description,
                        node_relationship_tran,merge_noimal,process):
    print_in_log('正在进行计算的进程是：'+str(process))
    print_in_log(str(resource_id))

    # 这里是对每个资源id进行补货的计算
    repl_resource = pd.DataFrame(columns=['resource_id', 'dem_tran_node', 'sup_tran_node', 'dem_cirl_node',
                                          'sup_cirl_node', 'resource_available_qty', 'replenish_qty', 'plan_qty',
                                          'aging', 'unit', 'decision_date', 'update_date'])
    for res_id in resource_id:
        print_in_log('当前正在计算如下的进程和资源id'+str(process) + str(res_id))
        '''# 先找到当前id的响应时间，此时的响应时间是当前id此时刻的响应时间'''
        response = schedule_df[schedule_df['resource_id'] == res_id]
        response.to_csv('./' + str(date_parameter) + '/' +
                        'response'+str(res_id)+'.csv',encoding="utf_8_sig")
        # print_in_log(response)
        '''# 注意这里进行了切片之后并不会改变dataframe的index，所以需要先对dataframe进行一次重置索引的操作'''
        response = response.reset_index(drop=True)
        response.to_csv('./' + str(date_parameter) + '/' +
                        'response_last'+str(res_id)+'.csv', encoding="utf_8_sig")
        # 因为对于同一个供需供需节点和资源id、来说的响应时间是一致的
        response_time = response['response'].iloc[0]
        deal_arrive, mid_repl = process_repl_date(response_time, pre_date, repl_cycle)
        print_in_log('deal_arrive')
        print_in_log(str(deal_arrive))
        print_in_log('mid_repl')
        print_in_log(str(mid_repl))
        # 接下来的这部操作，在下一次服务水平提前前，一共会进行几次的补货，count_repl代表需要进行几次补货
        # c代表的是下一次达到当前服务水平需要经过几次额外的补货
        count_repl = 0
        c = 0
        '''# 目前只考虑某个供需节点和资源id只存在一种的补货周期'''
        repl_cyc = repl_cycle[0]
        # 因为要考虑到多的那几部的补货周期，需要使得在后续的计算过程中不会出现大于索引的情况
        # print_in_log(response)
        pre_date_max = max(pre_date)
        max_date = response_time + repl_cyc + pre_date_max
        while (c + max_date) < len(response):
            # 这里是对服务水平进行一次判断，这里需要增加一部判断
            if response['service_level'].iloc[c] <= response['service_level'].iloc[(c + repl_cyc)]:
                pass
            else:
                count_repl += 1
            c += repl_cyc
        print_in_log('资源id：'+str(res_id)+'的repl_cyc是:'+str(repl_cyc))
        '''# count_repl就是截至下一个服务水平下将会需要进行几次补货
        # 这里做一个判断，如果是count_repl是0的话意味着不需要进行抢货逻辑，如果不为0则需要进行抢货逻辑'''
        if count_repl == 0:
            # try:
            print_in_log('资源id：'+ str(res_id)+ '该资源不需要进行抢货逻辑')
            repl_resource_mid = first_repl(process, pre_date, deal_arrive, mid_repl, res_id, response,
                                           resource_convert,
                                           today)
            repl_resource = repl_resource.append(repl_resource_mid)
            repl_resource.to_csv('./' + str(date_parameter) + '/' +
                                 'repl_resource'+str(res_id)+'.csv',encoding="utf_8_sig")
            '''#对于count_repl不为0的资源id，代表需要进行抢货逻辑'''
            # -=================================================================================================
            repl_resource = nominal_compare(res_id,resource_convert,repl_resource, merge_noimal, tran_node_id, node_description)

        else:
            # 对于需要进行抢货的供需关系，除了计算第一次补货的的基础最小单位的数量之外，还要再算后几次额外补货的数量，最后再进行转换
            print_in_log('资源id：'+str(res_id)+'该资源需要进行抢货逻辑的计算')
            repl_resource_mid = grab_the_goods(pre_date, deal_arrive, mid_repl, res_id,
                                               response, resource_convert, count_repl, response_time, repl_cycle,
                                               today)
            repl_resource = repl_resource.append(repl_resource_mid)
            # -=================================================================================================
            repl_resource = nominal_compare(res_id,resource_convert,repl_resource,merge_noimal, tran_node_id, node_description)
        print_in_log('已经计算完成的id:' + str(res_id))
    return repl_resource


#定义多进程函数，用来进行分进程进行数据处理
def sup_dem_plan_multi(resource_id, schedule_df, pre_date, repl_cycle,
                        resource_convert, today, tran_node_id, node_description,node_relationship_tran):
    length_res = len(resource_id)
    merge_noimal = nominal_algorithm(node_relationship_tran, resource_convert)
    merge_noimal.to_csv('./' + str(date_parameter) + '/' +
                        'merge_noimal.csv',encoding="utf_8_sig")
    n = 12  #这里计划设置60个线程进行计算
    step = int(math.ceil(length_res / n))
    lists = {}
    for i in range(1, n + 1):
        lists[(i - 1) * step + 1] = i * step + 1

    pool = multiprocessing.Pool(processes=12)  # 创建60个进程
    results = []
    data = pd.DataFrame(columns=['resource_id', 'dem_tran_node', 'sup_tran_node', 'dem_cirl_node',
                                          'sup_cirl_node', 'resource_available_qty', 'replenish_qty', 'plan_qty',
                                          'aging', 'unit', 'decision_date', 'update_date'])
    for start_user, end_user in lists.items():
        print_in_log('进行这段资源id的供应计划的计算' + str(start_user) +',' + str(end_user))
        resource_id_multi_process = resource_id[(start_user-1):(end_user-1)]
        print_in_log('进程：'+str(start_user)+'的资源id有：'+str(resource_id_multi_process))
        # resource_id_multi_process.to_csv('resource_id_multi_process'+str(start_user)+'.csv')
        print_in_log(str(start_user) + str(end_user))
        results.append(pool.apply_async(sup_dem_plan_inside,
                                        args=(resource_id_multi_process,schedule_df, pre_date, repl_cycle,
                                              resource_convert, today, tran_node_id, node_description,
                                              node_relationship_tran,merge_noimal,start_user)))
    pool.close()  # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
    pool.join()  # 等待进程池中的所有进程执行完毕

    for i in results:
        a = i.get()
        data = data.append(a, ignore_index=True)
    return data



#此函数是最后补货的数量计算的主函数，这里会涉及到抢货的行为
def sup_dem_plan(merge_plan, business_effective,sigma, sup_node_id,
                 tran_node_id,today,node_description,resource_convert,
                 node_relationship_tran):
    #主函数的输入包括，计划dataframe，处理后的业务数据，额外增加的补货天数，如果没有涉及抢货逻辑的话是零和供需节点

    schedule_df_mid = merge_plan[merge_plan['sup_node_id'] == sup_node_id]
    schedule_df_mid = schedule_df_mid[schedule_df_mid['tran_node_id'] == tran_node_id]
    schedule_df = pd.DataFrame(columns=['available_resource_qty','cirl_node_id','cnt_at','resource_id',
                                       'tran_node_id','up_cirl_error_response','up_cirl_resp_time',
                                       'up_tran_error_response','up_tran_resp_time','update_date',
                                       'response','sup_node_id','unit','nom_resource','response_time',
                                       'service_level','current_status','parent_status','sup_cirl_node',
                               'demand_qty','plan_qty','vendible_resource_qty','forecast_error'])
    schedule_df['available_resource_qty']     = schedule_df_mid['available_resource_qty']
    schedule_df['cirl_node_id']               = schedule_df_mid['cirl_node_id']
    schedule_df['cnt_at']                     = schedule_df_mid['cnt_at']
    schedule_df['resource_id']                = schedule_df_mid['resource_id']
    schedule_df['tran_node_id']               = schedule_df_mid['tran_node_id']
    schedule_df['up_cirl_error_response']     = schedule_df_mid['up_cirl_error_response']
    schedule_df['up_cirl_resp_time']          = schedule_df_mid['up_cirl_resp_time']
    schedule_df['up_tran_error_response']     = schedule_df_mid['up_tran_error_response']
    schedule_df['up_tran_resp_time']          = schedule_df_mid['up_tran_resp_time']
    schedule_df['update_date']                = schedule_df_mid['update_date']
    schedule_df['response']                   = schedule_df_mid['response']
    schedule_df['sup_node_id']                = schedule_df_mid['sup_node_id']
    schedule_df['unit']                       = schedule_df_mid['unit']
    schedule_df['nom_resource']               = schedule_df_mid['nom_resource']
    schedule_df['response_time']              = schedule_df_mid['response_time']
    schedule_df['service_level']              = schedule_df_mid['service_level']
    schedule_df['current_status']             = schedule_df_mid['current_status']
    schedule_df['parent_status']              = schedule_df_mid['parent_status']
    schedule_df['sup_cirl_node']              = schedule_df_mid['sup_cirl_node']
    schedule_df['demand_qty']                 = schedule_df_mid['demand_qty']
    schedule_df['plan_qty']                   = schedule_df_mid['plan_qty']
    schedule_df['vendible_resource_qty']      = schedule_df_mid['vendible_resource_qty']
    schedule_df['forecast_error']             = schedule_df_mid['forecast_error']
    schedule_df = schedule_df.reset_index()
    schedule_df['demand_qty'] = schedule_df['demand_qty'].astype(float)
    schedule_df['plan_qty'] = schedule_df['plan_qty'].astype(float)
    schedule_df['forecast_error'] = schedule_df['forecast_error'].astype(float)
    schedule_df['vendible_resource_qty'] = schedule_df['vendible_resource_qty'].astype(float)
    # schedule_df.to_csv('schedule_df_old' + str(sup_node_id) + str(tran_node_id) + '.csv', encoding="utf_8_sig")

    #这里需要对下游节点的需求进行总的汇总
    schedule_df = schedule_df.groupby(['available_resource_qty', 'cirl_node_id', 'cnt_at', 'resource_id',
                                       'tran_node_id', 'up_cirl_error_response', 'up_cirl_resp_time',
                                       'up_tran_error_response', 'up_tran_resp_time', 'update_date',
                                       'response', 'sup_node_id', 'unit', 'nom_resource', 'response_time',
                                       'service_level', 'current_status', 'parent_status', 'sup_cirl_node']
                                      , as_index=False)
    # schedule_df.to_csv("C:/Users/jinyu.ye/Desktop/schedule_df.csv", encoding="utf_8_sig")
    schedule_df = schedule_df['plan_qty','forecast_error','demand_qty','vendible_resource_qty'].agg(sum)
    # schedule_df.to_csv('schedule_df_old'+str(sup_node_id)+str(tran_node_id)+'.csv', encoding="utf_8_sig")
    #采用以上的匹配规则后悔出现需要对索引进行重置
    schedule_df = schedule_df.reset_index()
    schedule_df.to_csv('./' + str(date_parameter) + '/' +
                       'schedule_df_last'+str(sup_node_id)+str(tran_node_id)+'.csv', encoding="utf_8_sig")
    # schedule_df.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v2---补货/'
    #                        'v2---输入-输出/人工造的数据/test/schedule_df'+str(sup_node_id)+'.csv', encoding="utf_8_sig")

    '''# 匹配需要的供需节点，保留该有的业务数据，并且对保留的业务数据中的预告时间进行排序'''
    business_save = business_effective[business_effective['sup_node_id'] == sup_node_id]
    business_save = business_save[business_save['dem_node_id'] == tran_node_id]
    #此步骤的操作是为了
    business_save = business_save.sort_index(by=["pre_date"], ascending=[True])
    # business_save.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v2---补货/'
    #                        'v2---输入-输出/人工造的数据/test/business_save'+str(sup_node_id)+str(tran_node_id)+'.csv', encoding="utf_8_sig")
    '''# 先设定一个空列表，用来存储预告时间和补货周期'''
    pre_date = []
    repl_cycle = []
    k = 0
    while k < len(business_save):
        pre_date.append(business_save['pre_date'].iloc[k])
        repl_cycle.append(business_save['repl_cycle'].iloc[k] +sigma)
        k += 1
        '''# 接下来需要对补货预告时间和补货周期进行补货的逻辑计算，这里会存在每个资源id有着不同的响应时间，因此需要分SKU进行计算'''
    #这个compare_cycle是用于后面判断增加补货天数的时候使用
    compare_cycle = repl_cycle[0]

    resource_id = list(set(schedule_df['resource_id']))
    '''# 新建一个空的裂变用于接受每个资源id对应的补货数据，包括不同的补货'''
    if len(resource_id) == 0:
        repl_resource = pd.DataFrame(columns=['empty'])
        compare_cycle = 0
    else:
        repl_resource = sup_dem_plan_multi(resource_id, schedule_df, pre_date, repl_cycle,
                            resource_convert, today, tran_node_id, node_description,node_relationship_tran)

    return repl_resource,compare_cycle


# '''#这里需要定义一个函数来判断是否达到最小起订量'''
def min_order_policy(repl_dataframe,business_dataframe,resource_convert):
    #需要先对补货的数据进行改变，这里的规则是全部都按照package_unit进行补货的，现在需要换成最大的单位
    #首先是对取到的业务数据进行更新和转换，只要保留package_id和package_convert与box_convert的比值即可
    resource_convert_mid = resource_convert[['package_id','package_convert_rate','box_convert_rate']]
    resource_convert_mid['package_convert_rate'] = pd.to_numeric(resource_convert_mid['package_convert_rate'])
    resource_convert_mid['box_convert_rate'] = pd.to_numeric(resource_convert_mid['box_convert_rate'])
    resource_convert_mid['package_box_convert'] = resource_convert_mid['package_convert_rate']/resource_convert_mid['box_convert_rate']
    resource_convert_mid = resource_convert_mid.rename(index=str,
                                       columns={'package_id': 'resource_id'})
    # 因为在循环的过程中，会将里面的值的格式进行了修改，因此在进行merge操作的时候，需要对格式进行修改
    repl_dataframe[['resource_id']] = repl_dataframe[['resource_id']].astype(int)
    resource_convert_mid[['resource_id']] = resource_convert_mid[['resource_id']].astype(int)
    repl_dataframe_convert = pd.merge(repl_dataframe,resource_convert_mid,on=['resource_id'],how='inner')
    #这部操作是将补货的数量转成最大规格对应的资源id
    repl_dataframe_convert['plan_qty'] = repl_dataframe_convert['plan_qty'] * repl_dataframe_convert['package_box_convert']
    #需要先去业务数据进行取出有用的列
    business_dataframe_mid = business_dataframe[['dem_node_id','sup_node_id','min_order','min_unit']]
    business = business_dataframe_mid.drop_duplicates(subset=['sup_node_id','dem_node_id'],keep='first',inplace=False)
    '''目的是只要知道业务数据里面的最小起订量的补货单位和数量即可'''
    dem_node_id = repl_dataframe['dem_tran_node'].iloc[0]
    sup_node_id = repl_dataframe['sup_tran_node'].iloc[0]
    min_order_qty = business[business['dem_node_id']==dem_node_id]
    min_order_qty = min_order_qty[min_order_qty['sup_node_id'] == sup_node_id]
    min_qty = min_order_qty['min_order'].iloc[0]
    ''' #查看满足该单位的补货量，一共有多少'''
    plan_qty = repl_dataframe_convert['plan_qty'].sum()
    # print_in_log('计划总订货量'+ str(plan_qty))
    if plan_qty >= min_qty :
        print_in_log('满足最小起订量')
        return True
    else:
        print_in_log("未满足最小起订量")
        return False

#这里再定义一个函数专门来拿供需节点的补货周期的数值
def reple_cycle(tran_node_id,sup_node_id,business_data):
    business_data_cycle = business_data[['dem_node_id','sup_node_id','repl_cycle']]
    business_data_cycle = business_data_cycle[business_data_cycle['dem_node_id']==tran_node_id]
    business_data_cycle = business_data_cycle[business_data_cycle['sup_node_id'] == sup_node_id]
    business_data_cycle = business_data_cycle.drop_duplicates()
    compare_cycle_count = business_data_cycle['repl_cycle'].iloc[0]
    return compare_cycle_count


#定义主函数用于主函数进行计算,来返回需要的要最后的dataframe总的集合
def main_function(date_parameter,alpha,sigma,gamma,node_relationship_tran, node_relationship_cirl, node_demand,
    node_description, node_resource, parameter_revised, business_data, resource_convert,dem_node,sup_node):
#再开始程序就选定一个一个计算开始的时间
    now = time.time()
    local_time = time.localtime(now)
    today = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
    print_in_log('当前决策的时间为' +str(today))
    t0 = time.time()
    # #这一步是依据服务水平和相应时间等对节点需求进行修正
    # demand_response = demand_last(alpha,sigma,gamma, node_demand, node_relationship_tran, node_relationship_cirl,
    #                               parameter_revised)
    # demand_response.to_csv('demand_response.csv', encoding="utf_8_sig")
    demand_response = pd.read_csv('./' + str(date_parameter) + '/' +
                                  'demand_response'+str(date_parameter)+'.csv')
    t1 = time.time()
    print_in_log('demand_last耗时：'+str(t1 - t0))

    #这是不会补货计划进行第一步的计算
    # plan_resource = plan_supply(demand_response, node_resource)
    # plan_resource.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v2---补货/'
    #                        'v2---输入-输出/人工造的数据/test/plan_resource.csv', encoding="utf_8_sig")
    # plan_resource.to_csv('plan_resource.csv', encoding="utf_8_sig")
    plan_resource = pd.read_csv('./' + str(date_parameter) + '/' +
                                'plan_resource'+str(date_parameter)+str(dem_node)+'.csv')
    t2 = time.time()
    print_in_log('plan_supply耗时：'+str(t2 - t1))

    mid_business, merge_plan_01, business_effective = business_plan(node_relationship_tran, plan_resource, business_data)

    # mid_business.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v2---补货/'
    #                      'v2---输入-输出/人工造的数据/test/mid_business.csv', encoding="utf_8_sig")

    # merge_plan_01.to_csv('merge_plan_01.csv', encoding="utf_8_sig")
    t3 = time.time()
    print_in_log('business_plan耗时：'+str(t3 - t2))


    # merge_plan = up_cirl_merge(merge_plan_01, node_relationship_cirl)
    # merge_plan.to_csv('merge_plan_up_cirl_merge.csv', encoding="utf_8_sig")
    merge_plan = pd.read_csv('./' + str(date_parameter) + '/' +
                             'merge_plan'+str(date_parameter)+str(dem_node)+'.csv')
    # merge_plan.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v2---补货/'
    #               'v2---输入-输出/人工造的数据/test/merge_plan.csv', encoding="utf_8_sig")
    t4 = time.time()
    print_in_log('up_cirl_merge耗时：'+str(t4 - t3))

    # without_prediction_data = without_prediction(node_demand, node_resource, node_relationship_tran,
    #                                              node_relationship_cirl, resource_convert,business_data)


    without_prediction_data = pd.read_csv('./' + str(date_parameter) + '/' +
                                          'without_prediction_data'+str(date_parameter)+str(dem_node)+'.csv')
    t5 = time.time()
    print_in_log('without_prediction耗时：'+str(t5 - t4))

    # 设置一个总的函数来记录最后的所有的补货数据
    final_repl_toal = pd.DataFrame(
        columns=['resource_id', 'dem_tran_node', 'sup_tran_node', 'dem_cirl_node', 'sup_cirl_node'
            , 'resource_available_qty', 'replenish_qty', 'plan_qty', 'aging', 'unit',
                 'decision_date', 'update_date'])

    print_in_log('本次智能补货共有' + str(len(mid_business)) + '对的供需关系进行计算')
    mid_business = mid_business.reset_index(drop =True)
    mid_business_dem_node_id =mid_business['dem_node_id']
    mid_business_sup_node_id =mid_business['sup_node_id']
    # while v < len(mid_business):

        # 这里的每一层的每一个循环都代表着一个供需节点的补货计划
    print_in_log('这是这对供需关系对的：' + str(dem_node)+ str(sup_node) + '对，供需节点的补货计算')
    tran_node_id = dem_node
    sup_node_id = sup_node
    # print_in_log('当前的交易节点的供方是:' + str(sup_node_id) + '，交易节点的需方是： ' + str(tran_node_id))
    '''#以下这部操作是针对不同的供需节点进行补货计算'''
    final_repl = pd.DataFrame(columns=['resource_id', 'dem_tran_node', 'sup_tran_node', 'dem_cirl_node', 'sup_cirl_node'
            , 'resource_available_qty', 'replenish_qty', 'plan_qty', 'aging', 'unit', 'decision_date', 'update_date'])
    i = 0
    compare_cycle_count = reple_cycle(tran_node_id, sup_node_id, business_data)

    repl_resource, compare_cycle = sup_dem_plan(merge_plan, business_effective, i, sup_node_id,
                                                tran_node_id,today,node_description,
                                                resource_convert,node_relationship_tran)
    if repl_resource.empty == True:
        print_in_log('当前供需关系对，的资源是空值')
        final_repl = pd.DataFrame(columns=['resource_id', 'dem_tran_node', 'sup_tran_node', 'dem_cirl_node', 'sup_cirl_node'
            , 'resource_available_qty', 'replenish_qty', 'plan_qty', 'aging', 'unit', 'decision_date', 'update_date'])
    else:
        # print_in_log('这是第一次计划后应该的补货结果')
        # print_in_log(repl_resource)
        result = min_order_policy(repl_resource, business_effective, resource_convert)
        '''#得到第一批的补货建议之后，需要做一个判断并循环的逻辑，如果一次补货建议中并未能满足最小起订量，需要增加补货周期，然后再进行判断'''
        # 这里的i就代表需要增加一天的补货周期再进行计算，直到计算出来的补货数量满足最小起订量为止
        while result != True:
            i += 1
            print_in_log('这是增加了第' + str(i) + '天的补货计划')
            repl_resource, compare_cycle = sup_dem_plan(merge_plan, business_effective, i,
                                                        sup_node_id, tran_node_id,today,
                                                        node_description,resource_convert
                                                        ,node_relationship_tran)

            result = min_order_policy(repl_resource, business_effective, resource_convert)
            if i <= compare_cycle_count:
                pass
            else:
                print_in_log('经过' + str(compare_cycle_count) + '次的额外补货数量，仍然不能达到最小补货量')
                repl_resource, compare_cycle = sup_dem_plan(merge_plan, business_effective, i, sup_node_id,
                                                            tran_node_id,today,
                                                            node_description,resource_convert,node_relationship_tran)
                # final_repl = final_repl.append(repl_resource)
                result = True
        final_repl = final_repl.append(repl_resource)
    final_repl_toal = final_repl_toal.append(final_repl)
    if without_prediction_data.empty == True:
        # print_in_log("全部SKU进行了预测")
        final_repl_toal = final_repl_toal.fillna(method='ffill')
    else:
        # print_in_log("有部分的sku未进行预测，需要进行合并")
        final_repl_toal = final_repl_toal.append(without_prediction_data)
        final_repl_toal = final_repl_toal.append(without_prediction_data)
        final_repl_toal = final_repl_toal.drop_duplicates(subset=
                                        ['aging', 'dem_cirl_node', 'dem_tran_node', 'resource_id', 'sup_cirl_node',
                                         'sup_tran_node'], keep='first')
        final_repl_toal = final_repl_toal.fillna(method='ffill')

    print_in_log(str(type(final_repl_toal['update_date'].iloc[0])))
    print_in_log(str(type(final_repl_toal['update_date'])))
    final_repl_toal['update_date'] = pd.to_datetime(final_repl_toal['update_date'], format='%Y-%m-%d %H:%M:%S')
    print_in_log(str(type(final_repl_toal['update_date'].iloc[0])))
    print_in_log(str(type(final_repl_toal['update_date'])))
    final_repl_toal['update_date'] = final_repl_toal['update_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    print_in_log(str(type(final_repl_toal['update_date'])))
    t6 = time.time()
    print_in_log('最后循环耗时：'+str(t6 - t5))
    # print_in_log(final_repl_toal)
    return final_repl_toal


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

#<=============================================================================写入数据库
def connectdb():
    print_in_log('连接到mysql服务器...')
    db = pymysql.connect(host="172.16.4.7",database="supply_chain",user="bi_user",password="RL9FCS4@QTrmOsRk",port=3306,charset='utf8')
    print_in_log('数据库连接成功')
    return db

#《----------------------------------------------------------------------------删除重复日期数据
def drop_data(db,date_parameter):
    cursor = db.cursor()
    # Parameter_cnt = datetime.date(datetime.strptime(date_parameter, '%Y%m%d'))
    # predict_time = "date'%s'" % Parameter_cnt
    # sql = """delete from core_out_rp_replenishment where date(decision_date) = %s""" % predict_time
    # cursor.execute(sql)
    date_parameter_transform = pd.to_datetime(datetime.strptime(date_parameter, '%Y%m%d'))
    date_parameter_final = date_parameter_transform.strftime("%Y-%m-%d")
    sql = """delete from core_out_rp_replenishment where 
    decision_date = str_to_date(\'%s\','%%Y-%%m-%%d')"""%(date_parameter_final)
    cursor.execute(sql)


def insertdb(db,data):
    cursor = db.cursor()
    data = data.drop(['Unnamed: 0'], axis=1)
    data_list = data.values.tolist()
    print_in_log(str(data_list))
    sql = """INSERT IGNORE INTO core_out_rp_replenishment_mid (aging,
                                                            decision_date,
                                                            dem_cirl_node_id,
                                                            dem_tran_node_id,
                                                            plan_qty,
                                                            replenish_qty,
                                                            resource_available_qty,
                                                            resource_id,
                                                            sup_cirl_node_id,
                                                            sup_tran_node_id,
                                                            unit,
                                                            inv_update_time
                                                              ) VALUES ('%s',%s,'%s','%s','%s','%s','%s',
                                                            '%s','%s','%s',%s,%s)"""
    try:
        # db.commit()
        cursor.executemany(sql, data_list)
        db.commit()
    except (AttributeError, pymysql.OperationalError):
        db = pymysql.connect(host="172.16.4.7", database="supply_chain", user="bi_user", password="RL9FCS4@QTrmOsRk",
                             port=3306, charset='utf8')
        print_in_log('数据宕机，重新连接成功')
        cursor.executemany(sql, data_list)
        db.commit()
    except OSError as reason:
        print_in_log('出错原因是%s' % str(reason))
        db.rollback()

def closedb(db):
    db.close()
def main():
    date_parameter,dem_node,sup_node = date_parameter_read()
    # date_parameter = '20190423'
    node_relationship_tran, node_relationship_cirl, node_demand,\
    node_description, node_resource, parameter_revised, business_data, resource_convert = get_data(date_parameter)
    # drop_data(db, date_parameter)
    data = main_function(date_parameter,0.95,0.95,0.95,node_relationship_tran, node_relationship_cirl, node_demand,
    node_description, node_resource, parameter_revised, business_data, resource_convert,dem_node,sup_node)
    print(data)
    data.to_csv('./' + str(date_parameter) + '/' +
                'data'+str(date_parameter)+str(dem_node)+'.csv', encoding="utf_8_sig")
    db = connectdb()
    # data.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v2---补货/'
    #                        'v2---输入-输出/人工造的数据/test/data.csv', encoding="utf_8_sig")
    if data.empty:
        print_in_log("The data frame is empty")
        closedb(db)
    else:
        insertdb(db,data)
        closedb(db)
        end = datetime.now()
        print_in_log(str(end - begin))
        print("result:1")
#《============================================================================主函数入口
if __name__ == '__main__':
    try:
        st = time.time()
        begin = datetime.now()
        main()
        ed = time.time()
        print_in_log('总程序耗时：'+str(ed - st))
        print("result:1")
    except OSError as reason:
        print_in_log('出错原因是%s'+str(reason))
        print("result:0")






