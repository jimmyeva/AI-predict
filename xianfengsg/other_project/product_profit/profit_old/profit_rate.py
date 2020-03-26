# -*- coding: utf-8 -*-
# @Time    : 2019/8/05 9:23
# @Author  : Ye Jinyu__jimmy
# @File    : profit_rate.py

import  math
import os
import time
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
# 注：设置环境编码方式，可解决读取数据库乱码问题
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
#关闭链路警告
pd.set_option('mode.chained_assignment', None)
#
def CompletePack_Simple(W, V, MW):#
    #存储最大价值的一维数组
    valuelist = [0] * (MW + 1)
    #开始计算
    for ii in range(len(W)):#从第一个物品
        for jj in range(MW + 1):#从重量0
            if jj >= W[ii]:#如果重量大于物品重量
                valuelist[jj] = max(valuelist[jj - W[ii]] + V[ii], valuelist[jj])#选中第ii个物品和不选中，取大的
    return '最大价值：', valuelist[-1]



#  也输出选择物品的编号以及个数
def CompletePack(W, V, MW):#每个商品可以选择多次
    #存储最大价值的一维数组
    valuelist = [0] * (MW + 1)
    #存储物品编号的字典

    codedict = {i: [] for i in range(0, MW + 1)}
    #开始计算
    for ii in range(len(W)):#从第一个物品m
        copyvalue = valuelist.copy()
        copydict = codedict.copy()
        # print('正在计算',int(ii),'个商品')
        for jj in range(MW + 1):#从重量0
            # print('正在计算的重量是', int(jj))
            if jj >= W[ii]:#如果重量大于物品重量
                cc = copyvalue[jj]
                x = round(jj - W[ii])       #索引必须要整数，但是在实际中的金额可能会存在小数的情况
                copyvalue[jj] = max(copyvalue[x] + V[ii], copyvalue[jj])#选中第ii个物品和不选中，取大的
                #输出被选中的物品编号
                if copyvalue[jj] > cc:
                    copydict[jj] = [ii]
                    y = round(jj - W[ii])
                    for hh in copydict[y]:
                        copydict[jj].append(hh)
        codedict = copydict.copy()#更新
        valuelist = copyvalue.copy()#更新
    result = ''
    # print(list(set(copydict[MW])))
    total_cost = 0
    for hcode in sorted(list(set(copydict[MW]))):
        result += '%d,%d;' % (hcode, copydict[MW].count(hcode))
        weight_s = W[hcode] * copydict[MW].count(hcode)
        total_cost += weight_s
    return '最大价值：', valuelist[-1],total_cost,result




def CompletePack_min(W, V, MW,max_value):  # 不完全背包
    print('开始计算最小值')
    # 存储最大价值的一维数组

    valuelist = [max_value] * (MW + 1)
    valuelist[0] = 0

    # 存储物品编号的字典
    codedict = {i: [] for i in range(0, MW + 1)}
    # 开始计算
    for i in range(len(W)):  # 从第一个物品
        # print('正在计算',int(i),'个商品')
        copyvalue = valuelist.copy()
        copydict = codedict.copy()
        # start_num = math.ceil(MW/W[i])
        # print('start_num',start_num)
        for j in range(0,MW + 1):  # 从重量0
            if j >= W[i]:  # 如果重量大于物品重量
                cc = copyvalue[j]
                x =round(j - W[i])
                copyvalue[j] = min(copyvalue[x] + V[i], copyvalue[j])  # 选中第i个物品和不选中，取最小
                # 输出被选中的物品编号
                # # copydict[j] = [i]
                # print( copyvalue[j])
                # for hh in copydict[j - i]:
                #     copydict[j].append(hh)
                if copyvalue[j] < cc:       #逐步迭代操作
                    copydict[j] = [i]
                    y = round(j - W[i])
                    for hh in copydict[y]:       #将最小值记录在copydict内
                        copydict[j].append(hh)
        codedict = copydict.copy()  # 更新
        valuelist = copyvalue.copy()  # 更新
    result = ''
    total_cost = 0
    for hcode in sorted(list(set(copydict[MW]))):
        result += '%d,%d;' %(hcode, copydict[MW].count(hcode))
        weight_s = W[hcode] * copydict[MW].count(hcode)
        total_cost += weight_s
    return '最小价值：', valuelist[-1], total_cost, result


def algorithm_max_min_reward(weight,value, maxweight):
    start_time = time.time()
    result = CompletePack(weight, value, maxweight)
    total_cost = result[2]
    final_maxweight = maxweight
    while total_cost < final_maxweight:
        maxweight += 1
        result = CompletePack(weight, value, maxweight)
        total_cost = result[2]
    print(result)
    max_value = result[1]
    # print('max_value',max_value)
    # print(maxweight)
    result_min=CompletePack_min(weight, value, maxweight,max_value)
    print(result_min)
    end_time  =time.time()
    total_time = end_time-start_time
    print('DP算法总耗时:',total_time)
    return result[3],result_min[3],result[1],result[2],result_min[1],result_min[2]





def original_process(path):
    raw_data = pd.read_excel(path,encoding='utf_8_sig')
    raw_data['price_after_discount'] = raw_data['sales_price']*raw_data['discount_price_rate']
    raw_data['profit_after'] = raw_data['profit']*raw_data['discount_price_rate']
    raw_data['cost'] = raw_data['sales_price']-raw_data['profit']
    raw_data['origin_profit_rate'] = raw_data['profit_after']/raw_data['cost']
    raw_data['max_profit_rate'] =  pd.Series()
    raw_data['min_profit_rate'] =  pd.Series()
    return raw_data


def algorithm_max(raw_data):
    mid_data = raw_data[raw_data['participate'] == 'Y']
    if mid_data.empty==True:
        print('没有商品参加共享')
        pass
    else:
        print(mid_data)
        mid_data = mid_data.reset_index(drop=True)
        weight = mid_data['price_after_discount'].to_list()
        value = mid_data['profit_after'].to_list()
        maxweight = mid_data['threshold'].iloc[0]
        result_max,result_min,max_value,max_cost,min_value,min_cost = algorithm_max_min_reward(weight,value,maxweight)
        data_max = result_max.split(';')

        max_good_list = list()
        max_good_count = list()
        for i in data_max:
            mid_data_max = i.split(',')
            if len(mid_data_max) > 1:   #存在情况切割后的list里面有一个逗号
                max_good_list.append(mid_data_max[0])
                max_good_count.append(mid_data_max[1])
            else:
                pass

        data_min = result_min.split(';')
        min_good_list = list()
        min_good_count = list()
        for i in data_min:
            mid_data_min = i.split(',')
            if len(mid_data_min) > 1:   #存在情况切割后的list里面有一个逗号
                min_good_list.append(mid_data_min[0])
                min_good_count.append(mid_data_min[1])
            else:
                pass
        max_good_list = list(map(int, max_good_list))
        max_good_count = list(map(int, max_good_count))
        min_good_list = list(map(int, min_good_list))
        min_good_count = list(map(int, min_good_count))

        max_good_name_count  = pd.DataFrame(columns=['name','count'])
        min_good_name_count  = pd.DataFrame(columns=['name','count'])

        #设置一个循环用来计算每个sku的max_min毛利率
        for i in range(len(max_good_list)):
            index = max_good_list[i]
            name = mid_data['name'].iloc[index]
            cost = mid_data['cost'].iloc[index]
            original_profit = mid_data['profit_after'].iloc[index]
            discount = mid_data['discount'].iloc[index]
            final_profit = max_value - discount
            num_of_sku =max_good_count[i]
            final_profit_sku = ((original_profit * num_of_sku) * final_profit)/(max_value*num_of_sku)
            mid_data['max_profit_rate'].iloc[index] = float(final_profit_sku/cost)
            max_good_name_count['name'] = max_good_name_count['name'].append(pd.DataFrame({'name':name}),ignore_index=True)

        for x in range(len(min_good_list)):
            index = min_good_list[x]
            cost = mid_data['cost'].iloc[index]
            original_profit = mid_data['profit_after'].iloc[index]
            discount = mid_data['discount'].iloc[index]
            final_profit = min_value-discount
            num_of_sku =min_good_count[x]
            final_profit_sku = ((original_profit *num_of_sku)*final_profit)/(min_value*num_of_sku)
            mid_data['min_profit_rate'].iloc[index] = final_profit_sku/cost


        print('max_good_name_count',max_good_name_count)
        return mid_data

def algorithm_min(raw_data,mid_data):
    without_algorithm = raw_data[raw_data['participate'] == 'N']
    if without_algorithm.empty==True:
        print('没有商品不参加共享：')
        pass
    else:
        final_data = without_algorithm.append(mid_data)
        final_data = final_data.fillna(axis=1,method='pad')
        mid_data_N = raw_data[raw_data['participate'] == 'N']
        # print(mid_data_N)
        mid_data_N = mid_data_N.reset_index(drop=True)
        weight_N = mid_data_N['sales_price'].to_list()
        value_N = mid_data_N['profit'].to_list()
        maxweight_N = mid_data_N['threshold'].iloc[0]
        # print('weight_N',weight_N)
        # print('value_N',value_N)
        # print('maxweight_N',maxweight_N)
        result_max_N,result_min_N,max_value_N,max_cost_N,min_value_N,min_cost_N = algorithm_max_min_reward(weight_N,value_N,maxweight_N)
        data_max_N = result_max_N.split(';')


        max_good_list_N = list()
        max_good_count_N = list()
        for i in data_max_N:
            mid_data_max_N = i.split(',')
            if len(mid_data_max_N) > 1:   #存在情况切割后的list里面有一个逗号
                max_good_list_N.append(mid_data_max_N[0])
                max_good_count_N.append(mid_data_max_N[1])
            else:
                pass

        data_min_N = result_min_N.split(';')
        min_good_list_N = list()
        min_good_count_N = list()
        for i in data_min_N:
            mid_data_min_N = i.split(',')
            if len(mid_data_min_N) > 1:   #存在情况切割后的list里面有一个逗号
                min_good_list_N.append(mid_data_min_N[0])
                min_good_count_N.append(mid_data_min_N[1])
            else:
                pass
        max_good_list_N = list(map(int, max_good_list_N))
        max_good_count_N = list(map(int, max_good_count_N))
        min_good_list_N = list(map(int, min_good_list_N))
        min_good_count_N = list(map(int, min_good_count_N))

        name_max_list = []
        num_max_list = []
        name_min_list = []
        num_min_list = []
        #设置一个循环用来计算每个sku的max_min毛利率
        for z in range(len(max_good_list_N)):
            index_N = max_good_list_N[z]
            name_N = mid_data_N['name'].iloc[index_N]
            cost_N = mid_data_N['cost'].iloc[index_N]
            original_profit_N = mid_data_N['profit_after'].iloc[index_N]
            discount_N = mid_data_N['discount'].iloc[index_N]
            final_profit_N = max_value_N - discount_N
            num_of_sku_N =max_good_count_N[z]
            final_profit_sku_N = ((original_profit_N * num_of_sku_N) * final_profit_N)/(max_value_N*num_of_sku_N)
            mid_data_N['max_profit_rate'].iloc[index_N] = final_profit_sku_N/cost_N
            name_max_list.append(name_N)
            num_max_list.append(num_of_sku_N)
        max_good_name_count_N = pd.DataFrame({'name':name_max_list,'count':num_max_list})

        for k in range(len(min_good_list_N)):
            index_N = min_good_list_N[k]
            cost_N = mid_data_N['cost'].iloc[index_N]
            name_N = mid_data_N['name'].iloc[index_N]
            original_profit_N = mid_data_N['profit_after'].iloc[index_N]
            discount_N = mid_data_N['discount'].iloc[index_N]
            final_profit_N = min_value_N-discount_N
            num_of_sku_N =min_good_count_N[k]
            final_profit_sku_N = ((original_profit_N *num_of_sku_N)*final_profit_N)/(min_value_N*num_of_sku_N)
            mid_data_N['min_profit_rate'].iloc[index_N] = final_profit_sku_N/cost_N
            name_min_list.append(name_N)
            num_min_list.append(num_of_sku_N)
        min_good_name_count_N = pd.DataFrame({'name':name_min_list,'count':num_min_list})
        mid_data_N = mid_data_N.fillna(axis=1, method='pad')
        print('max_good_name_count_N',max_good_name_count_N)
        print('min_good_name_count_N', min_good_name_count_N)
        return mid_data_N

def main(path):
    raw_data = original_process(path)
    mid_max = algorithm_max(raw_data)
    final_data = algorithm_min(raw_data,mid_max)
    print(final_data)



main('D:/AI/xianfengsg/other_project/product_profit/毛利测算试用数据1.xlsx')