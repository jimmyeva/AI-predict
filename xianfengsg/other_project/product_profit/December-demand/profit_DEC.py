# -*- coding: utf-8 -*-
# @Time    : 2019/8/20 18:58
# @Author  : Ye Jinyu__jimmy
# @File    : profit_temp.py
import  math
import os
import time
import pandas as pd
import numpy as np
from itertools import combinations
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
# 注：设置环境编码方式，可解决读取数据库乱码问题
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
#关闭链路警告
pd.set_option('mode.chained_assignment', None)
import tqdm
import redis


#读取原始数据
# data = pd.read_excel('D:/AI/xianfengsg/other_project/product_profit/test_model.xlsx')


def read_excel1(path):
    data_xls = pd.ExcelFile(path)
    data=pd.DataFrame()
    for name in data_xls.sheet_names:
        df=data_xls.parse(sheet_name=name,header=None)
        data = data.append(df, ignore_index=True)
    return data

def original_process(path):
    raw_data = read_excel1(path)
    raw_data.columns = ['name','code','sales_price','cost',
                        # 'useless',
                        'discount_price_rate','discount','threshold','participate',
                        'seckill','second_kill','customer','store_discount','store_threshold','discont_rate', 'channel']
    raw_data_01 =raw_data[raw_data['channel'] == '美团']
    raw_data_02 = raw_data[raw_data['channel'] == '饿了么']
    raw_data = pd.concat([raw_data_01,raw_data_02],ignore_index=True)
    raw_data.reset_index(drop=True)
    # raw_data.drop(raw_data.index[0], inplace=True)
    raw_data.reset_index(drop=True)
    raw_data['sales_price'] =raw_data['sales_price'].astype(float)
    raw_data['price_after_discount'] = raw_data['sales_price']*raw_data['discount_price_rate']
    raw_data['profit'] = raw_data['sales_price']-raw_data['cost']
    raw_data['profit_after'] = raw_data['price_after_discount']-raw_data['cost']
    raw_data['origin_profit_rate'] = raw_data['profit_after']/raw_data['cost']
    return raw_data


#根据规则活动平台分为：饿了么要*0.9，美团要按照保底4，高于4取支付金额*0.88；
def separate_channel(data):
    data_meituan = data[data['channel'] == '美团']
    data_elema = data[data['channel'] == '饿了么']
    return data_elema,data_meituan

#---------------------------------------------------------将数据区分成三个不同的产品
#第一步是秒杀商品的集合
def process_data(data):
    seckill_data =  data[data['seckill'] == 'Y']
    seckill_name = seckill_data['name'].tolist()
    seckill_price_after_discount = seckill_data['price_after_discount'].tolist()
    seckill_profit_after = seckill_data['profit_after'].tolist()
    seckill_price = seckill_data['sales_price'].tolist()
    seckill_discont_rate = seckill_data['discount_price_rate'].tolist()
    #第二步是特价商品的集合
    mid_data = data[data['seckill'] == 'N']
    second_kill = mid_data[mid_data['second_kill'] <= 5]

    second_kill = second_kill.sort_values(by=['discount_price_rate'], ascending=True)
    second_kill.reset_index()
    def second_set(second_name):
        new_list = []
        for i in range(0, len(second_kill)):
            for k in range(0, second_num[i]):
                new_list.append(second_name[i])
        return new_list
    second_kill.reset_index()
    second_name_mid = second_kill['name'].tolist()
    second_price_after_discount_mid = second_kill['price_after_discount'].tolist()
    second_profit_after_mid = second_kill['profit_after'].tolist()
    second_price_mid = second_kill['sales_price'].tolist()
    second_num = second_kill['second_kill'].tolist()
    second_discont_rate_mid = second_kill['discount_price_rate'].tolist()
    second_name = second_set(second_name_mid)
    #计算特价商品的折后价格
    second_price_after_discount = second_set(second_price_after_discount_mid)
    second_profit_after = second_set(second_profit_after_mid)
    second_price = second_set(second_price_mid)
    second_discont_rate = second_set(second_discont_rate_mid)

    #最后一部是所有不含秒杀商品的其他所有商品
    '''原价商品有两个部分组成，一个是原来设置就是原价，另一部分是特价商品超过一定限额后的价格'''
    normal_mid_data = data[data['seckill'] == 'N']
    normal_data = normal_mid_data[normal_mid_data['second_kill'] > 4]
    normal_name = normal_data['name'].tolist()
    normal_price_after_discount = normal_data['price_after_discount'].tolist()
    normal_profit_after = normal_data['profit_after'].tolist()
    normal_data_back_second = normal_data[normal_data['second_kill'] <5]
    print('normal_mid_data', '\n', normal_mid_data)
    #因为是恢复原价进行了销售，所以里面的折扣，折后价都会恢复原价
    print('normal_data_back_second','\n',normal_data_back_second)
    normal_data_back_second['discount_price_rate'] = 1
    normal_data_back_second['price_after_discount'] = normal_data_back_second['sales_price']
    normal_data_back_second['profit_after'] =  normal_data_back_second['profit']
    print('normal_data_back_second', '\n', normal_data_back_second)

    normal_price = normal_mid_data['sales_price'].tolist()
    normal_name.extend(normal_data_back_second['name'].tolist())
    normal_price.extend(normal_data_back_second['sales_price'].tolist())
    normal_price_after_discount.extend(normal_data_back_second['sales_price'].tolist())
    normal_profit_after.extend(normal_data_back_second['profit'].tolist())
    normal_discont_rate = normal_mid_data['discount_price_rate'].tolist()
    normal_discont_rate.extend(normal_data_back_second['discount_price_rate'].tolist())
    max_price = data['threshold'].iloc[0]
    discount = data['discount'].iloc[0]
    seckill_total = np.sum(seckill_price_after_discount)
    x = data['customer'].iloc[0]
    print('normal_discont_rate', '\n', normal_discont_rate)
    return seckill_data,seckill_name,seckill_price_after_discount,seckill_profit_after,seckill_price,second_name,\
           second_price_after_discount,second_profit_after,second_price,normal_name,normal_price_after_discount,\
           normal_profit_after,normal_price,max_price,discount,seckill_total,x,seckill_discont_rate,second_discont_rate,\
           normal_discont_rate


# 这是先算毛利最低的
def main_function(seckill_data, seckill_name, seckill_price_after_discount, seckill_profit_after, seckill_price,
                   second_name,
                   second_price_after_discount, second_profit_after, second_price, normal_name,
                   normal_price_after_discount,
                   normal_profit_after, normal_price, max_price, discount, seckill_total, seckill_discont_rate,
                   second_discont_rate, normal_discont_rate):
    #防止数据量过大导致无法进行计算
    if len(second_name)>15:
        print('特价商品超过15个')
        second_name_01 =second_name[:15]
        second_price_after_discount_01 = second_price_after_discount[:15]
        second_profit_after_01 = second_profit_after[:15]
        second_price_01 = second_price[:15]
        second_discont_rate_01 = second_discont_rate[:15]

        result_01 = core_algorithm(seckill_data, seckill_name, seckill_price_after_discount, seckill_profit_after, seckill_price,
                   second_name_01,
                   second_price_after_discount_01, second_profit_after_01, second_price_01, normal_name,
                   normal_price_after_discount,
                   normal_profit_after, normal_price, max_price, discount, seckill_total, seckill_discont_rate,
                   second_discont_rate_01, normal_discont_rate)

        second_name_02 =second_name[15:30]
        second_price_after_discount_02 = second_price_after_discount[15:30]
        second_profit_after_02 = second_profit_after[15:30]
        second_price_02 = second_price[15:30]
        second_discont_rate_02 = second_discont_rate[15:30]
        result_02 = core_algorithm(seckill_data, seckill_name, seckill_price_after_discount, seckill_profit_after, seckill_price,
                   second_name_02,
                   second_price_after_discount_02, second_profit_after_02, second_price_02, normal_name,
                   normal_price_after_discount,
                   normal_profit_after, normal_price, max_price, discount, seckill_total, seckill_discont_rate,
                   second_discont_rate_02, normal_discont_rate)

        second_name_03 =second_name[30:]
        second_price_after_discount_03 = second_price_after_discount[30:]
        second_profit_after_03 = second_profit_after[30:]
        second_price_03 = second_price[30:]
        second_discont_rate_03 = second_discont_rate[30:]
        result_03 = core_algorithm(seckill_data, seckill_name, seckill_price_after_discount, seckill_profit_after, seckill_price,
                   second_name_03,
                   second_price_after_discount_03, second_profit_after_03, second_price_03, normal_name,
                   normal_price_after_discount,
                   normal_profit_after, normal_price, max_price, discount, seckill_total, seckill_discont_rate,
                   second_discont_rate_03, normal_discont_rate)

        #
        # second_name_04 =second_name[45:60]
        # second_price_after_discount_04 = second_price_after_discount[45:60]
        # second_profit_after_04 = second_profit_after[45:60]
        # second_price_04 = second_price[45:60]
        # second_discont_rate_04 = second_discont_rate[45:60]
        # result_04 = core_algorithm(seckill_data, seckill_name, seckill_price_after_discount, seckill_profit_after, seckill_price,
        #            second_name_04,
        #            second_price_after_discount_04, second_profit_after_04, second_price_04, normal_name,
        #            normal_price_after_discount,
        #            normal_profit_after, normal_price, max_price, discount, seckill_total, seckill_discont_rate,
        #            second_discont_rate_04, normal_discont_rate)
        # second_name_05 =second_name[60:]
        # second_price_after_discount_05 = second_price_after_discount[60:]
        # second_profit_after_05 = second_profit_after[60:]
        # second_price_05 = second_price[60:]
        # second_discont_rate_05 = second_discont_rate[60:]
        # result_05 = core_algorithm(seckill_data, seckill_name, seckill_price_after_discount, seckill_profit_after, seckill_price,
        #            second_name_05,
        #            second_price_after_discount_05, second_profit_after_05, second_price_05, normal_name,
        #            normal_price_after_discount,
        #            normal_profit_after, normal_price, max_price, discount, seckill_total, seckill_discont_rate,
        #            second_discont_rate_05, normal_discont_rate)


        result = pd.concat([result_01,result_02,result_03])
    else:
        print('计算的物品在15个以内')
        result = core_algorithm(seckill_data, seckill_name, seckill_price_after_discount, seckill_profit_after, seckill_price,
                   second_name,
                   second_price_after_discount, second_profit_after, second_price, normal_name,
                   normal_price_after_discount,
                   normal_profit_after, normal_price, max_price, discount, seckill_total, seckill_discont_rate,
                   second_discont_rate, normal_discont_rate)
    return result

def core_algorithm(seckill_data, seckill_name, seckill_price_after_discount, seckill_profit_after, seckill_price,
                   second_name,
                   second_price_after_discount, second_profit_after, second_price, normal_name,
                   normal_price_after_discount,
                   normal_profit_after, normal_price, max_price, discount, seckill_total, seckill_discont_rate,
                   second_discont_rate, normal_discont_rate):
    all_profit = []
    all_profit_total = []
    all_value = []
    all_permutation = []
    all_sales_price = []
    all_discount_rate = []
    '''all_value = 所有的组合总价
    all_permutation = 所有满足条件组合
    all_profit = []  所有满足条件的毛利组合
    all_profit_total = [] 所有满足条件的毛利总价
    all_sales_price = [] 所有满足条件商品的原价
    '''
    def sum_list(x):
        result = 0
        for i in range(len(x)):
            result += x[i]
        return result
    seckill_profit_after_total = np.sum(seckill_profit_after)

    #将特价商品采用全部的排列的方式,C(10,1),C(10,2)..C(10,10)，然后依次往下进行
    #len(second_name)+1
    for i in range(1,4):
        print('i',i,len(second_name))
        combins_name = [c for c in combinations(second_name,i)]
        combins_price_after_discount = [c for c in combinations(second_price_after_discount, i)]
        combins_profit_after = [c for c in combinations(second_profit_after, i)]
        combins_price = [c for c in combinations(second_price, i)]
        combins_discount_rate = [c for c in combinations(second_discont_rate, i)]
        print(len(combins_name))
        for k in range(0,len(combins_name)):
            combins_name_list = combins_name[k]
            combins_second_kill_total = sum_list(combins_price_after_discount[k])
            combins_second_price = combins_price[k]
            combins_second_price_total = sum_list(combins_second_price)
            combins_second_profit_after = combins_profit_after[k]
            combins_second_discount_rate = combins_discount_rate[k]
            combins_second_profit_after_total = sum_list(combins_second_profit_after)
            #对每一种组合进行最大金额进行更新
            max_price_new = max_price - np.sum(seckill_price_after_discount) - combins_second_kill_total
            # 记录一个包含了前两种商品的name列表
            first_second_name = seckill_name
            list_01 = list(combins_name_list)
            first_second_name = first_second_name + list_01
            #记录一开始秒杀商品的折扣率
            first_discount_rate = seckill_discont_rate
            list_02 = list(combins_second_discount_rate)
            first_discount_rate = first_discount_rate + list_02
            #记录已有商品原价的list
            first_second_price = seckill_price
            list_03 = list(combins_second_price)
            first_second_price = list_03 + first_second_price
            #记录包含已经涵盖的商品折后毛利
            first_second_all_profit = seckill_profit_after
            list_04= list(combins_second_profit_after)
            first_second_all_profit = first_second_all_profit + list_04
            # 记录下前秒杀商品和特价商品总价和总毛利
            two_value = combins_second_kill_total + np.sum(seckill_price_after_discount)
            two_profit_total = combins_second_profit_after_total + seckill_profit_after_total
            #----------------------------------------------------------------
            # '''将每个特价商品的组合也列入组合数中'''
            all_permutation.append(first_second_name)
            # 当前组合下所有sku对应折扣率
            all_discount_rate.append(first_discount_rate)
            # 当前折扣率下所有sku对应的原价
            all_sales_price.append(first_second_price)
            # 当前组合所有折后毛利列表
            all_profit.append(first_second_all_profit)
            # 分别是总计实际折后的金额和折后毛利总和
            all_value.append(two_value)
            all_profit_total.append(two_profit_total)
            for x in range(len(normal_name)):
                print('加入原价商品计算')
                if max_price_new < normal_price_after_discount[x]:
                    #用来记录在当前组合下所有sku的名字
                    list_001 = [normal_name[x]]
                    mid_list_name = first_second_name + list_001
                    all_permutation.append(mid_list_name)
                    #当前组合下所有sku对应折扣率
                    list_discount = [normal_discont_rate[x]]
                    mid_discount_rate = first_discount_rate + list_discount
                    all_discount_rate.append(mid_discount_rate)
                    #当前折扣率下所有sku对应的原价
                    list_price = [normal_price[x]]
                    mid_list_price = first_second_price + list_price
                    all_sales_price.append(mid_list_price)
                    #当前组合所有折后毛利列表
                    list_002 = [normal_profit_after[x]]
                    mid_list_all_profit = first_second_all_profit + list_002
                    all_profit.append(mid_list_all_profit)
                    #分别是总计实际折后的金额和折后毛利总和
                    new_value = two_value + normal_price_after_discount[x]
                    # print('normal_price_after_discount[x]','\n',normal_price_after_discount[x])
                    # print('all_value', '\n', all_value)
                    all_value.append(new_value)
                    # print('new_value', '\n', new_value)
                    # print('all_value', '\n', all_value)
                    all_profit_total.append(two_profit_total + normal_profit_after[x])
                    # print('list_001','\n',list_001)
                    # print('mid_list_name', '\n', mid_list_name)
                    # print('first_discount_rate', '\n', first_discount_rate)
                    # print('list_discount', '\n', list_discount)
                    # print('list_price', '\n', list_price)
                    # print('mid_list_price','\n',mid_list_price)
                    # print('mid_list_name', '\n', mid_list_name)
                    # print('list_002', '\n', list_002)
                    # print('mid_list_all_profit', '\n', mid_list_all_profit)
                else:
                    two_max_price = max_price_new - normal_price_after_discount[x]
                    for y in range(len(normal_name)):
                        if two_max_price < normal_price_after_discount[y]:
                            list_001 = [normal_name[x], normal_name[y]]
                            mid_list_name = first_second_name + list_001
                            all_permutation.append(mid_list_name)
                            list_discount = [normal_discont_rate[x],normal_discont_rate[y]]
                            mid_discount_rate = first_discount_rate + list_discount
                            all_discount_rate.append(mid_discount_rate)
                            list_price = [normal_price[x], normal_price[y]]
                            mid_list_price = first_second_price + list_price
                            all_sales_price.append(mid_list_price)
                            list_002 = [normal_profit_after[x],normal_profit_after[y]]
                            mid_list_all_profit = first_second_all_profit + list_002
                            all_profit.append(mid_list_all_profit)
                            all_value.append(
                                two_value + normal_price_after_discount[x] + normal_price_after_discount[y])
                            all_profit_total.append(two_profit_total + normal_profit_after[x] + normal_profit_after[y])
                        else:
                            three_max_price = max_price - normal_price_after_discount[x] - normal_price_after_discount[
                                y]

                            for z in range(len(normal_name)):
                                if three_max_price < normal_price_after_discount[z]:
                                    list_001 = [normal_name[x], normal_name[y], normal_name[z]]
                                    mid_list_name = first_second_name + list_001
                                    all_permutation.append(mid_list_name)

                                    list_price = [normal_price[x], normal_price[y], normal_price[z]]
                                    mid_list_price = first_second_price + list_price
                                    all_sales_price.append(mid_list_price)

                                    list_discount = [normal_discont_rate[x], normal_discont_rate[y],normal_discont_rate[z]]
                                    mid_discount_rate = first_discount_rate + list_discount
                                    all_discount_rate.append(mid_discount_rate)

                                    list_002 = [normal_profit_after[x], normal_profit_after[y],
                                               normal_profit_after[z]]
                                    mid_list_all_profit = first_second_all_profit + list_002
                                    all_profit.append(mid_list_all_profit)

                                    all_value.append(
                                        two_value + normal_price_after_discount[x] +
                                        normal_price_after_discount[y] + normal_price_after_discount[z])
                                    all_profit_total.append(
                                        two_profit_total + normal_profit_after[x] + normal_profit_after[y] +
                                        normal_profit_after[z])
    data = pd.DataFrame({'goods':all_permutation,'value':all_value,'all_profit':all_profit,
                         'all_profit_total':all_profit_total,'all_sales_price':all_sales_price,
                         'discount_rate': all_discount_rate})
    # data.to_csv('D:/AI/xianfengsg/other_project/product_profit/data.csv', encoding='utf_8_sig')
    return data

#再定义一个函数用来计算完全原价商品购买商品的组合,这里data已经区分了两个不同平台
def original_alogorithm(data,max_price):
    all_profit = []
    all_profit_total = []
    all_value = []
    all_permutation = []
    all_sales_price = []
    all_discount_rate = []

    original_price_data = data[data['discount_price_rate'] == 1]
    original_name = original_price_data['name'].tolist()
    original_price_after_discount = original_price_data['price_after_discount'].tolist()
    original_profit_after = original_price_data['profit_after'].tolist()
    original_sales_price = original_price_data['sales_price'].tolist()
    original_discount_price_rate = original_price_data['discount_price_rate'].tolist()

    def sum_list(x):
        result = 0
        for i in range(len(x)):
            result += x[i]
        return result
    # 将特价商品采用全部的排列的方式,C(10,1),C(10,2)..C(10,10)，然后依次往下进行
    if len(original_name) > 6:
        x = 6
    else:
        x = len(original_name)
    for i in range(1, x):
        print('i',i)
        combins_name = [c for c in combinations(original_name, i)]
        combins_price_after_discount = [c for c in combinations(original_price_after_discount, i)]
        combins_profit_after = [c for c in combinations(original_profit_after, i)]
        combins_price = [c for c in combinations(original_sales_price, i)]
        combins_discount_rate = [c for c in combinations(original_discount_price_rate, i)]
        print(len(combins_name))
        for k in range(0, len(combins_name)):
            combins_name_list = combins_name[k]
            combins_second_kill_total = sum_list(combins_price_after_discount[k])
            combins_second_price = combins_price[k]
            combins_second_price_total = sum_list(combins_second_price)
            combins_second_profit_after = combins_profit_after[k]
            combins_second_discount_rate = combins_discount_rate[k]
            combins_second_profit_after_total = sum_list(combins_second_profit_after)
            # 对每一种组合进行最大金额进行更新
            max_price_new = max_price  - combins_second_kill_total
            # 记录一个包含了前两种商品的name列表

            first_second_name = list(combins_name_list)

            # 记录一开始秒杀商品的折扣率

            first_discount_rate = list(combins_second_discount_rate)
            # 记录已有商品原价的list
            first_second_price = list(combins_second_price)
            # 记录包含已经涵盖的商品折后毛利
            first_second_all_profit = list(combins_second_profit_after)

            # 记录下前秒杀商品和特价商品总价和总毛利
            two_value = combins_second_kill_total
            two_profit_total = combins_second_profit_after_total
            # ----------------------------------------------------------------
            '''将没个特价商品的组合也列入组合数中'''
            all_permutation.append(first_second_name)
            # 当前组合下所有sku对应折扣率
            all_discount_rate.append(first_discount_rate)
            # 当前折扣率下所有sku对应的原价
            all_sales_price.append(first_second_price)
            # 当前组合所有折后毛利列表
            all_profit.append(first_second_all_profit)
            # 分别是总计实际折后的金额和折后毛利总和
            all_value.append(two_value)
            all_profit_total.append(two_profit_total)

    original_data = pd.DataFrame({'goods':all_permutation,'value':all_value,'all_profit':all_profit,
                         'all_profit_total':all_profit_total,'all_sales_price':all_sales_price,
                         'discount_rate':all_discount_rate})
    # original_data.to_csv('D:/AI/xianfengsg/other_project/product_profit/original_data.csv', encoding='utf_8_sig')
    return original_data


#再定义一个函数用来

def get_resulted_data(original_data,data,max_price):
    mid_data = pd.concat([original_data,data])
    # mid_data = mid_data[mid_data['value'] >= max_price]
    #与产品商量范围小于100
    mid_data = mid_data[mid_data['value'] <= 100]
    mid_data = mid_data.reset_index(drop=True)
    mid_data['all_profit_total'].map(lambda x:('%.2f')%x)
    # mid_data['profit'].map(lambda x:('%.2f')%x)
    # mid_data['all_sales_price'].map(lambda x:('%.2f')%x)
    mid_data.drop_duplicates('all_profit_total','first',inplace=True)
    mid_data.drop_duplicates('value','first',inplace=True)
    return mid_data

#这里需要再增加以下商铺满减的规则，如果是1的话就是满59-10，如果2的话就是直接折扣
def rule_discount(data):
    store_discount = data['store_discount'].iloc[0]
    store_threshold = data['store_threshold'].iloc[0]
    discont_rate = data['discont_rate'].iloc[0]
    return store_discount,store_threshold,discont_rate

#只算满减
def max_profit_subtract(mid_data,discount,max_price,store_threshold,store_discount,Commission,coefficient):
    Permutation = mid_data
    print('进行满减等相关计算')
    def int_convert(x):
        y =[]
        x['all_sales_price'] = [int(i) for i in  x['all_sales_price']]
        # x['value'] = [int(i) for i in x['value']]
        # 需要查看最后的结果内是否有原价商品满足满减条件,满足条件discount 直接乘以1，不满足条件是0
        origianl_total_sales = 0
        for k in range(len(x['all_sales_price'])):
            if x['discount_rate'][k] == 1:
                origianl_total_sales += x['all_sales_price'][k]
        if origianl_total_sales > max_price:
            coefficient_01 = 1
        else:
            coefficient_01 =0
        #这里还需要一个参数来控制两个平台的满减规则
        if 1 in x['discount_rate'] and len(set(x['discount_rate'])) == 1:
            coefficient_02 = 1
        else:
            coefficient_02 = -1
        coefficient_final = coefficient_02*coefficient*coefficient_01
        #通过sigmoid函数确定到底该不该考虑满减
        def sigmoid(x):
            s = 1 / (1 + np.exp(-x))
            if s >0.5 :
                return 1
            else:
                return 0
        result_coefficient = sigmoid(coefficient_final)
        fianl_discount = discount*result_coefficient
        for i in range(len(x['all_sales_price'])):
            if x['value'] - fianl_discount > int(store_threshold):
                x['商家满减'] = 'Y'
                numerator = (x['all_profit'][i]/x['all_profit_total']) * (x['all_profit_total']-fianl_discount-store_discount)
                denominator = x['all_sales_price'][i]
                result=numerator/denominator
                result = "%.4f%%" % (result * 100)
                # y.append(((x['all_profit'][i]/x['all_profit_total']) * (x['all_profit_total']-discount))/x['all_sales_price'] )
                y.append(result)
            else:
                x['商家满减'] = 'N'
                numerator = (x['all_profit'][i]/x['all_profit_total']) * (x['all_profit_total']-fianl_discount)
                denominator = x['all_sales_price'][i]
                result=numerator / denominator
                result = "%.4f%%" % (result * 100)
                # y.append(((x['all_profit'][i]/x['all_profit_total']) * (x['all_profit_total']-discount))/x['all_sales_price'] )
                y.append(result)
            # y = x['all_sales_price']+x['profit']
        return y

    def judge_size(df):
        # 需要查看最后的结果内是否有原价商品满足满减条件,满足条件discount 直接乘以1，不满足条件是0
        origianl_total_sales = 0
        for k in range(len(df['all_sales_price'])):
            if df['discount_rate'][k] == 1:
                origianl_total_sales += df['all_sales_price'][k]
        if origianl_total_sales > max_price:
            coefficient_01 = 1
        else:
            coefficient_01 =0
        #这里还需要一个参数来控制两个平台的满减规则
        if 1 in df['discount_rate'] and len(set(df['discount_rate']))==1:
            coefficient_02 = 1
        else:
            coefficient_02 = -1
        coefficient_final = coefficient_02*coefficient*coefficient_01
        #通过sigmoid函数确定到底该不该考虑满减
        def sigmoid(x):
            s = 1 / (1 + np.exp(-x))
            if s >0.5 :
                return 1
            else:
                return 0
        result_coefficient = sigmoid(coefficient_final)
        fianl_discount = discount*result_coefficient

        if df['value'] - fianl_discount > store_threshold:
            df['value']  = (df['value'] - fianl_discount - store_discount)*Commission
            return df['value']
        else:
            df['value'] = (df['value'] - fianl_discount ) * Commission
            return df['value']

    def judge_size_profit(df):
        # 需要查看最后的结果内是否有原价商品满足满减条件,满足条件discount 直接乘以1，不满足条件是0
        origianl_total_sales = 0
        for k in range(len(df['all_sales_price'])):
            if df['discount_rate'][k] == 1:
                origianl_total_sales += df['all_sales_price'][k]
        if origianl_total_sales > max_price:
            coefficient_01 = 1
        else:
            coefficient_01 = 0
        # 这里还需要一个参数来控制两个平台的满减规则
        if 1 in df['discount_rate'] and len(set(df['discount_rate'])) == 1:
            coefficient_02 = 1
        else:
            coefficient_02 = -1
        coefficient_final = coefficient_02 * coefficient * coefficient_01

        # 通过sigmoid函数确定到底该不该考虑满减
        def sigmoid(x):
            s = 1 / (1 + np.exp(-x))
            if s > 0.5:
                return 1
            else:
                return 0

        result_coefficient = sigmoid(coefficient_final)
        fianl_discount = discount * result_coefficient

        if df['value'] - fianl_discount > store_threshold:
            #佣金
            commission_fee = (df['value'] - fianl_discount - store_discount) * (1-Commission)
            df['all_profit_total'] = df['all_profit_total'] - fianl_discount - store_discount-commission_fee
            return df['all_profit_total']
        else:
            # 佣金
            commission_fee = (df['value'] - fianl_discount - store_discount) * (1-Commission)
            df['all_profit_total'] = df['all_profit_total'] - fianl_discount - commission_fee
            return df['all_profit_total']
    Permutation['all_profit_total']  = Permutation.apply\
        (lambda x: judge_size_profit(x),axis=1)
    Permutation['profit_rate'] = Permutation.apply \
            (lambda x: int_convert(x), axis=1)
    Permutation['value']  = Permutation.apply\
        (lambda x: judge_size(x),axis=1)

    max_Permutation = Permutation.sort_values(by=['all_profit_total'], ascending=False)
    max_Permutation = max_Permutation.head(10)
    max_Permutation['max_rank'] =  range(len(max_Permutation))
    max_Permutation['max_rank'] = max_Permutation['max_rank'].apply(lambda x: x+1)

    min_Permutation = Permutation.sort_values(by=['all_profit_total'], ascending=True)
    min_Permutation = min_Permutation.head(10)
    min_Permutation['max_rank'] =  range(len(min_Permutation))
    min_Permutation['max_rank'] = min_Permutation['max_rank'].apply(lambda x: x+1)
    return max_Permutation,min_Permutation,Permutation

#只算满减
def min_profit_subtract(mid_data,discount,max_price,store_threshold,store_discount,Commission,coefficient):
    min_Permutation= mid_data
    # min_Permutation.to_csv('D:/AI/xianfengsg/other_project/product_profit/min_Permutation_old.csv', encoding='utf_8_sig')
    #计算最后每个商品对应的毛利率
    def int_convert(x):
        y = []
        # x['all_sales_price'] = [int(i) for i in x['all_sales_price']]

        origianl_total_sales = 0
        for k in range(len(x['all_sales_price'])):
            if x['discount_rate'] == 1:
                origianl_total_sales += x['all_sales_price'][k]
        if origianl_total_sales > max_price:
            coefficient_01 = 1
        else:
            coefficient_01 =0
        #这里还需要一个参数来控制两个平台的满减规则
        if 1 in x['discount_rate'] and len(set(x['discount_rate']))==1:
            coefficient_02 = 1
        else:
            coefficient_02 = -1
        coefficient_final = coefficient_02*coefficient*coefficient_01
        #通过sigmoid函数确定到底该不该考虑满减
        def sigmoid(x):
            s = 1 / (1 + np.exp(-x))
            if s >0.5 :
                return 1
            else:
                return 0
        result_coefficient = sigmoid(coefficient_final)
        fianl_discount = discount*result_coefficient
        for i in range(len(x['all_sales_price'])):
            if x['value'] - fianl_discount > store_threshold:
                numerator = (x['all_profit'][i]/x['all_profit_total']) * (x['all_profit_total']-fianl_discount-store_discount)
                denominator = x['all_sales_price'][i]
                result=numerator/denominator
                result = "%.4f%%" % (result * 100)
                # y.append(((x['all_profit'][i]/x['all_profit_total']) * (x['all_profit_total']-discount))/x['all_sales_price'] )
                y.append(result)
            else:
                numerator = (x['all_profit'][i]/x['all_profit_total']) * (x['all_profit_total']-fianl_discount)
                denominator = x['all_sales_price'][i]
                result=numerator/denominator
                result = "%.4f%%" % (result * 100)
                # y.append(((x['all_profit'][i]/x['all_profit_total']) * (x['all_profit_total']-discount))/x['all_sales_price'] )
                y.append(result)
            # y = x['all_sales_price']+x['profit']
        return y
    #计算最后顾客实际的付款金额
    def judge_size(df):
        origianl_total_sales = 0
        for k in range(len(df['all_sales_price'])):
            if df['discount_rate'] == 1:
                origianl_total_sales += df['all_sales_price'][k]
        if origianl_total_sales > max_price:
            coefficient_01 = 1
        else:
            coefficient_01 =0
        #这里还需要一个参数来控制两个平台的满减规则
        if 1 in df['discount_rate'] and len(set(df['discount_rate']))==1:
            coefficient_02 = 1
        else:
            coefficient_02 = -1
        coefficient_final = coefficient_02*coefficient*coefficient_01
        #通过sigmoid函数确定到底该不该考虑满减
        def sigmoid(x):
            s = 1 / (1 + np.exp(-x))
            if s >0.5 :
                return 1
            else:
                return 0
        result_coefficient = sigmoid(coefficient_final)
        fianl_discount = discount*result_coefficient
        if df['value'] - fianl_discount > store_threshold:
            df['value']  = (df['value'] - fianl_discount - store_discount)*Commission
            return df['value']
        else:
            df['value'] = (df['value'] - fianl_discount )*Commission
            return df['value']
    #计算最后该组合下所有的毛利总和
    def judge_size_profit(df):
        # 需要查看最后的结果内是否有原价商品满足满减条件,满足条件discount 直接乘以1，不满足条件是0
        origianl_total_sales = 0
        for k in range(len(df['all_sales_price'])):
            if df['discount_rate'] == 1:
                origianl_total_sales += df['all_sales_price'][k]
        if origianl_total_sales > max_price:
            coefficient_01 = 1
        else:
            coefficient_01 = 0
        # 这里还需要一个参数来控制两个平台的满减规则
        if 1 in df['discount_rate'] and len(set(df['discount_rate'])) == 1:
            coefficient_02 = 1
        else:
            coefficient_02 = -1
        coefficient_final = coefficient_02 * coefficient * coefficient_01

        # 通过sigmoid函数确定到底该不该考虑满减
        def sigmoid(x):
            s = 1 / (1 + np.exp(-x))
            if s > 0.5:
                return 1
            else:
                return 0

        result_coefficient = sigmoid(coefficient_final)
        fianl_discount = discount * result_coefficient

        if df['value'] - fianl_discount > store_threshold:
            #佣金
            commission_fee = (df['value'] - fianl_discount - store_discount) * (1-Commission)
            reslut = df['all_profit_total'] - fianl_discount - store_discount-commission_fee
            # print('value', df['value'], 'fianl_discount', fianl_discount, 'store_discount', store_discount,
            #       'commission_fee', commission_fee,'all_profit_total',df['all_profit_total'],'reslut',reslut)

            return reslut
        else:
            # 佣金
            commission_fee = (df['value'] - fianl_discount - store_discount) * (1-Commission)
            reslut = df['all_profit_total'] - fianl_discount- commission_fee
            return reslut

    min_Permutation['all_profit_total']  = min_Permutation.apply\
        (lambda x: judge_size_profit(x),axis=1)
    min_Permutation['profit_rate'] = min_Permutation.apply \
        (lambda x: int_convert(x), axis=1)
    min_Permutation['value']  = min_Permutation.apply\
        (lambda x: judge_size(x),axis=1)
    min_Permutation = min_Permutation.sort_values(by=['all_profit_total'], ascending=True)
    min_Permutation = min_Permutation.head(10)
    min_Permutation['max_rank'] = range(len(min_Permutation))
    min_Permutation['max_rank'] = min_Permutation['max_rank'].apply(lambda x: x + 1)
    return min_Permutation

#只算折扣
def max_profit_discount_rate(mid_data,discount,max_price,discont_rate,Commission,coefficient):
    Permutation = mid_data
    print('正在进行折扣计算')
    def int_convert(x):
        y =[]
        x['all_sales_price'] = [int(i) for i in  x['all_sales_price']]
        origianl_total_sales = 0
        for k in range(len(x['all_sales_price'])):
            if x['discount_rate'][k] == 1:
                origianl_total_sales += x['all_sales_price'][k]
        if origianl_total_sales > max_price:
            coefficient_01 = 1
        else:
            coefficient_01 = 0
        #这里还需要一个参数来控制两个平台的满减规则
        if 1 in x['discount_rate'] and len(set(x['discount_rate'])) == 1:
            coefficient_02 = 1
        else:
            coefficient_02 = -1
        coefficient_final = coefficient_02 * coefficient * coefficient_01
        #通过sigmoid函数确定到底该不该考虑满减
        def sigmoid(x):
            s = 1 / (1 + np.exp(-x))
            if s > 0.5 :
                return 1
            else:
                return 0
        result_coefficient = sigmoid(coefficient_final)
        fianl_discount = discount*result_coefficient
        result_value = (x['value'] - (x['value'] - fianl_discount) * discont_rate * Commission)

        for i in range(len(x['all_sales_price'])):
            numerator = (x['all_profit'][i]/x['all_profit_total']) * (x['all_profit_total']-result_value)
            denominator = x['all_sales_price'][i]
            result=numerator/denominator
            result = "%.2f%%" % (result * 100)
            # y.append(((x['all_profit'][i]/x['all_profit_total']) * (x['all_profit_total']-discount))/x['all_sales_price'] )
            y.append(result)
        return y

    #计算总毛利价格
    def judge_all_profit(x):
            #佣金
        commission_fee = x['value']*(1-(discont_rate*Commission))
        x['all_profit_total'] = x['all_profit_total'] -commission_fee
        return x['all_profit_total']

    #计算最后的付款金额
    def judge_size(df):
        # 需要查看最后的结果内是否有原价商品满足满减条件,满足条件discount 直接乘以1，不满足条件是0
        origianl_total_sales = 0
        for k in range(len(df['all_sales_price'])):
            if df['discount_rate'][k] == 1:
                origianl_total_sales += df['all_sales_price'][k]
        if origianl_total_sales > max_price:
            coefficient_01 = 1
        else:
            coefficient_01 =0
        #这里还需要一个参数来控制两个平台的满减规则
        if 1 in df['discount_rate'] and len(set(df['discount_rate']))==1:
            coefficient_02 = 1
        else:
            coefficient_02 = -1
        coefficient_final = coefficient_02*coefficient*coefficient_01
        #通过sigmoid函数确定到底该不该考虑满减
        def sigmoid(x):
            s = 1 / (1 + np.exp(-x))
            if s >0.5 :
                return 1
            else:
                return 0
        result_coefficient = sigmoid(coefficient_final)
        fianl_discount = discount*result_coefficient
        result_value = (df['value'] - fianl_discount) * discont_rate * Commission
        return result_value

    Permutation['profit_rate'] = Permutation.apply\
        (lambda x: int_convert(x), axis=1)
    print('毛利率数据整理完成')
    Permutation['all_profit_total'] = Permutation.apply\
        (lambda x: judge_all_profit(x), axis=1)
    print('总毛利数据整理完成')
    Permutation['value'] = Permutation.apply\
        (lambda x: judge_size(x), axis=1)
    print('总价值数据整理完成')
    # Permutation.to_csv('D:/AI/xianfengsg/other_project/product_profit/Permutation.csv', encoding='utf_8_sig')
    max_Permutation = Permutation.sort_values(by=['all_profit_total'],ascending=False)
    print('倒序排序完成')
    max_Permutation = max_Permutation.head(10)
    max_Permutation['max_rank'] =  range(len(max_Permutation))
    max_Permutation['max_rank'] = max_Permutation['max_rank'].apply(lambda x: x+1)
    print('倒序排列名次完成')

    min_Permutation = Permutation.sort_values(by=['all_profit_total'],ascending=True)
    print('正序排序完成')
    min_Permutation = min_Permutation.head(10)
    min_Permutation['max_rank'] =  range(len(min_Permutation))
    min_Permutation['max_rank'] = min_Permutation['max_rank'].apply(lambda x: x+1)
    print('正序排列名次完成')
    return max_Permutation,min_Permutation,Permutation

#只算折扣
def min_profit_discount_rate(mid_data,discount,max_price,discont_rate,Commission,coefficient):
    min_Permutation = mid_data
    def int_convert(x):
        y = []
        x['all_sales_price'] = [int(i) for i in x['all_sales_price']]
        origianl_total_sales = 0
        for k in range(len(x['all_sales_price'])):
            if x['discount_rate'] == 1:
                origianl_total_sales += x['all_sales_price'][k]
        if origianl_total_sales > max_price:
            coefficient_01 = 1
        else:
            coefficient_01 =0
        #这里还需要一个参数来控制两个平台的满减规则
        if 1 in x['discount_rate'] and len(set(x['discount_rate']))==1:
            coefficient_02 = 1
        else:
            coefficient_02 = -1
        coefficient_final = coefficient_02*coefficient*coefficient_01
        #通过sigmoid函数确定到底该不该考虑满减
        def sigmoid(x):
            s = 1 / (1 + np.exp(-x))
            if s >0.5 :
                return 1
            else:
                return 0
        result_coefficient = sigmoid(coefficient_final)
        fianl_discount = discount*result_coefficient
        for i in range(len(x['all_sales_price'])):
            numerator = (x['all_profit'][i]/x['all_profit_total']) * ((x['all_profit_total']-fianl_discount)*discont_rate)
            denominator = x['all_sales_price'][i]
            result=numerator/denominator
            result = "%.2f%%" % (result * 100)
            # y.append(((x['all_profit'][i]/x['all_profit_total']) * (x['all_profit_total']-discount))/x['all_sales_price'] )
            y.append(result)
        return y

    # 计算总毛利价格
    def judge_all_profit(x):
        # 佣金
        commission_fee = x['value'] * (1- (discont_rate * Commission))
        discount_fee = x['value'] * (1 - discont_rate)
        x['all_profit_total'] = x['all_profit_total'] - discount_fee - commission_fee
        return x['all_profit_total']

    min_Permutation['all_profit_total'] = min_Permutation.apply \
        (lambda x: judge_all_profit(x), axis=1)
    min_Permutation['value'] = min_Permutation['value'].apply\
        (lambda x: x * discont_rate * Commission)
    min_Permutation['profit_rate'] = min_Permutation.apply \
        (lambda x: int_convert(x), axis=1)
    min_Permutation = min_Permutation.sort_values(by=['all_profit_total'], ascending=True)
    min_Permutation = min_Permutation.head(10)
    min_Permutation['max_rank'] = range(len(min_Permutation))
    min_Permutation['max_rank'] = min_Permutation['max_rank'].apply(lambda x: x + 1)
    return min_Permutation

#最后将所有的英文标题转成中文字符
def convert_chinese(data):
    mid_data = data
    final_chinese_data = pd.DataFrame()
    mid_data = mid_data[mid_data['goods'].map(len) >= 2]
    num = len(mid_data)
    print(num)
    for i in tqdm.tqdm(range(num)):
        goods_list = mid_data['goods'].iloc[i]
        all_profit_list = mid_data['all_profit'].iloc[i]
        all_sales_price_list = mid_data['all_sales_price'].iloc[i]
        discount_rate_list = mid_data['discount_rate'].iloc[i]
        profit_rate_list = mid_data['profit_rate'].iloc[i]
        for k in range(len(goods_list)):
            goods_list_temp = goods_list[k]
            all_profit_list_temp = all_profit_list[k]
            all_sales_price_list_temp = all_sales_price_list[k]
            discount_rate_list_temp = discount_rate_list[k]
            profit_rate_list_temp = profit_rate_list[k]
            final_chinese_data = final_chinese_data.append({'当前组合商品名称': goods_list_temp,
                                                            '每款商品的折扣毛利': all_profit_list_temp,
                                                            '每款商品的原始售价': all_sales_price_list_temp,
                                                            '每款商品的折扣率': discount_rate_list_temp,
                                                            '每款商品对应的毛利率': profit_rate_list_temp,
                                                            '该组合下客户最终的订单金额':mid_data['value'].iloc[i],
                                                            '总毛利':mid_data['all_profit_total'].iloc[i],
                                                            '毛利排名':i+1}, ignore_index=True)

    # final_chinese_data = mid_data.rename(index=str, columns={
    #     'goods': '当前组合商品名称',
    #     'value': '该组合下客户最终的订单金额',
    #     'all_profit': '每款商品的折扣毛利',
    #     'all_profit_total': '总毛利',
    #     'all_sales_price': '每款商品的原始售价',
    #     'discount_rate': '每款商品的折扣率',
    #     'profit_rate': '每款商品对应的毛利率'
    # })
    return final_chinese_data


#饿了么的规则是只要不买折扣品的
def elema_main(data,path_max,path_min):
    store_discount, store_threshold, discont_rate = rule_discount(data)
    seckill_data, seckill_name, seckill_price_after_discount, seckill_profit_after, seckill_price, second_name, \
    second_price_after_discount, second_profit_after, second_price, normal_name, normal_price_after_discount, \
    normal_profit_after, normal_price, max_price, discount, seckill_total, x,seckill_discont_rate,\
    second_discont_rate,normal_discont_rate\
        = process_data(data)
    Commission = 0.9
    print('正在测算饿了么')
    print(store_discount, store_threshold, discont_rate)
    pro_data = main_function(seckill_data, seckill_name, seckill_price_after_discount,
                              seckill_profit_after, seckill_price, second_name,
                              second_price_after_discount, second_profit_after, second_price, normal_name,
                              normal_price_after_discount,
                              normal_profit_after, normal_price, max_price, discount, seckill_total,
                              seckill_discont_rate,
                              second_discont_rate,normal_discont_rate)
    print('主计算完成')
    # pro_data.to_csv(path_max + 'pro_data.csv', encoding='utf_8_sig')
    original_data = original_alogorithm(data, max_price)
    print('全价商品组合计算完成')
    # original_data.to_csv(path_max + 'original_data.csv', encoding='utf_8_sig')
    mid_data = get_resulted_data(original_data,pro_data, max_price)
    print('数据合并完成')
    # mid_data.to_csv(path_max + 'test.csv', encoding='utf_8_sig')
    if x == 1:
        max_Permutation,min_Permutation,Permutation = max_profit_subtract\
            (mid_data, discount, max_price,store_threshold, store_discount,Commission,1)
        # min_Permutation = min_profit_subtract(mid_data, discount, max_price,store_threshold, store_discount,Commission,-1)
        max_Permutation = convert_chinese(max_Permutation)
        min_Permutation = convert_chinese(min_Permutation)
        max_Permutation.to_csv(path_max + 'max_Permutation_elema.csv', encoding='utf_8_sig')
        min_Permutation.to_csv(path_min + 'min_Permutation_elema.csv', encoding='utf_8_sig')
        Permutation = convert_chinese(Permutation)
        Permutation.to_csv(path_min + 'Permutation_elema.csv', encoding='utf_8_sig')
    else:
        max_Permutation,min_Permutation,Permutation = max_profit_discount_rate\
            (mid_data, discount, max_price,discont_rate,Commission,1)
        # min_Permutation = min_profit_discount_rate(mid_data, discount, max_price,discont_rate,Commission,-1)
        max_Permutation = convert_chinese(max_Permutation)
        min_Permutation = convert_chinese(min_Permutation)
        max_Permutation.to_csv(path_max + 'max_Permutation_elema.csv', encoding='utf_8_sig')
        min_Permutation.to_csv(path_min + 'min_Permutation_elema.csv', encoding='utf_8_sig')
        Permutation = convert_chinese(Permutation)
        Permutation.to_csv(path_min + 'Permutation_elema.csv', encoding='utf_8_sig')

#单独计算美团的订单组合数
def meituan_main(data,path_max,path_min):
    store_discount, store_threshold, discont_rate = rule_discount(data)
    seckill_data, seckill_name, seckill_price_after_discount, seckill_profit_after, seckill_price, second_name, \
    second_price_after_discount, second_profit_after, second_price, normal_name, normal_price_after_discount, \
    normal_profit_after, normal_price, max_price, discount, seckill_total, x,seckill_discont_rate,\
    second_discont_rate,normal_discont_rate\
        = process_data(data)

    Commission = 0.88
    print('正在测算meituan_main')
    pro_data = main_function(seckill_data, seckill_name, seckill_price_after_discount,
                              seckill_profit_after, seckill_price, second_name,
                              second_price_after_discount, second_profit_after, second_price, normal_name,
                              normal_price_after_discount,
                              normal_profit_after, normal_price, max_price, discount, seckill_total,
                              seckill_discont_rate,
                              second_discont_rate,normal_discont_rate)

    original_data = original_alogorithm(data,max_price)
    mid_data = get_resulted_data(original_data,pro_data, max_price)
    #这是为了查看是含有
    if x == 1:
        max_Permutation,min_Permutation,Permutation = \
            max_profit_subtract(mid_data, discount,max_price,store_threshold, store_discount,Commission,1)
        # min_Permutation = min_profit_subtract(mid_data, discount,max_price, store_threshold, store_discount,Commission,1)
        max_Permutation = convert_chinese(max_Permutation)
        min_Permutation = convert_chinese(min_Permutation)
        max_Permutation.to_csv(path_max + 'max_Permutation_meituan.csv', encoding='utf_8_sig')
        min_Permutation.to_csv(path_min + 'min_Permutation_meituan.csv', encoding='utf_8_sig')
        Permutation = convert_chinese(Permutation)
        Permutation.to_csv(path_min + 'Permutation_meituan.csv', encoding='utf_8_sig')
    else:
        max_Permutation,min_Permutation,Permutation = \
            max_profit_discount_rate(mid_data, discount,max_price, discont_rate,Commission,1)
        # min_Permutation = min_profit_discount_rate(mid_data, discount,max_price, discont_rate,Commission,1)
        max_Permutation = convert_chinese(max_Permutation)
        min_Permutation = convert_chinese(min_Permutation)
        max_Permutation.to_csv(path_max + 'max_Permutation_meituan.csv', encoding='utf_8_sig')
        min_Permutation.to_csv(path_min + 'min_Permutation_meituan.csv', encoding='utf_8_sig')
        Permutation = convert_chinese(Permutation)
        Permutation.to_csv(path_min + 'Permutation_meituan.csv', encoding='utf_8_sig')


def main(path,path_max,path_min):
    data = original_process(path)
    data_elema, data_meituan = separate_channel(data)
    if data_meituan.empty == True:
        elema_main(data_elema, path_max, path_min)
    elif data_elema.empty == True:
        meituan_main(data_meituan, path_max, path_min)
    else:
        meituan_main(data_meituan,path_max,path_min)
        elema_main(data_elema,path_max,path_min)



if __name__ == "__main__":
    path = 'D:/AI/xianfengsg/other_project/product_profit/data/毛利测算测试数据1024.xlsx'
    path_max ='D:/AI/xianfengsg/other_project/product_profit/data/'
    path_min = 'D:/AI/xianfengsg/other_project/product_profit/data/'
    data = main(path,path_max,path_min)


#分sheet读取Excel内容

#
# data = read_excel1('D:/AI/xianfengsg/other_project/product_profit/毛利测算测试数据0902美团先试下.xlsx')
# print(data)