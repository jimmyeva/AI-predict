# -*- coding = utf-8 -*-
'''
@Time: 2018/10/29 13:46
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

def month_evalution(A,B,parameter_01):
    evaluation_factor_sum = pd.DataFrame()
    mid_used_sum = pd.DataFrame()
    for i in range(A,B):
        print(i)
        def Today_mysql():
            Today_date_01 = date.today() - timedelta(i)
            Today_date_02 = str(Today_date_01)
            Today_date = "date'%s'" % Today_date_02
            return Today_date, Today_date_01

        def get_basic_information():
            today, today_01 = Today_mysql()

            parameter_condition_sql = """select * from mid_parameter_condition where status =1"""
            parameter_condition = Mysql_Data(parameter_condition_sql)
            parameter_condition = parameter_condition.set_index('id')
            merge_evaluation = pd.DataFrame()
            for i in range(len(parameter_condition)):
                # import the parameters of custom_business_num ...
                parameters = parameter_condition.iloc[i, :]
                # global manufacturer_num, custom_business_num, custom_stock_num, custom_terminal_num,t2
                manufacturer_num = parameters['manufacturer_num']
                custom_business_num = parameters['custom_business_num']
                custom_stock_num = parameters['custom_stock_num']
                custom_terminal_num = parameters['custom_terminal_num']
                # 此处用来显示提前的日期构成
                interval_time = int(int(parameters['order_delivery_time']) + int(parameters['delivery_arrival_time']))
                interval_time_str = str(parameters['order_delivery_time']) + str('+') + str(parameters['delivery_arrival_time'])
                mid_parameter_condition_id = parameter_condition.index[i]
                print(i)
                print(interval_time)
                print(interval_time_str)

                # 从mid_evaluation表中获取成本，毛利和订单量,期初和期末相应的库存
                # -----------------------------------------------------------------------------------------------
                get_mid_evaluation = """select account_date,piece_bar_code,manufacturer_num,custom_business_num,custom_stock_num,custom_terminal_num,cost_untax,gross_profit_untax,
                                            start_available_inv_qty,end_available_inv_qty,start_inv_qty,end_inv_qty,
                                            sale_qty,order_qty
                                            from mid_evaluation  where account_date = %s 
                                            and manufacturer_num = %s
                                            and custom_business_num = %s 
                                            and custom_stock_num = %s
                                           and custom_terminal_num = %s""" % (
                today, manufacturer_num, custom_business_num, custom_stock_num, custom_terminal_num)
                get_mid_evaluation = Mysql_Data(get_mid_evaluation)
                get_mid_evaluation.columns = ['account_date', 'piece_bar_code', 'manufacturer_num', 'custom_business_num',
                                              'custom_stock_num', 'custom_terminal_num',
                                              'cost_untax', 'gross_profit_untax',
                                              'start_available_inv_qty', 'end_available_inv_qty', 'start_inv_qty',
                                              'end_inv_qty',
                                              'sale_qty', 'order_qty']
                #取计算当日第二天的实际销量情况，补货行为影响的是期末库存，而期末库存则会影响
                date_tomorrow = today_01 + timedelta(1)
                date_tomorrow = str(date_tomorrow)
                Date_tomorrow = "date'%s'" % date_tomorrow
                get_sale_tomorrow = """select account_date,piece_bar_code,manufacturer_num,custom_business_num,custom_stock_num,custom_terminal_num,
                                            sale_qty,order_qty
                                            from mid_evaluation where account_date = %s 
                                            and manufacturer_num = %s
                                            and custom_business_num = %s 
                                            and custom_stock_num = %s
                                           and custom_terminal_num = %s""" % (
                    Date_tomorrow, manufacturer_num, custom_business_num, custom_stock_num, custom_terminal_num)
                get_sale_tomorrow = Mysql_Data(get_sale_tomorrow)
                get_sale_tomorrow.columns = ['account_date', 'piece_bar_code', 'manufacturer_num', 'custom_business_num',
                                              'custom_stock_num', 'custom_terminal_num',
                                              'sale_qty_tomorrow', 'order_qty_tomorrow']
                get_mid_evaluation_mid = pd.merge(get_mid_evaluation, get_sale_tomorrow, on='piece_bar_code',
                                                  how='inner')
                get_goods = """select piece_bar_code from mid_cj_goods
                                where goods_allow = 'Y'"""
                get_goods = Mysql_Data(get_goods)
                get_goods.columns = ['piece_bar_code']
                # print(get_goods)
                get_mid_evaluation = pd.merge(get_mid_evaluation_mid,get_goods, on='piece_bar_code',
                                              how='inner')  # 选择inner是有可能有库存，但是sku是不可下单的SKU
                get_mid_evaluation['piece_bar_code'] = get_mid_evaluation['piece_bar_code'].apply(int)

                # 获取每日人工的下单量,#获取required_date是当天的订单量是多少
                # -----------------------------------------------------------------------------------------------
                get_order_qty = """SELECT piece_bar_code,po_bill_type,max_uom_quantity FROM cj_order
                                        WHERE ship_to_num in (2002639293,2002896806,2002763968,2001675058)
                                        and comments not like '%%批发%%'
                                         and audit_date = %s """ % (today)

                get_order_qty = Mysql_Data(get_order_qty)
                get_order_qty.columns = ['piece_bar_code', 'po_bill_type', 'max_uom_quantity']
                get_order_qty['po_bill_type'].replace('N', 3, inplace=True)
                get_order_qty['po_bill_type'].replace('Z', 3, inplace=True)
                get_order_qty['po_bill_type'].replace('A', 15, inplace=True)
                get_order_qty['po_bill_type'] = get_order_qty['po_bill_type'].apply(int)
                get_order_qty['piece_bar_code'] = get_order_qty['piece_bar_code'].apply(int)
                get_order_qty = process_order(get_order_qty)
                merge_manual_evaluation = pm_pbc(get_mid_evaluation, get_order_qty, 'left')
                merge_manual_evaluation = missing_value(merge_manual_evaluation)
                merge_manual_evaluation['piece_bar_code'] = merge_manual_evaluation['piece_bar_code'].apply(int)
                merge_manual_evaluation['po_bill_type'] = merge_manual_evaluation['po_bill_type'].apply(int)
                # 获取机器补货的建议数据
                #---------------------------------------------------------------------------
                get_artificial_qty = """select piece_bar_code,replenish_qty,interval_time from dm_cj_replenishment
                                            where cnt_at = %s
                                            and manufacturer_num = %s
                                            and custom_business_num = %s 
                                            and custom_stock_num = %s
                                           and custom_terminal_num = %s
                                           and interval_time = '%s'
                                          """ \
                                     % (today, manufacturer_num, custom_business_num, custom_stock_num, custom_terminal_num,
                                        interval_time_str)
                # print(get_artificial_qty)
                get_artificial_qty = Mysql_Data(get_artificial_qty)
                get_artificial_qty.columns = ['piece_bar_code', 'replenish_qty', 'interval_time']
                get_artificial_qty['interval_time'].replace('1+2', 3, inplace=True)
                get_artificial_qty['interval_time'].replace('10+2', 12, inplace=True)
                get_artificial_qty['interval_time'].replace('13+2', 15, inplace=True)
                get_artificial_qty['interval_time'] = get_artificial_qty['interval_time'].apply(int)
                get_artificial_qty['piece_bar_code'] = get_artificial_qty['piece_bar_code'].apply(int)
                # print(get_artificial_qty)
                merge_manual_artificial_evaluation = pm_pbc(merge_manual_evaluation, get_artificial_qty,
                                                            'left')  # 采用left的原因是因为有可能建议补货里面有的sku，例如6903148248461，但是现在的goods信息表里面是不允许下单的商品
                # print(merge_manual_artificial_evaluation)
                #以下的操作是获取从今天往前倒提前期的实际补货和建议补货的数量，可以理解为是今天的库存情况其实是在前提前期那天的下单量决定的
                # --------------------------------------------------------------------------------
                #代码笔记：对于interval_time我希望的对‘13+2’进行字符串的锁定，而不是一个int类型，那样的话就就是需要写mysql语句的时候
                #%s需要加上‘’，两个单引号进行锁定，这样就不需要对后面的筛选条件的内容进行修改
                #为什么在已知了某款SKU的提前期和其他条件的情况下还要对LT阶段的补货量进行匹配，因为如果不加入提前期这个筛选条件的话，可能会出现
                #组织终端都唯一的情况下出现两个提前期的补货量，对计算来说是一个不确定性
                date_LT_01 = today_01 - timedelta(interval_time)
                date_LT_02 = str(date_LT_01)
                date_LT = "date'%s'" % date_LT_02
                get_LT_replenshiment = """select piece_bar_code,interval_time,replenish_qty from dm_cj_replenishment
                                            where cnt_at = %s
                                            and manufacturer_num = %s
                                            and custom_business_num = %s
                                            and custom_stock_num = %s
                                           and custom_terminal_num = %s
                                           and interval_time = '%s'""" \
                                       % (date_LT, manufacturer_num, custom_business_num, custom_stock_num, custom_terminal_num,interval_time_str)
                get_LT_replenshiment = Mysql_Data(get_LT_replenshiment)
                get_LT_replenshiment.columns = ['piece_bar_code', 'interval_time', 'replenish_qty_before_LT']
                get_LT_replenshiment['interval_time'].replace('1+2', 3, inplace=True)
                get_LT_replenshiment['interval_time'].replace('10+2', 12, inplace=True)
                get_LT_replenshiment['interval_time'].replace('13+2', 15, inplace=True)
                get_LT_replenshiment['interval_time'] = get_LT_replenshiment['interval_time'].apply(int)
                get_LT_replenshiment['piece_bar_code'] = get_LT_replenshiment['piece_bar_code'].apply(int)
                merge_all_01 = pm_pbc_interval(merge_manual_artificial_evaluation, get_LT_replenshiment, 'inner')
                # merge_all_01[['po_bill_type', 'piece_bar_code']] = merge_all_01['po_bill_type', 'piece_bar_code'].apply(pd.to_numeric)
                # merge_all_01.apply(pd.to_numeric, errors='ignore')
                # 获取提前期前的那天的实际补货量是多少
                get_order_qty_before_LT = """SELECT piece_bar_code,po_bill_type,max_uom_quantity FROM cj_order
                                          WHERE ship_to_num in (2002639293,2002896806,2002763968,2001675058)
                                          and comments not like '%%批发%%'
                                           and audit_date = %s """ % (date_LT)
                get_order_qty_before_LT = Mysql_Data(get_order_qty_before_LT)
                get_order_qty_before_LT.columns = ['piece_bar_code', 'po_bill_type_before', 'max_uom_quantity_before_LT']
                get_order_qty_before_LT['po_bill_type_before'].replace('N', 3, inplace=True)
                get_order_qty_before_LT['po_bill_type_before'].replace('Z', 3, inplace=True)
                get_order_qty_before_LT['po_bill_type_before'].replace('A', 15, inplace=True)
                get_order_qty_before_LT = process_order_brfore(get_order_qty_before_LT)
                get_order_qty_before_LT[['po_bill_type_before', 'piece_bar_code']] = get_order_qty_before_LT[
                    ['po_bill_type_before', 'piece_bar_code']].apply(pd.to_numeric)
                get_order_qty_before_LT.apply(pd.to_numeric, errors='ignore')
                merge_all = pm_pbc(merge_all_01, get_order_qty_before_LT, 'left')
                merge_all = merge_all.drop_duplicates()
                merge_evaluation = merge_evaluation.append(merge_all,ignore_index=True)
                merge_evaluation = merge_evaluation.drop_duplicates()
                #merge_evaluation.to_csv('D:/project/P&G/Code/output/merge_evaluation.csv',encoding="utf_8_sig")
                #这一步操作的目的是为了剔除那些与要求提前期不同的补货量，整个evaluation的程序过程中都是进行左匹配的，那么就会导致一个情况
                #某些SKU会出现三种或者多种（以后可能会有）状态，那么程序会自动的对满足条形码的SKU进行数值填充
                #那样的话会出一个SKU，补货量一样但是会出现不同的提前期的行里面，这样就会导致数值增大
            merge_evaluation = missing_value(merge_evaluation)
            #此部操作是删除那些不匹配的提前期的补货数据，例如按照机器补货的建议，在15的提前期应该补货的是10件，找到15天提前期的补货行为
            #发现确实发生了补货，但是补货是选择提前期是三天的，所以对于前15天的补货行为不满足15天提前期下的补货，采取删除的操作


            for i in range(len(merge_evaluation)):
                # print(i)
                if merge_evaluation['po_bill_type'].iloc[i] != merge_evaluation['interval_time'].iloc[i]:
                    merge_evaluation['max_uom_quantity'].iloc[i] = 0
                else:
                    merge_evaluation['max_uom_quantity'].iloc[i] = merge_evaluation['max_uom_quantity'].iloc[i]
                if merge_evaluation['po_bill_type_before'].iloc[i] != merge_evaluation['interval_time'].iloc[i]:
                    merge_evaluation['max_uom_quantity_before_LT'].iloc[i] = 0
                else:
                    merge_evaluation['max_uom_quantity_before_LT'].iloc[i] = merge_evaluation['max_uom_quantity_before_LT'].iloc[i]
            merge_evaluation = merge_evaluation.drop_duplicates()
            #删除max_uom_quantity_before_LT值使0的行
            # merge_evaluation = merge_evaluation[~merge_evaluation['max_uom_quantity_before_LT'].isin([0])]
            # 下面这部操作是删除那些在提前期满足条件的实际补货为0的数据，达到真正的对比操作
            # -----------------------------------------------------------------------------------
            # 此操作是计算如果按照智能补货的建议的话，期初可用库存和期初实际库存的情况
            # 此步骤操作的目的是再计算的时候，每一个宝洁生产的SKU虽然不同的提前期，订单和销量是不一样，但库存是一致的，每一个SKU占了两行的位置，会导致出现双倍计算的情况
            for i in range(len(merge_evaluation)):
                if merge_evaluation['manufacturer_num'].iloc[i] == 320 & merge_evaluation['custom_stock_num'].iloc[i] == 1:
                    merge_evaluation['end_available_inv_qty'].iloc[i] = (merge_evaluation['end_available_inv_qty'].iloc[i]) * 0.5
                    merge_evaluation['end_inv_qty'].iloc[i] = (merge_evaluation['end_inv_qty'].iloc[i]) * 0.5
                    merge_evaluation['order_qty'].iloc[i] = (merge_evaluation['order_qty'].iloc[i]) * 0.5
                    merge_evaluation['sale_qty'].iloc[i] = (merge_evaluation['sale_qty'].iloc[i]) * 0.5
                    merge_evaluation['start_inv_qty'].iloc[i] = (merge_evaluation['start_inv_qty'].iloc[i]) * 0.5
                    merge_evaluation['start_available_inv_qty'].iloc[i] = (merge_evaluation['start_available_inv_qty'].iloc[i]) * 0.5
                    merge_evaluation['order_qty_tomorrow'].iloc[i] = (merge_evaluation['order_qty_tomorrow'].iloc[i]) * 0.5
                    merge_evaluation['sale_qty_tomorrow'].iloc[i] = (merge_evaluation['sale_qty_tomorrow'].iloc[i]) * 0.5
                #以下的操作是进行订单了的修正
                if merge_evaluation['order_qty'].iloc[i] > merge_evaluation['start_available_inv_qty'].iloc[i]:
                    merge_evaluation['order_qty_Revised'].iloc[i] = merge_evaluation['start_available_inv_qty'].iloc[i]
                else:
                    merge_evaluation['order_qty_Revised'].iloc[i] = merge_evaluation['order_qty'].iloc[i]
            #以下的操作是计算期末可用/实际库存的修正值
            merge_evaluation['end_inv_Revised'] = merge_evaluation['end_inv_qty'] + merge_evaluation['sale_qty'] - merge_evaluation['order_qty_Revised']
            merge_evaluation['end_available_inv_Revised'] = merge_evaluation['end_available_inv_qty'] + merge_evaluation['sale_qty'] - \
                                                  merge_evaluation['order_qty_Revised']
            merge_evaluation['end_inv_AI'] = merge_evaluation['end_inv_Revised'] - merge_evaluation[
                'max_uom_quantity_before_LT'] + merge_evaluation['replenish_qty_before_LT']
            merge_evaluation['end_available_inv_AI'] = merge_evaluation['end_available_inv_Revised'] - merge_evaluation[
                'max_uom_quantity_before_LT'] + merge_evaluation['replenish_qty_before_LT']

            #以下是计算AI补货出现的缺货量的计算，补货行为影响的缺货量是在第二天体现
            merge_evaluation['shortage_qty_Artificial'] = merge_evaluation['end_available_inv_AI'] - merge_evaluation['order_qty_tomorrow']
            #如果有缺货行为的话取绝对值
            abs(merge_evaluation['shortage_qty_Artificial'])
            merge_evaluation = merge_evaluation.drop_duplicates()
            return merge_evaluation
            # merge_evaluation.to_csv('D:/project/P&G/Code/output/merge_evaluation_test.csv',encoding="utf_8_sig")
            #设置新的表用来封装补货的各项指标
            #-----------------------------------------------------------------------------------------------
        def get_evaluation():
            logistics_fee = per_sku_oc()
            stock_fee= per_day_sku_sc()
            # print(stock_fee)
            mid_used= get_basic_information()
            # mid_used = mid_used.drop_duplicates(['custom_business_num','custom_stock_num',
            #                                      'custom_terminal_num','interval_time','piece_bar_code'])
            # mid_used.to_csv('D:/project/P&G/Code/output/mid_used_new.csv',encoding="utf_8_sig")

            #以下的操作是用来计算每一个的仿真指标
            # #------------------------------------------------------------------------------------------
            evaluation_factor = pd.DataFrame(columns= ['account_date','manufacturer_num',	'custom_business_num','custom_stock_num',
                                                       'custom_terminal_num	','interval_time',
                                            'piece_bar_code','manufacture','ROCC_manual','ROCC_AI',
                                        'logistical_manual','logistical_AI','stock_manual',
                                         'stock_AI','sc_manual','sc_AI','tc_manual','tc_AI','ROI_manual',
                                         'ROI_AI','profit_manual','profit_AI'])
            #以下为设计表格的基本信息
            evaluation_factor['account_date'] = mid_used['account_date']
            evaluation_factor['manufacturer_num'] = mid_used['manufacturer_num']
            evaluation_factor['custom_business_num'] = mid_used['custom_business_num']
            evaluation_factor['custom_stock_num'] = mid_used['custom_stock_num']
            evaluation_factor['custom_terminal_num'] = mid_used['custom_terminal_num']
            evaluation_factor['interval_time'] = mid_used['interval_time']
            evaluation_factor['piece_bar_code'] = mid_used['piece_bar_code']
            #计算人工补货的ROCC费用，物流费用，以及仓储费用，但是仓储费用针对每一个SKU应该是一个
            evaluation_factor['ROCC_manual'] = ((mid_used['start_available_inv_qty']-mid_used['order_qty_Revised'])
                                                *(mid_used['cost_untax'])*0.0002) + (mid_used['order_qty_Revised'] *
                                                                                     mid_used['cost_untax'])
            evaluation_factor['logistical_manual'] = mid_used['order_qty_Revised'] * logistics_fee
            evaluation_factor['stock_manual'] = (mid_used['end_inv_Revised']+mid_used['start_inv_qty']) * 0.5 * stock_fee
            evaluation_factor['sc_manual'] = 0

            #人工缺货成本的考虑，暂时先 不考虑
            #接下来是计算如果按照机器补货的策略应该会产生的成本
            evaluation_factor['ROCC_AI'] = ((mid_used['start_available_inv_qty']-mid_used['order_qty_Revised'])
                                                *(mid_used['cost_untax'])*0.0002) + (mid_used['order_qty_Revised'] *
                                                                                     mid_used['cost_untax'])
            evaluation_factor['logistical_AI'] = mid_used['order_qty_Revised'] * logistics_fee
            evaluation_factor['stock_AI'] = (mid_used['end_inv_AI']+mid_used['start_inv_qty'])* 0.5 * stock_fee
            evaluation_factor['sc_AI'] = mid_used['shortage_qty_Artificial'] * mid_used['gross_profit_untax'] * parameter_01
            evaluation_factor['tc_manual'] = evaluation_factor['sc_manual'] + evaluation_factor['stock_manual'] + evaluation_factor['logistical_manual']\
                                            + evaluation_factor['ROCC_manual']
            evaluation_factor['profit_manual'] = mid_used['order_qty_Revised'] *(mid_used['cost_untax'] +mid_used['gross_profit_untax'])
            evaluation_factor['ROI_manual'] = evaluation_factor['profit_manual']/evaluation_factor['tc_manual']
            evaluation_factor['tc_AI'] = evaluation_factor['sc_artificial'] + evaluation_factor['stock_artificial'] + evaluation_factor['logistical_artificial']\
                                             + evaluation_factor['ROCC_artificial']
            evaluation_factor['profit_AI'] = mid_used['order_qty_Revised'] *(mid_used['cost_untax'] +mid_used['gross_profit_untax'])
            evaluation_factor['ROI_AI'] = evaluation_factor['profit_artificial']/evaluation_factor['tc_artificial']
            return evaluation_factor,mid_used
        #此部的操作是将每一个循环的得到的两个dadaframe添加,总结之前是分别调用get_basic和evaluation的数据，导致程序运行缓慢，原因是在evaluation的函数是与get_basic函数耦合的
        #是有一定的关联性在的，也可以理解为evaluation的数据是依赖于get_basic的，如果是对两个函数都进行运行的话，会导致一个循环过后，还要再进行一次
        #运算，故运算量增加了一倍，极大的增加了运算的时间成本
        #关于append的API是一个比较好的dataframe叠加的函数，但是在循环的过程中，如果只是用了一次的append是会导致每次循环只保留了最新运行出来的结果
        #之所以会存在两个append同时使用的情况，那是因为我需要要对新的dataframe添加了数据之后还要再此基础上继续添加数据
        #在使用的过程中也遇到了，每次循环天数后对空的dataframe进行数据添加的过程中，总是会出现日期只在最新的一天的情况，那要是因为
        #这个问题目前还不能够给出正确的解释，但是我个人认为是append使用导致的，以为再出现了两次append后此类问题就得到了解决
        #对于for循环想要把得到的数据进行累加或者叠加操作的话，一般情况需要再大的循环发生之前就建立好一个空的列表或者Dataframe
        #以后再将每次的循环处理之后得到的数据进行添加操作
        evaluation_factor, mid_used = get_evaluation()
        evaluation_factor_sum = evaluation_factor_sum.append(evaluation_factor_sum.append(evaluation_factor))
        mid_used_sum = mid_used_sum.append(mid_used_sum.append(mid_used))
        mid_used_sum = mid_used_sum.drop_duplicates()
        evaluation_factor_sum = evaluation_factor_sum.drop_duplicates()
    return evaluation_factor_sum, mid_used_sum
evaluation_factor_sum, merge_evaluation_sum = month_evalution(23,53)
evaluation_factor_sum.to_csv('D:/project/P&G/Code/output/evaluation_factor_20_50_02.csv',encoding="utf_8_sig")
merge_evaluation_sum.to_csv('D:/project/P&G/Code/output/merge_evaluation_20_50_02.csv', encoding="utf_8_sig")

#这个函数是把所有创洁没有补货的数据给整体删除了，只比较了，AI补货和创洁切实补货的之间的差值






