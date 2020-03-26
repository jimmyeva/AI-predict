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
from datetime import timedelta
import datetime
import time
from Parameters import *
import Data_cleaning
import random
import warnings
import math

warnings.filterwarnings("ignore")
import pymysql

# ---------------------------------------------------------------------------------------------
## parameters for clean data(combine_data)
DataRead_0 = Original_Data()
DataClean = Data_cleaning.CleanData()


warnings.filterwarnings("ignore")
import pymysql

def Calculate_finance(Data_Frame,parameter):
    logistics_fee = per_sku_oc()
    stock_fee = per_day_sku_sc()

    # 以下的操作是用来计算每一个的仿真指标
    # #------------------------------------------------------------------------------------------
    calculate_finance = pd.DataFrame(columns=['cnt_at', 'manufacturer_num', 'custom_business_num', 'custom_stock_num',
                                              'custom_terminal_num', 'piece_bar_code', 'manufacturer_num', 'ROCC_manual', 'ROCC_AI',
                                              'logistical_manual', 'logistical_AI', 'stock_manual',
                                              'stock_AI', 'sc_manual', 'sc_AI', 'tc_manual',
                                              'tc_AI', 'ROI_manual',
                                              'ROI_AI', 'profit_manual', 'profit_AI','Sales_manual','Sales_AI'])
    # 以下为设计表格的基本信息
    calculate_finance['cnt_at'] = Data_Frame['account_date']
    calculate_finance['manufacturer_num'] = Data_Frame['manufacturer_num']
    calculate_finance['custom_business_num'] = Data_Frame['custom_business_num']
    calculate_finance['custom_stock_num'] = Data_Frame['custom_stock_num']
    calculate_finance['custom_terminal_num'] = Data_Frame['custom_terminal_num']
    # calculate_finance['interval_time'] = Data_Frame['interval_time']
    calculate_finance['piece_bar_code'] = Data_Frame['piece_bar_code']
    # 计算人工补货的ROCC费用，物流费用，以及仓储费用，但是仓储费用针对每一个SKU应该是一个
    calculate_finance['ROCC_manual'] = (Data_Frame['start_interest_inv'] - Data_Frame['order_qty']) * (
        Data_Frame['cost_untax']) * 0.0002 + Data_Frame['order_qty']*Data_Frame['cost_untax']
    calculate_finance['logistical_manual'] = Data_Frame['order_qty'] * logistics_fee
    calculate_finance['stock_manual'] = (Data_Frame['start_inv_qty']+Data_Frame['end_inv_qty']) * 0.5\
                                         * stock_fee
    calculate_finance['sc_manual'] = Data_Frame['shortage_qty_manual'] * Data_Frame['gross_profit_untax'] * parameter

    # 人工缺货成本的考虑，暂时先 不考虑
    # 接下来是计算如果按照机器补货的策略应该会产生的成本
    calculate_finance['ROCC_AI'] = (Data_Frame['start_interest_inv_AI'] - Data_Frame['order_qty_AI']) * (
        Data_Frame['cost_untax']) * 0.0002 +Data_Frame['order_qty_AI']*Data_Frame['cost_untax']
    calculate_finance['logistical_AI'] = Data_Frame['order_qty_AI'] * logistics_fee
    calculate_finance['stock_AI'] = (Data_Frame['start_inv_qty_AI']+Data_Frame['end_inv_qty_AI']) * 0.5\
                                         * stock_fee
    calculate_finance['sc_AI'] = Data_Frame['shortage_qty_AI'] * Data_Frame['gross_profit_untax'] * parameter
    calculate_finance['tc_manual'] = calculate_finance['sc_manual'] + calculate_finance['stock_manual'] + \
                                     calculate_finance['logistical_manual'] \
                                     + calculate_finance['ROCC_manual']
    calculate_finance['profit_manual'] = Data_Frame['delivery_qty'] * Data_Frame['gross_profit_untax']
    calculate_finance['ROI_manual'] = calculate_finance['profit_manual'] / calculate_finance['tc_manual']
    calculate_finance['tc_AI'] = calculate_finance['sc_AI'] + calculate_finance['stock_AI'] + \
                                 calculate_finance['logistical_AI'] \
                                         + calculate_finance['ROCC_AI']
    calculate_finance['profit_AI'] = Data_Frame['delivery_qty_AI'] * Data_Frame['gross_profit_untax']
    calculate_finance['ROI_AI'] = calculate_finance['profit_AI'] / calculate_finance['tc_AI']
    calculate_finance['Sales_manual'] = Data_Frame['order_qty'] * \
                                        (Data_Frame['gross_profit_untax']+Data_Frame['cost_untax'])
    calculate_finance['Sales_AI'] = Data_Frame['order_qty_AI'] \
                                    *  (Data_Frame['gross_profit_untax']+Data_Frame['cost_untax'])
    return calculate_finance



def start_end_date(A,B,parameter):
    #新建一个空的Dataframe便于以后的数据对内存入
    basic_sheet_merge = pd.DataFrame
    final_sheet = pd.DataFrame
    #replenishment_AI_total是仿真环境里面每天补货的填入
    replenishment_AI_total = pd.DataFrame(columns=['piece_bar_code', 'account_date','custom_stock_num','custom_business_num',
                              'custom_terminal_num', 'interval_time', 'suggestion_qty','purchase_order_qty_AI', 'arrived_date'])
    replenishment_AI_total[['custom_stock_num', 'custom_terminal_num','custom_business_num']] = \
        replenishment_AI_total[['custom_stock_num', 'custom_terminal_num','custom_business_num']].apply(pd.to_numeric)

    for days in pd.date_range(A,B):
        print(A)
        mid_get_replenishment = pd.DataFrame
        today = days
        print(today)
        days = days.strftime('%Y%m%d')
        print(days)
        get_mid_evaluation_new = """select account_date,piece_bar_code,manufacturer_num,custom_business_num,custom_stock_num,
        custom_terminal_num,gross_profit_untax,sale_qty,order_qty from mid_evaluation  where account_date = %s""" % (days)
        get_mid_evaluation_new = Mysql_Data(get_mid_evaluation_new)
        get_mid_evaluation_new.columns = ['account_date', 'piece_bar_code', 'manufacturer_num', 'custom_business_num',
                                          'custom_stock_num', 'custom_terminal_num',
                                           'gross_profit_untax','delivery_qty','order_qty']
        # 在获取一下进销存的数据
        get_Invoicing_sql = """select account_date,custom_stock_num,custom_business_num,piece_bar_code,
                                    storge_move,refund_move,return_move,overflow_qty,storage_remove,cost_untax,
                                    inv_begin_qty,inv_end_qty,inv_begin_storge_qty,inv_end_storge_qty
                            	from dm_inv_parameters_revised_real where account_date =%s""" % (days)
        get_Invoicing_sql = Mysql_Data(get_Invoicing_sql)
        get_Invoicing_sql.columns = ['account_date', 'custom_stock_num', 'custom_business_num', 'piece_bar_code',
                                     'storge_move', 'refund_move', 'return_move',
                                     'overflow_qty', 'storage_remove','cost_untax','start_interest_inv', 'end_interest_inv', 'start_inv_qty',
                                          'end_inv_qty']
        # 取商品信息的目的是为了屏蔽那些不可下单的商品清单
        get_goods = """select piece_bar_code from mid_cj_goods
                        where goods_allow = 'Y'"""
        get_goods = Mysql_Data(get_goods)
        # get_goods.to_csv('D:/project/P&G/Code/output/get_goods.csv', encoding="utf_8_sig")
        get_goods.columns = ['piece_bar_code']
        get_Invoicing = pd.merge(get_Invoicing_sql, get_goods, on='piece_bar_code', how='inner')
        get_Invoicing[['custom_stock_num', 'custom_business_num']] = \
            get_Invoicing[['custom_stock_num', 'custom_business_num']].apply(pd.to_numeric)
        # get_Invoicing.to_csv('D:/project/P&G/Code/output/test_01.csv',encoding="utf_8_sig")
        get_mid_evaluation_new = pd.merge(get_mid_evaluation_new, get_goods, on='piece_bar_code',
                                          how='inner')  # 选择inner是有可能有库存，但是sku是不可下单的SKU
        get_mid_evaluation_new[['custom_stock_num', 'custom_business_num','custom_terminal_num']] = \
            get_mid_evaluation_new[['custom_stock_num', 'custom_business_num','custom_terminal_num']].apply(pd.to_numeric)
        # get_mid_evaluation_new.to_csv('D:/project/P&G/Code/output/get_mid_evaluation_new.csv',encoding="utf_8_sig")
        print('get_mid_evaluation_new')
        print(get_mid_evaluation_new.empty)
        print('get_Invoicing')
        print(get_Invoicing.empty)
        mid_basic_sheet = pd.merge(get_mid_evaluation_new, get_Invoicing,
                               on=['piece_bar_code', 'custom_stock_num', 'custom_business_num','account_date'],
                               how='inner')
        #接下来的操作是存在两个dataframe ，一个是获取人工采取AI建议的每天的实际下单量，另一个是获取那些并没有采用建议的人工下单的实际到货的实际到货的日期，
        #用于去匹配那些由于信息不全导致的缺货，在仿真的信息内加入
        get_order_manual = """SELECT audit_date,piece_bar_code,min_uom_quantity FROM cj_order co WHERE co.ship_to_num IN 
        (2002639293, 2002763968, 2002896806, 2001675058, 2002898840, 2001683243, 2001677839)
        AND co.comments lIKE '汉口%%' AND co.audit_date =%s""" % (days)
        get_order_manual_sql = Mysql_Data(get_order_manual)
        get_order_manual_sql.columns = ['account_date', 'piece_bar_code', 'purchase_order_manual']
        #这里需要做一个判断，如果当天没有发生下单行为，这里不需要进行合并的操作
        if get_order_manual_sql.empty == True:
            basic_sheet_mid = mid_basic_sheet
            basic_sheet_mid['purchase_order_manual'] = 0
        else:
            # 先将获取人工创洁向宝洁下单的数据进行合并
            # 这部操作是为了防止发生创洁当天会有几个下单行为
            get_order_manual_sql = get_order_manual_sql.groupby(['account_date', 'piece_bar_code'], as_index=False).agg(
                sum)
            basic_sheet_mid = pd.merge(mid_basic_sheet, get_order_manual_sql, on=['account_date', 'piece_bar_code'],
                                       how='left')
            basic_sheet_mid = basic_sheet_mid.fillna(0)
        basic_sheet_mid.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/v1---仿真/02_仿真输出/evaluation-17/basic_sheet_mid' + str(
                days) + '.csv', encoding="utf_8_sig")
        #接下来要获取的是并没有按照网仓建议值下单的下单行为，这里主要是区分那些额外的下单的实际到货的日期
        get_order_required_manual = """SELECT require_date,piece_bar_code,min_uom_quantity FROM cj_order co WHERE co.ship_to_num IN 
        (2002639293, 2002763968, 2002896806, 2001675058, 2002898840, 2001683243, 2001677839)
        AND co.comments NOT lIKE '汉口%%' AND co.require_date = %s""" % (days)
        get_order_required_manual_sql = Mysql_Data(get_order_required_manual)
        get_order_required_manual_sql.columns = ['account_date', 'piece_bar_code','arrived_order_manual']
        # 这里需要做一个判断，如果当天没有发生下单行为，这里不需要进行合并的操作
        if get_order_required_manual_sql.empty == True:
            basic_sheet = mid_basic_sheet
            basic_sheet['arrived_order_manual'] = 0
        else:
            # 这部操作是为了防止发生创洁当天会有几个下单行为
            get_order_required_manual_sql = get_order_required_manual_sql.groupby(['account_date', 'piece_bar_code'],
                                                                                  as_index=False).agg(sum)
            # 先将获取人工创洁向宝洁下单的数据进行合并
            basic_sheet = pd.merge(basic_sheet_mid, get_order_required_manual_sql, on=['account_date', 'piece_bar_code'],
                                   how='left')
            basic_sheet = basic_sheet.fillna(0)
        basic_sheet['account_date'] = today
        #将所有的日期格式转换成统一
        basic_sheet['account_date'] = basic_sheet['account_date'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
        # replenishment_AI是用于在for循环内每天的补货数据
        replenishment_AI = pd.DataFrame(columns=['piece_bar_code', 'account_date','custom_stock_num',
                                                              'custom_business_num','custom_terminal_num', 'interval_time',
                                                              'suggestion_qty', 'purchase_order_qty_AI','arrived_date'])
        basic_sheet.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/v1---仿真/02_仿真输出/evaluation-17/basic_sheet' + str(
                days) + '.csv', encoding="utf_8_sig")
        # 在计算AI库存等数据的时候，如果是仿真第一天使用真实数据，从第二天开始使用前一天的数据进行计算
        # 以下是将第一天的数据与每一次循环的基本数据进行合并
        if days == A:
            basic_sheet['start_interest_inv_AI'] = basic_sheet['start_interest_inv']
            basic_sheet['end_interest_inv_AI'] = basic_sheet['end_interest_inv']
            basic_sheet['start_inv_qty_AI'] = basic_sheet['start_interest_inv']
            basic_sheet['end_inv_qty_AI'] = basic_sheet['end_interest_inv']
            basic_sheet['order_qty_AI'] = basic_sheet['order_qty']
            basic_sheet['delivery_qty_AI'] = basic_sheet['delivery_qty']
            basic_sheet['shortage_qty_AI'] = 0
            basic_sheet['shortage_qty_manual'] = 0
            basic_sheet['storage_remove_AI'] = 0
            #这里需要做一步数据判断的操作，存在在算进销存的时候，出现负值的情况，因此在算出为负值的时候，将库存标记为0
            basic_sheet['start_interest_inv_AI'] = basic_sheet['start_interest_inv_AI'].apply(
                lambda x: 0 if x < 0 else x)
            basic_sheet['end_interest_inv_AI'] = basic_sheet['end_interest_inv_AI'].apply(
                lambda x: 0 if x < 0 else x)
            basic_sheet['start_inv_qty_AI'] = basic_sheet['start_inv_qty_AI'].apply(
                lambda x: 0 if x < 0 else x)
            basic_sheet['end_inv_qty_AI'] = basic_sheet['end_inv_qty_AI'].apply(
                lambda x: 0 if x < 0 else x)

            parameter_condition_sql = """select * from mid_parameter_condition where status = 1 
                                          and custom_start_time <= %s and custom_end_time >= %s """ %(days,days)
            parameter_condition_read = DataRead_0.Mysql_Data(parameter_condition_sql)
            #加快速度只算311的情况
            parameter_condition= parameter_condition_read[parameter_condition_read['custom_business_num'] == 3]
            print(len(parameter_condition))

            parameter_condition = parameter_condition.set_index('id')
            # 新建一个dataframe用于存储仿真环境中AI的补货量
            ## parameters for weeklist that caculate the error between predict num and real num
            end_time = today
            start_time = end_time - timedelta(15*28-1)
          ## parameters for predict time
            end_time_pre = today
            start_time_pre = end_time_pre - timedelta(30)
            srq_end = today
            srq_start = srq_end - timedelta(91)
            ## date parameter for sale_sql
            sale_sql_date = today-timedelta(400)

            start_sale_time = sale_sql_date.strftime('%Y%m%d')
            ## date parameter for usable_inventory_sql
            usable_inventory_time = "date'%s'"%today
            order_date = today - timedelta(today.day)
            order_time = order_date.strftime('%Y%m%d')
            quota_time = days
            t3 = 35 # for predictive cycle roll time
            sigma = 1.2 #for The uncertainty of prediction
            replenishment = pd.DataFrame(columns=['piece_bar_code', 'account_date','custom_stock_num','custom_business_num',
                                      'custom_terminal_num', 'interval_time', 'suggestion_qty',
                                      'purchase_order_qty_AI', 'arrived_date'])
            #以下操作是在仿真环境中run一次补货算法
            for i in range(len(parameter_condition)):
                # import the parameters of custom_business_num ...
                parameters = parameter_condition.iloc[i, :]
                # global manufacturer_num, custom_business_num, custom_stock_num, custom_terminal_num,t2
                manufacturer_num = parameters['manufacturer_num']
                custom_business_num = parameters['custom_business_num']
                custom_stock_num = parameters['custom_stock_num']
                custom_terminal_num = parameters['custom_terminal_num']
                interval_time = int(parameters['order_delivery_time']) + int(parameters['delivery_arrival_time'])
                mid_parameter_condition_id = parameter_condition.index[i]
                t2 = int(parameters['arrival_time'])
                print(manufacturer_num, custom_business_num, custom_stock_num, custom_terminal_num, t2)
                # Parameters.sql_read(manufacturer_num,custom_business_num,custom_stock_num,custom_terminal_num)
                predict_sql = """select cnt_at,piece_bar_code,forecast_qty from dm_cj_forecast 
                                                 where manufacturer_num = %s
                                                  and custom_business_num = %s
                                                 and custom_stock_num = %s
                                                 and custom_terminal_num = %s
                                                 and belonged_date= %s
                                                 """ % (
                manufacturer_num, custom_business_num, custom_stock_num, custom_terminal_num, days)
                sale_sql = """select piece_bar_code ,delivery_qty,account_date from mid_cj_sales
                                              where manufacturer_num = %s
                                              and custom_business_num = %s 
                                              and custom_stock_num = %s
                                              and custom_terminal_num = %s
                                              and account_date > %s""" \
                           % (manufacturer_num, custom_business_num, custom_stock_num, custom_terminal_num, start_sale_time)
                usable_inventory_sql_mid = basic_sheet[basic_sheet['custom_stock_num'] == custom_stock_num]
                usable_inventory_sql = pd.DataFrame(columns=['piece_bar_code', 'account_date', 'available_inv_qty'])
                usable_inventory_sql['piece_bar_code'] = usable_inventory_sql_mid['piece_bar_code']
                usable_inventory_sql['account_date'] = usable_inventory_sql_mid['account_date']
                usable_inventory_sql['available_inv_qty'] = usable_inventory_sql_mid['start_interest_inv_AI']
                ## the parameters for base information of sku:including brand ,box factor
                base_information_sql = """select piece_bar_code,category_num,spaced,
                                                                  brand_num,case_conversion_rate,segment5,status   
                                                          from mid_cj_goods """
                ## parameters for order time
                order_sql = """select order_date,piece_bar_code,max_uom_qty from mid_cj_order
                                                where manufacturer_num = %s
                                                and custom_business_num = %s 
                                                and custom_stock_num = %s
                                                and order_date>= %s """ % (
                    manufacturer_num, custom_business_num, custom_stock_num, order_time)
                ## the parameters for usable quota of promotional sku
                usable_promotion_quota_sql = """SELECT effective_start_date,piece_bar_code,quota  from mid_cj_quota a 
                                                               where custom_business_num = %s 
                                                               and effective_start_date < %s
                                                               and effective_start_date  > %s""" % (
                custom_business_num, quota_time,quota_time)
                ## the parameters for feedback frame
                feedback_sql = """select piece_bar_code, correct_cause_id from mid_cj_cause
                                where correct_cause_id in (1,2)"""
                if manufacturer_num == '000320':
                    def Deal_with_Sale_Data():
                        Sale_Data = DataRead_0.Mysql_Data(sale_sql)
                        Sale_Data.columns = ['BJ_BARCODE', 'BJ_TOTAL_QTY', 'BJ_DATE']
                        Sale_Data['BJ_DATE'] = pd.to_datetime(Sale_Data['BJ_DATE']).dt.normalize()
                        Sale_Data_group = Sale_Data.groupby(['BJ_BARCODE', 'BJ_DATE'], as_index=False)
                        Sale_Data_group_sum = Sale_Data_group.agg(sum)
                        ## fill  0 which has no number in that day
                        # the function use for fill the number which frame has no number of other days
                        Min_Date = min(Sale_Data_group_sum['BJ_DATE'])
                        Max_Date = max(Sale_Data_group_sum['BJ_DATE'])
                        Time_Series = pd.DataFrame(pd.date_range(Min_Date, Max_Date), columns=['BJ_DATE'])

                        ## the method for fill the dataframe
                        def fill_frame(df, TS=Time_Series, column='BJ_DATE'):
                            sale_merge = pd.merge(TS, df, on=column, how='left')
                            sale_merge['BJ_TOTAL_QTY'] = sale_merge['BJ_TOTAL_QTY'].fillna(0)
                            try:
                                barcode = list(set(sale_merge['BJ_BARCODE'][sale_merge['BJ_BARCODE'].notna()]))[0]
                            except IndexError:
                                barcode = 'unknow'
                            sale_merge['BJ_BARCODE'] = sale_merge['BJ_BARCODE'].fillna(barcode)
                            return sale_merge

                        ## select the data that use for calculate error between predict data and real data
                        ## use apply methon fullfill the split-apply-combine,group_keys=False can cancel the index of group
                        Sale_Data_Fill = Sale_Data_group_sum.groupby(['BJ_BARCODE'], group_keys=False).apply(fill_frame)

                        ## add week attribute information
                        Sale_Data_Fill['WEEK_DAY'] = DataClean.to_weekday(Sale_Data_Fill['BJ_DATE'])

                        ## select the date in a month
                        Sale_Selected_Date = Sale_Data_Fill[
                            Sale_Data_Fill['BJ_DATE'].isin(DataRead_0.rq_range(start_time_pre, end_time_pre))]
                        ## use the sales data to calculate week probablity and calculate the maximal storage
                        Sale_Data_Week = Sale_Data_Fill[
                            Sale_Data_Fill['BJ_DATE'].isin(DataRead_0.rq_range(start_time, end_time))]
                        Week_agg = Sale_Data_Week.groupby(['BJ_BARCODE', 'WEEK_DAY'], as_index=False).agg(sum)
                        Week_sum = Week_agg.groupby(['BJ_BARCODE'], as_index=False)
                        Week_sum = Week_sum['BJ_TOTAL_QTY'].agg(sum)
                        Week_agg_sum = pd.merge(Week_agg, Week_sum, on='BJ_BARCODE', how='left')
                        Week_agg_sum['PROB'] = Week_agg_sum['BJ_TOTAL_QTY_x'] / (Week_agg_sum['BJ_TOTAL_QTY_y'] + 0.0000001)

                        Sale_Barcode = set(Week_agg_sum['BJ_BARCODE'])
                        return Sale_Selected_Date, Week_agg_sum, Sale_Barcode, Sale_Data_Fill

                    ## deal with the storage of today:Instore_Road_Store
                    def Deal_with_Instorage():
                        Instore_Road_Store = usable_inventory_sql
                        Instore_Road_Store.columns = ['BARCODE', 'CNT_AT', 'PURCHASE_TOTAL_NUM']
                        Instore_Road_Store_Selected = Instore_Road_Store[['BARCODE', 'PURCHASE_TOTAL_NUM']]
                        Instore_Barcode = set(Instore_Road_Store_Selected['BARCODE'])
                        return Instore_Road_Store_Selected, Instore_Barcode

                    ## deal with the predict data
                    def Deal_With_Predict():
                        Pre_Data = DataRead_0.Mysql_Data(predict_sql)
                        Pre_Data = DataClean.missing_value(Pre_Data)
                        Pre_Data = DataClean.to_str(Pre_Data, 'piece_bar_code')
                        Pre_Barcode = set(Pre_Data['piece_bar_code'])
                        return Pre_Data, Pre_Barcode

                    Sale_Selected_Date, Week_agg_sum_group, Sale_Barcode, Sale_Data_Fill = Deal_with_Sale_Data()
                    Instore_Road_Store_Selected, Instore_Barcode = Deal_with_Instorage()
                    Pre_Data, Pre_Barcode = Deal_With_Predict()
                    Unique_Barcode = set(Sale_Barcode & Instore_Barcode & Pre_Barcode)
                    # =========================================================================================================
                    Final_Ordering = pd.DataFrame(
                        columns=["BARCODE", "Usable_Storage", "Max_SS", "Order_or_not", "Order_num"])
                    for barcode in Unique_Barcode:
                        usable_storage = Instore_Road_Store_Selected[Instore_Road_Store_Selected['BARCODE'] == barcode]
                        if usable_storage.empty:
                            usable_inv = 0
                        else:
                            usable_inv = float(usable_storage['PURCHASE_TOTAL_NUM'])

                        # sales data of a sku
                        sales_qty = Sale_Selected_Date[Sale_Selected_Date['BJ_BARCODE'] == barcode]
                        predict_sales = Pre_Data[Pre_Data['piece_bar_code'] == barcode]
                        predict_sales['WEEK_DAY'] = DataClean.to_weekday(predict_sales['cnt_at'])

                        ## deal with predict sales
                        predict_sales_agg = predict_sales.groupby(['cnt_at'], as_index=False).mean()

                        ## get out the week list
                        week_list = Week_agg_sum_group[Week_agg_sum_group['BJ_BARCODE'] == barcode]
                        learn_error = pd.DataFrame({
                            "bj_date": np.array(sales_qty['BJ_DATE']),
                            "real_sales": np.array(sales_qty['BJ_TOTAL_QTY']),
                            "pre_sales": np.array(predict_sales_agg['forecast_qty'][0:len(sales_qty)]),
                            "error": np.array(np.array(sales_qty['BJ_TOTAL_QTY']) - np.array(
                                predict_sales_agg['forecast_qty'][0:len(sales_qty)])),
                            "WEEK_DAY": np.array(DataClean.to_weekday(sales_qty['BJ_DATE']))
                        })
                        learn_error_join = pd.merge(learn_error, week_list, on='WEEK_DAY', how='left')
                        learn_error_join['prob_star'] = learn_error_join['PROB'] / (
                                    sum(learn_error_join['PROB']) + 0.000001)
                        learn_error_join['prob_num'] = sum(learn_error_join['error']) * learn_error_join['prob_star']

                        learn_error_join_select = learn_error_join[['prob_num', 'WEEK_DAY']]
                        basic_error_num = learn_error_join_select.drop_duplicates(subset=['WEEK_DAY'], keep='first')
                        predict_sales_num = pd.merge(predict_sales, basic_error_num, on='WEEK_DAY', how="left")

                        ## deal with the predict data,put it as a list type,distinct the circulation
                        pre_list = DataClean.agg_to_frame(predict_sales, t3)
                        error_list = DataClean.agg_to_frame(predict_sales_num, t3)

                        # -------------------------------------------------------------------------------------
                        # 1 prediction sum for t2 days
                        pre_sum = DataClean.Pre_Error_sum_320(pre_list, 'forecast_qty', t2, t3)
                        error_sum = DataClean.Pre_Error_sum_320(error_list, 'prob_num', t2, t3)

                        # ------------------------------------------------------------------------------------

                        Error_sum_days_sigma = DataClean.Err_sigma_V1(error_sum, sigma)

                        Max_SS = np.array(pre_sum) + np.array(Error_sum_days_sigma)

                        # ------------------------------------------------------------------------------------------
                        ## if order or not
                        def Order_num():
                            Available_in_stock = usable_inv
                            if Available_in_stock < Max_SS[len(Max_SS) - 1]:
                                order_or_not = 1
                                Order_num = Max_SS[len(Max_SS) - 1] - Available_in_stock
                            else:
                                order_or_not = 2
                                Order_num = 0
                            return order_or_not, Order_num

                        # --------------------------------------------------------------------------------------------
                        order_or_not, Order_num = Order_num()
                        Order_frame_End = {'BARCODE': barcode,
                                           'Usable_Storage': usable_inv,
                                           'Max_SS': Max_SS[len(Max_SS) - 1],
                                           'Order_or_not': order_or_not,
                                           'Order_num': Order_num}
                        Order_frame_End = pd.DataFrame(Order_frame_End, index=[0])
                        Final_Ordering = pd.concat([Final_Ordering, Order_frame_End])
                    Final_Ordering.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/'
                                          'v1---仿真/02_仿真输出/evaluation-18/piece_bar_code/Final_Ordering' + str(t2) + str(
                        days) + '.csv', encoding="utf_8_sig")
                    Base_Information = DataRead_0.Mysql_Data(base_information_sql)
                    Base_Information.columns = ['BARCODE', 'CATEGORY_CODE', 'SPACED',
                                                'BRAND_NUM', 'OUT_FACTOR', 'PRODUCT_NATURE', 'STATUS']
                    ## delete the duplicate number
                    Base_Information_Unique = Base_Information
                    ## through the sales of every month,calculate mean of sales number
                    sales_slow_selected = Sale_Data_Fill[
                        Sale_Data_Fill['BJ_DATE'].isin(DataRead_0.rq_range(srq_start, srq_end))]
                    sales_slow_selected_delete_group = sales_slow_selected.groupby(['BJ_BARCODE'], as_index=False)
                    sales_slow_selected_month = sales_slow_selected_delete_group['BJ_TOTAL_QTY'].mean()
                    sales_slow_selected_month['BJ_TOTAL_QTY'] = sales_slow_selected_month['BJ_TOTAL_QTY'] * 30
                    sales_slow_selected_month.columns = ['BJ_BARCODE', 'Month_Sale']
                    ## join the box factor
                    box_factor_frame = Base_Information_Unique[['BARCODE', 'OUT_FACTOR']]
                    slow_sales_month = pd.merge(sales_slow_selected_month, box_factor_frame, how='left', right_on='BARCODE',
                                                left_on='BJ_BARCODE')
                    slow_sales_month['OUT_FACTOR'] = pd.to_numeric(slow_sales_month['OUT_FACTOR'].fillna(1))
                    slow_sales_month['month_box_sale'] = slow_sales_month['Month_Sale'] / slow_sales_month['OUT_FACTOR']
                    slow_sales_month['slow_selling'] = list(
                        map(lambda x: 1 if x <= 1 else 2, slow_sales_month['month_box_sale']))

                    ## 1 mean slow good and 2 mean not slow goods
                    slow_barcode = slow_sales_month[['BJ_BARCODE', 'month_box_sale', 'slow_selling']]

                    ##=================================================================================================
                    ## calculate the usable quota
                    Promotion_Quato = DataRead_0.Mysql_Data(usable_promotion_quota_sql)
                    Promotion_Quato.columns = ['CNT_AT', 'BARCODE', 'QUOTA_QTY']
                    Promotion_Quato['CNT_AT'] = pd.to_datetime(Promotion_Quato['CNT_AT']).dt.normalize()
                    quota_frame = Promotion_Quato.groupby(['BARCODE'], as_index=False).sum()

                    ## though the order frame calculate the usable quota
                    Order_Frame = DataRead_0.Mysql_Data(order_sql)
                    Order_Frame.columns = ['AUDIT_DATE', 'BARCODE', 'QUANTITY']
                    Order_Frame['AUDIT_DATE'] = pd.to_datetime(Order_Frame['AUDIT_DATE']).dt.normalize()
                    Order_Frame_group = Order_Frame.groupby(['BARCODE'], as_index=False).sum()
                    Order_Frame_num = Order_Frame_group[['BARCODE', 'QUANTITY']]

                    quota_ordered_frame = pd.merge(quota_frame, Order_Frame_num, on='BARCODE', how='left')
                    quota_ordered_frame['QUANTITY'] = quota_ordered_frame['QUANTITY'].fillna(0)
                    quota_ordered_frame['USABLE_QUOTY'] = quota_ordered_frame['QUOTA_QTY'] - quota_ordered_frame['QUANTITY']
                    quota_ordered_frame['USABLE_QUOTY'][quota_ordered_frame['USABLE_QUOTY'] <= 0] = 0
                    quota_barcode = quota_ordered_frame.drop_duplicates(subset=['BARCODE'], keep='first')
                    quota_barcode = quota_barcode[['BARCODE', 'USABLE_QUOTY']]
                    ## join the Final_Ordering with the box factor
                    Final_Ordering_Box = pd.merge(Final_Ordering, box_factor_frame, how='left', on='BARCODE')
                    Final_Ordering_Box['OUT_FACTOR'] = pd.to_numeric(Final_Ordering_Box['OUT_FACTOR'].fillna(1))
                    Final_Ordering_Box['ORDER_NUM_BOX'] = Final_Ordering_Box['Order_num'] / Final_Ordering_Box['OUT_FACTOR']
                    Final_Ordering_Box['ORDER_NUM_BOX'] = list(
                        map(lambda x: round(x), Final_Ordering_Box['ORDER_NUM_BOX']))

                    ## join the slow sales factor
                    Final_Ordering_Box_Slow = pd.merge(Final_Ordering_Box, slow_barcode, left_on='BARCODE',
                                                       right_on='BJ_BARCODE',
                                                       how='left')
                    Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'] = Final_Ordering_Box_Slow['ORDER_NUM_BOX']
                    Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'][Final_Ordering_Box_Slow['slow_selling'] == 1] = 0
                    Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'] = Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'].fillna(0)
                    Final_Ordering_Box_Slow.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/'
                                          'v1---仿真/02_仿真输出/evaluation-18/piece_bar_code/Final_Ordering_Box_Slow' + str(t2) + str(
                        days) + '.csv', encoding="utf_8_sig")
                    ## join the quota information
                    Final_Ordering_Box_Quota = pd.merge(Final_Ordering_Box_Slow, quota_barcode, on='BARCODE', how='left')
                    Rep_Base = Base_Information_Unique[
                        ['BARCODE', 'CATEGORY_CODE', 'SPACED', 'BRAND_NUM', 'OUT_FACTOR', 'PRODUCT_NATURE', 'STATUS']]
                    Final_Ordering_Box_Quota_Base = pd.merge(Final_Ordering_Box_Quota, Rep_Base, on='BARCODE', how='left')
                    Final_Ordering_Box_Quota_Base['USABLE_QUOTY'] = Final_Ordering_Box_Quota_Base['USABLE_QUOTY'].fillna(0)
                    Final_Ordering_Box_Quota_Base['USABLE_QUOTY'][
                        Final_Ordering_Box_Quota_Base['PRODUCT_NATURE'] == 'Y'] = float(
                        'inf')

                    ## if the order number is bigger than quota,use the quota replace for it
                    Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'] = 0
                    Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'][
                        Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s'] > Final_Ordering_Box_Quota_Base['USABLE_QUOTY']] = \
                        Final_Ordering_Box_Quota_Base['USABLE_QUOTY']
                    Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'][
                        Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s'] <= Final_Ordering_Box_Quota_Base['USABLE_QUOTY']] = \
                        Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s']

                    ## join the product category
                    Final_Ordering_Box_V0 = Final_Ordering_Box_Quota_Base
                    Final_Ordering_Box_V0.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/'
                                          'v1---仿真/02_仿真输出/evaluation-18/piece_bar_code/Final_Ordering_Box_V0' + str(t2) + str(
                        days) + '.csv', encoding="utf_8_sig")
                    ## join the objective information
                    FeedBack = DataRead_0.Mysql_Data(feedback_sql)
                    FeedBack.columns = ['BARCODE', 'REMARKS']
                    stop_barcode = list(set(FeedBack['BARCODE']))

                    Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback'] = list(
                        map(lambda x: math.floor(x), Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL']))
                    Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback'][
                        Final_Ordering_Box_V0['BARCODE'].isin(stop_barcode)] = 0
                    Final_Ordering_Box_V0['BJ_DATE'] = today
                    Final_Ordering_Box_V0['USABLE_BOX'] = Final_Ordering_Box_V0['Usable_Storage'] / Final_Ordering_Box_V0[
                        'OUT_FACTOR_x']
                    # 这一步是将原来考虑进入的装箱因子给反乘回来
                    Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback'] = Final_Ordering_Box_V0[
                                                                                 'ORDER_NUM_BOX_FINAL_causeback'] * \
                                                                             Final_Ordering_Box['OUT_FACTOR']
                    # 以下是对补货的输出进行修改，来满足仿真的需要输出格式
                    #----------------------------------------------------------------------------------------------
                    Final_Ordering_Box_V0['interval_time'] = interval_time
                    Final_Ordering_Box_V0['account_date'] = today
                    Final_Ordering_Box_V0['account_date'] = Final_Ordering_Box_V0['account_date'].\
                        apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
                    #加入判断，如果是尤妮佳，到货时间是3-5天随机
                    Final_Ordering_Box_V0['arrived_date'] = today + timedelta(interval_time)
                    Final_Ordering_Box_V0['arrived_date'] = Final_Ordering_Box_V0['arrived_date'].\
                        apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
                    Final_Ordering_AI = pd.DataFrame(
                        columns=['piece_bar_code', 'account_date','custom_stock_num','custom_business_num',
                                 'custom_terminal_num','interval_time', 'suggestion_qty','purchase_order_qty_AI', 'arrived_date'])
                    Final_Ordering_AI['piece_bar_code'] = Final_Ordering_Box_V0['BARCODE']
                    Final_Ordering_AI['account_date'] = Final_Ordering_Box_V0['account_date']
                    Final_Ordering_AI['interval_time'] = Final_Ordering_Box_V0['interval_time']
                    Final_Ordering_AI['suggestion_qty'] = Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback']
                    Final_Ordering_AI['arrived_date'] = Final_Ordering_Box_V0['arrived_date']
                    Final_Ordering_AI['custom_stock_num'] = custom_stock_num
                    Final_Ordering_AI['custom_business_num'] = custom_business_num
                    Final_Ordering_AI['custom_terminal_num'] =custom_terminal_num
                    Final_Ordering_AI['purchase_order_qty_AI'] = Final_Ordering_AI['suggestion_qty']
                    replenishment = replenishment.append(Final_Ordering_AI)
                elif manufacturer_num == '000323':
                    def Deal_with_Sale_Data():
                        Sale_Data = DataRead_0.Mysql_Data(sale_sql)
                        Sale_Data.columns = ['BJ_BARCODE', 'BJ_TOTAL_QTY', 'BJ_DATE']
                        Sale_Data['BJ_DATE'] = pd.to_datetime(Sale_Data['BJ_DATE']).dt.normalize()
                        Sale_Data_group = Sale_Data.groupby(['BJ_BARCODE', 'BJ_DATE'], as_index=False)
                        Sale_Data_group_sum = Sale_Data_group.agg(sum)
                        ## fill  0 which has no number in that day
                        # the function use for fill the number which frame has no number of other days
                        Min_Date = min(Sale_Data_group_sum['BJ_DATE'])
                        Max_Date = max(Sale_Data_group_sum['BJ_DATE'])
                        Time_Series = pd.DataFrame(pd.date_range(Min_Date, Max_Date), columns=['BJ_DATE'])

                        ## the method for fill the dataframe
                        def fill_frame(df, TS=Time_Series, column='BJ_DATE'):
                            sale_merge = pd.merge(TS, df, on=column, how='left')
                            sale_merge['BJ_TOTAL_QTY'] = sale_merge['BJ_TOTAL_QTY'].fillna(0)
                            try:
                                barcode = list(set(sale_merge['BJ_BARCODE'][sale_merge['BJ_BARCODE'].notna()]))[0]
                            except IndexError:
                                barcode = 'unknow'
                            sale_merge['BJ_BARCODE'] = sale_merge['BJ_BARCODE'].fillna(barcode)
                            return sale_merge

                        ## select the data that use for calculate error between predict data and real data
                        ## use apply methon fullfill the split-apply-combine,group_keys=False can cancel the index of group
                        Sale_Data_Fill = Sale_Data_group_sum.groupby(['BJ_BARCODE'], group_keys=False).apply(fill_frame)

                        ## add week attribute information
                        Sale_Data_Fill['WEEK_DAY'] = DataClean.to_weekday(Sale_Data_Fill['BJ_DATE'])

                        ## select the date in a month
                        Sale_Selected_Date = Sale_Data_Fill[
                            Sale_Data_Fill['BJ_DATE'].isin(DataRead_0.rq_range(start_time_pre, end_time_pre))]
                        ## use the sales data to calculate week probablity and calculate the maximal storage
                        Sale_Data_Week = Sale_Data_Fill[
                            Sale_Data_Fill['BJ_DATE'].isin(DataRead_0.rq_range(start_time, end_time))]
                        Week_agg = Sale_Data_Week.groupby(['BJ_BARCODE', 'WEEK_DAY'], as_index=False).agg(sum)
                        Week_sum = Week_agg.groupby(['BJ_BARCODE'], as_index=False)
                        Week_sum = Week_sum['BJ_TOTAL_QTY'].agg(sum)
                        Week_agg_sum = pd.merge(Week_agg, Week_sum, on='BJ_BARCODE', how='left')
                        Week_agg_sum['PROB'] = Week_agg_sum['BJ_TOTAL_QTY_x'] / (Week_agg_sum['BJ_TOTAL_QTY_y'] + 0.0000001)

                        Sale_Barcode = set(Week_agg_sum['BJ_BARCODE'])
                        return Sale_Selected_Date, Week_agg_sum, Sale_Barcode, Sale_Data_Fill

                    ## deal with the storage of today:Instore_Road_Store
                    def Deal_with_Instorage():
                        Instore_Road_Store = usable_inventory_sql
                        Instore_Road_Store.columns = ['BARCODE', 'CNT_AT', 'PURCHASE_TOTAL_NUM']
                        Instore_Road_Store_Selected = Instore_Road_Store[['BARCODE', 'PURCHASE_TOTAL_NUM']]
                        Instore_Barcode = set(Instore_Road_Store_Selected['BARCODE'])
                        return Instore_Road_Store_Selected, Instore_Barcode

                    ## deal with the predict data
                    def Deal_With_Predict():
                        Pre_Data = DataRead_0.Mysql_Data(predict_sql)
                        Pre_Data = DataClean.missing_value(Pre_Data)
                        Pre_Data = DataClean.to_str(Pre_Data, 'piece_bar_code')
                        Pre_Barcode = set(Pre_Data['piece_bar_code'])
                        return Pre_Data, Pre_Barcode

                    Sale_Selected_Date, Week_agg_sum_group, Sale_Barcode, Sale_Data_Fill = Deal_with_Sale_Data()
                    Instore_Road_Store_Selected, Instore_Barcode = Deal_with_Instorage()
                    Pre_Data, Pre_Barcode = Deal_With_Predict()
                    Unique_Barcode = set(Sale_Barcode & Instore_Barcode & Pre_Barcode)
                    # =========================================================================================================
                    Final_Ordering = pd.DataFrame(
                        columns=["BARCODE", "Usable_Storage", "Max_SS", "Order_or_not", "Order_num"])
                    for barcode in Unique_Barcode:
                        usable_storage = Instore_Road_Store_Selected[Instore_Road_Store_Selected['BARCODE'] == barcode]
                        if usable_storage.empty:
                            usable_inv = 0
                        else:
                            usable_inv = float(usable_storage['PURCHASE_TOTAL_NUM'])
                        # sales data of a sku
                        sales_qty = Sale_Selected_Date[Sale_Selected_Date['BJ_BARCODE'] == barcode]
                        predict_sales = Pre_Data[Pre_Data['piece_bar_code'] == barcode]
                        predict_sales['WEEK_DAY'] = DataClean.to_weekday(predict_sales['cnt_at'])
                        ## deal with predict sales
                        predict_sales_agg = predict_sales.groupby(['cnt_at'], as_index=False).mean()
                        ## get out the week list
                        week_list = Week_agg_sum_group[Week_agg_sum_group['BJ_BARCODE'] == barcode]
                        learn_error = pd.DataFrame({
                            "bj_date": np.array(sales_qty['BJ_DATE']),
                            "real_sales": np.array(sales_qty['BJ_TOTAL_QTY']),
                            "pre_sales": np.array(predict_sales_agg['forecast_qty'][0:len(sales_qty)]),
                            "error": np.array(np.array(sales_qty['BJ_TOTAL_QTY']) - np.array(
                                predict_sales_agg['forecast_qty'][0:len(sales_qty)])),
                            "WEEK_DAY": np.array(DataClean.to_weekday(sales_qty['BJ_DATE']))
                        })
                        learn_error_join = pd.merge(learn_error, week_list, on='WEEK_DAY', how='left')
                        learn_error_join['prob_star'] = learn_error_join['PROB'] / (
                                sum(learn_error_join['PROB']) + 0.000001)
                        learn_error_join['prob_num'] = sum(learn_error_join['error']) * learn_error_join['prob_star']

                        learn_error_join_select = learn_error_join[['prob_num', 'WEEK_DAY']]
                        basic_error_num = learn_error_join_select.drop_duplicates(subset=['WEEK_DAY'], keep='first')
                        predict_sales_num = pd.merge(predict_sales, basic_error_num, on='WEEK_DAY', how="left")

                        ## deal with the predict data,put it as a list type,distinct the circulation
                        pre_list = DataClean.agg_to_frame(predict_sales, t3)
                        error_list = DataClean.agg_to_frame(predict_sales_num, t3)

                        # -------------------------------------------------------------------------------------
                        # 1 prediction sum for t2 days
                        pre_sum = DataClean.Pre_Error_sum_323(pre_list, 'forecast_qty', t2, t3)
                        error_sum = DataClean.Pre_Error_sum_323(error_list, 'prob_num', t2, t3)

                        # ------------------------------------------------------------------------------------

                        Error_sum_days_sigma = DataClean.Err_sigma_V1(error_sum, sigma)

                        Max_SS = np.array(pre_sum) + np.array(Error_sum_days_sigma)

                        # ------------------------------------------------------------------------------------------
                        ## if order or not
                        def Order_num():
                            Available_in_stock = usable_inv
                            if Available_in_stock < Max_SS[len(Max_SS) - 1]:
                                order_or_not = 1
                                Order_num = Max_SS[len(Max_SS) - 1] - Available_in_stock
                            else:
                                order_or_not = 2
                                Order_num = 0
                            return order_or_not, Order_num

                        # --------------------------------------------------------------------------------------------
                        order_or_not, Order_num = Order_num()
                        Order_frame_End = {'BARCODE': barcode,
                                           'Usable_Storage': usable_inv,
                                           'Max_SS': Max_SS[len(Max_SS) - 1],
                                           'Order_or_not': order_or_not,
                                           'Order_num': Order_num}
                        Order_frame_End = pd.DataFrame(Order_frame_End, index=[0])
                        Final_Ordering = pd.concat([Final_Ordering, Order_frame_End])
                        # Final_Ordering =Final_Ordering.append(Final_Ordering)

                    Base_Information = DataRead_0.Mysql_Data(base_information_sql)
                    Base_Information.columns = ['BARCODE', 'CATEGORY_CODE', 'SPACED',
                                                'BRAND_NUM', 'OUT_FACTOR', 'PRODUCT_NATURE', 'STATUS']

                    ## delete the duplicate number
                    Base_Information_Unique = Base_Information

                    ## through the sales of every month,calculate mean of sales number
                    sales_slow_selected = Sale_Data_Fill[
                        Sale_Data_Fill['BJ_DATE'].isin(DataRead_0.rq_range(srq_start, srq_end))]
                    sales_slow_selected_delete_group = sales_slow_selected.groupby(['BJ_BARCODE'], as_index=False)
                    sales_slow_selected_month = sales_slow_selected_delete_group['BJ_TOTAL_QTY'].mean()
                    sales_slow_selected_month['BJ_TOTAL_QTY'] = sales_slow_selected_month['BJ_TOTAL_QTY'] * 30
                    sales_slow_selected_month.columns = ['BJ_BARCODE', 'Month_Sale']
                    ## join the box factor
                    box_factor_frame = Base_Information_Unique[['BARCODE', 'OUT_FACTOR']]
                    slow_sales_month = pd.merge(sales_slow_selected_month, box_factor_frame, how='left', right_on='BARCODE',
                                                left_on='BJ_BARCODE')
                    slow_sales_month['OUT_FACTOR'] = pd.to_numeric(slow_sales_month['OUT_FACTOR'].fillna(1))
                    slow_sales_month['month_box_sale'] = slow_sales_month['Month_Sale'] / slow_sales_month['OUT_FACTOR']
                    slow_sales_month['slow_selling'] = list(
                        map(lambda x: 1 if x <= 1 else 2, slow_sales_month['month_box_sale']))

                    ## 1 mean slow good and 2 mean not slow goods
                    slow_barcode = slow_sales_month[['BJ_BARCODE', 'month_box_sale', 'slow_selling']]

                    ##=================================================================================================
                    ## calculate the usable quota
                    Promotion_Quato = DataRead_0.Mysql_Data(usable_promotion_quota_sql)
                    Promotion_Quato.columns = ['CNT_AT', 'BARCODE', 'QUOTA_QTY']
                    Promotion_Quato['CNT_AT'] = pd.to_datetime(Promotion_Quato['CNT_AT']).dt.normalize()
                    quota_frame = Promotion_Quato.groupby(['BARCODE'], as_index=False).sum()

                    ## though the order frame calculate the usable quota
                    Order_Frame = DataRead_0.Mysql_Data(order_sql)
                    Order_Frame.columns = ['AUDIT_DATE', 'BARCODE', 'QUANTITY']
                    Order_Frame['AUDIT_DATE'] = pd.to_datetime(Order_Frame['AUDIT_DATE']).dt.normalize()
                    Order_Frame_group = Order_Frame.groupby(['BARCODE'], as_index=False).sum()
                    Order_Frame_num = Order_Frame_group[['BARCODE', 'QUANTITY']]

                    quota_ordered_frame = pd.merge(quota_frame, Order_Frame_num, on='BARCODE', how='left')
                    quota_ordered_frame['QUANTITY'] = quota_ordered_frame['QUANTITY'].fillna(0)
                    quota_ordered_frame['USABLE_QUOTY'] = quota_ordered_frame['QUOTA_QTY'] - quota_ordered_frame['QUANTITY']
                    quota_ordered_frame['USABLE_QUOTY'][quota_ordered_frame['USABLE_QUOTY'] <= 0] = 0
                    quota_barcode = quota_ordered_frame.drop_duplicates(subset=['BARCODE'], keep='first')
                    quota_barcode = quota_barcode[['BARCODE', 'USABLE_QUOTY']]
                    ## join the Final_Ordering with the box factor
                    Final_Ordering_Box = pd.merge(Final_Ordering, box_factor_frame, how='left', on='BARCODE')
                    Final_Ordering_Box['OUT_FACTOR'] = pd.to_numeric(Final_Ordering_Box['OUT_FACTOR'].fillna(1))
                    Final_Ordering_Box['ORDER_NUM_BOX'] = Final_Ordering_Box['Order_num'] / Final_Ordering_Box['OUT_FACTOR']
                    Final_Ordering_Box['ORDER_NUM_BOX'] = list(
                        map(lambda x: math.floor(x), Final_Ordering_Box['ORDER_NUM_BOX']))

                    ## join the slow sales factor
                    Final_Ordering_Box_Slow = pd.merge(Final_Ordering_Box, slow_barcode, left_on='BARCODE',
                                                       right_on='BJ_BARCODE',
                                                       how='left')
                    Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'] = Final_Ordering_Box_Slow['ORDER_NUM_BOX']
                    Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'][Final_Ordering_Box_Slow['slow_selling'] == 1] = 0
                    Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'] = Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'].fillna(0)
                    ## join the quota information
                    Final_Ordering_Box_Quota = pd.merge(Final_Ordering_Box_Slow, quota_barcode, on='BARCODE', how='left')
                    Rep_Base = Base_Information_Unique[
                        ['BARCODE', 'CATEGORY_CODE', 'SPACED', 'BRAND_NUM', 'OUT_FACTOR', 'PRODUCT_NATURE', 'STATUS']]
                    Final_Ordering_Box_Quota_Base = pd.merge(Final_Ordering_Box_Quota, Rep_Base, on='BARCODE', how='left')
                    Final_Ordering_Box_Quota_Base['USABLE_QUOTY'] = Final_Ordering_Box_Quota_Base['USABLE_QUOTY'].fillna(0)
                    Final_Ordering_Box_Quota_Base['USABLE_QUOTY'][
                        Final_Ordering_Box_Quota_Base['PRODUCT_NATURE'] == 'Y'] = float(
                        'inf')

                    ## if the order number is bigger than quota,use the quota replace for it
                    Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'] = 0
                    Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'][
                        Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s'] > Final_Ordering_Box_Quota_Base['USABLE_QUOTY']] = \
                        Final_Ordering_Box_Quota_Base['USABLE_QUOTY']
                    Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'][
                        Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s'] <= Final_Ordering_Box_Quota_Base['USABLE_QUOTY']] = \
                        Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s']

                    ## join the product category
                    Final_Ordering_Box_V0 = Final_Ordering_Box_Quota_Base

                    ## join the objective information
                    FeedBack = DataRead_0.Mysql_Data(feedback_sql)
                    FeedBack.columns = ['BARCODE', 'REMARKS']
                    stop_barcode = list(set(FeedBack['BARCODE']))

                    Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback'] = list(
                        map(lambda x: math.floor(x), Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL']))
                    Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback'][
                        Final_Ordering_Box_V0['BARCODE'].isin(stop_barcode)] = 0
                    Final_Ordering_Box_V0['BJ_DATE'] = today
                    Final_Ordering_Box_V0['USABLE_BOX'] = Final_Ordering_Box_V0['Usable_Storage'] / Final_Ordering_Box_V0[
                        'OUT_FACTOR_x']
                    # 这一步是将原来考虑进入的装箱因子给反乘回来
                    Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback'] = Final_Ordering_Box_V0[
                                                                                 'ORDER_NUM_BOX_FINAL_causeback'] * \
                                                                             Final_Ordering_Box['OUT_FACTOR']
                    # 以下是对补货的输出进行修改，来满足仿真的需要输出格式
                    # ----------------------------------------------------------------------------------------------
                    Final_Ordering_Box_V0['interval_time'] = interval_time
                    Final_Ordering_Box_V0['account_date'] = today
                    Final_Ordering_Box_V0['account_date'] = Final_Ordering_Box_V0['account_date']. \
                        apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
                    # 加入判断，如果是尤妮佳，到货时间是3-5天随机
                    Final_Ordering_Box_V0['arrived_date'] = today + timedelta(random.randint(3, 5))
                    Final_Ordering_Box_V0['arrived_date'] = Final_Ordering_Box_V0['arrived_date']. \
                        apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
                    Final_Ordering_AI = pd.DataFrame(
                        columns=['piece_bar_code', 'account_date', 'custom_stock_num', 'custom_business_num',
                                 'custom_terminal_num', 'interval_time', 'suggestion_qty', 'purchase_order_qty_AI',
                                 'arrived_date'])
                    Final_Ordering_AI['piece_bar_code'] = Final_Ordering_Box_V0['BARCODE']
                    Final_Ordering_AI['account_date'] = Final_Ordering_Box_V0['account_date']
                    Final_Ordering_AI['interval_time'] = Final_Ordering_Box_V0['interval_time']
                    Final_Ordering_AI['suggestion_qty'] = Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback']
                    Final_Ordering_AI['arrived_date'] = Final_Ordering_Box_V0['arrived_date']
                    Final_Ordering_AI['custom_stock_num'] = custom_stock_num
                    Final_Ordering_AI['custom_business_num'] = custom_business_num
                    Final_Ordering_AI['custom_terminal_num'] = custom_terminal_num
                    Final_Ordering_AI['purchase_order_qty_AI'] = Final_Ordering_AI['suggestion_qty']

                    replenishment = replenishment.append(Final_Ordering_AI)
                elif manufacturer_num == '000053':
                    def Deal_with_Sale_Data():
                        Sale_Data = DataRead_0.Mysql_Data(sale_sql)
                        Sale_Data.columns = ['BJ_BARCODE', 'BJ_TOTAL_QTY', 'BJ_DATE']
                        Sale_Data['BJ_DATE'] = pd.to_datetime(Sale_Data['BJ_DATE']).dt.normalize()
                        Sale_Data_group = Sale_Data.groupby(['BJ_BARCODE', 'BJ_DATE'], as_index=False)
                        Sale_Data_group_sum = Sale_Data_group.agg(sum)
                        ## fill  0 which has no number in that day
                        # the function use for fill the number which frame has no number of other days
                        Min_Date = min(Sale_Data_group_sum['BJ_DATE'])
                        Max_Date = max(Sale_Data_group_sum['BJ_DATE'])
                        Time_Series = pd.DataFrame(pd.date_range(Min_Date, Max_Date), columns=['BJ_DATE'])

                        ## the method for fill the dataframe
                        def fill_frame(df, TS=Time_Series, column='BJ_DATE'):
                            sale_merge = pd.merge(TS, df, on=column, how='left')
                            sale_merge['BJ_TOTAL_QTY'] = sale_merge['BJ_TOTAL_QTY'].fillna(0)
                            try:
                                barcode = list(set(sale_merge['BJ_BARCODE'][sale_merge['BJ_BARCODE'].notna()]))[0]
                            except IndexError:
                                barcode = 'unknow'
                            sale_merge['BJ_BARCODE'] = sale_merge['BJ_BARCODE'].fillna(barcode)
                            return sale_merge

                        ## select the data that use for calculate error between predict data and real data
                        ## use apply methon fullfill the split-apply-combine,group_keys=False can cancel the index of group
                        Sale_Data_Fill = Sale_Data_group_sum.groupby(['BJ_BARCODE'], group_keys=False).apply(fill_frame)

                        ## add week attribute information
                        Sale_Data_Fill['WEEK_DAY'] = DataClean.to_weekday(Sale_Data_Fill['BJ_DATE'])

                        ## select the date in a month
                        Sale_Selected_Date = Sale_Data_Fill[
                            Sale_Data_Fill['BJ_DATE'].isin(DataRead_0.rq_range(start_time_pre, end_time_pre))]
                        ## use the sales data to calculate week probablity and calculate the maximal storage
                        Sale_Data_Week = Sale_Data_Fill[
                            Sale_Data_Fill['BJ_DATE'].isin(DataRead_0.rq_range(start_time, end_time))]
                        Week_agg = Sale_Data_Week.groupby(['BJ_BARCODE', 'WEEK_DAY'], as_index=False).agg(sum)
                        Week_sum = Week_agg.groupby(['BJ_BARCODE'], as_index=False)
                        Week_sum = Week_sum['BJ_TOTAL_QTY'].agg(sum)
                        Week_agg_sum = pd.merge(Week_agg, Week_sum, on='BJ_BARCODE', how='left')
                        Week_agg_sum['PROB'] = Week_agg_sum['BJ_TOTAL_QTY_x'] / (
                                    Week_agg_sum['BJ_TOTAL_QTY_y'] + 0.0000001)

                        Sale_Barcode = set(Week_agg_sum['BJ_BARCODE'])
                        return Sale_Selected_Date, Week_agg_sum, Sale_Barcode, Sale_Data_Fill

                    ## deal with the storage of today:Instore_Road_Store
                    def Deal_with_Instorage():
                        Instore_Road_Store = usable_inventory_sql
                        Instore_Road_Store.columns = ['BARCODE', 'CNT_AT', 'PURCHASE_TOTAL_NUM']
                        Instore_Road_Store_Selected = Instore_Road_Store[['BARCODE', 'PURCHASE_TOTAL_NUM']]
                        Instore_Barcode = set(Instore_Road_Store_Selected['BARCODE'])
                        return Instore_Road_Store_Selected, Instore_Barcode

                    ## deal with the predict data
                    def Deal_With_Predict():
                        Pre_Data = DataRead_0.Mysql_Data(predict_sql)
                        Pre_Data = DataClean.missing_value(Pre_Data)
                        Pre_Data = DataClean.to_str(Pre_Data, 'piece_bar_code')
                        Pre_Barcode = set(Pre_Data['piece_bar_code'])
                        return Pre_Data, Pre_Barcode

                    Sale_Selected_Date, Week_agg_sum_group, Sale_Barcode, Sale_Data_Fill = Deal_with_Sale_Data()
                    Instore_Road_Store_Selected, Instore_Barcode = Deal_with_Instorage()
                    Pre_Data, Pre_Barcode = Deal_With_Predict()
                    Unique_Barcode = set(Sale_Barcode & Instore_Barcode & Pre_Barcode)
                    # =========================================================================================================
                    Final_Ordering = pd.DataFrame(
                        columns=["BARCODE", "Usable_Storage", "Max_SS", "Order_or_not", "Order_num"])
                    for barcode in Unique_Barcode:
                        usable_storage = Instore_Road_Store_Selected[Instore_Road_Store_Selected['BARCODE'] == barcode]
                        if usable_storage.empty:
                            usable_inv = 0
                        else:
                            usable_inv = float(usable_storage['PURCHASE_TOTAL_NUM'])

                        # sales data of a sku
                        sales_qty = Sale_Selected_Date[Sale_Selected_Date['BJ_BARCODE'] == barcode]
                        predict_sales = Pre_Data[Pre_Data['piece_bar_code'] == barcode]
                        predict_sales['WEEK_DAY'] = DataClean.to_weekday(predict_sales['cnt_at'])

                        ## deal with predict sales
                        predict_sales_agg = predict_sales.groupby(['cnt_at'], as_index=False).mean()

                        ## get out the week list
                        week_list = Week_agg_sum_group[Week_agg_sum_group['BJ_BARCODE'] == barcode]
                        learn_error = pd.DataFrame({
                            "bj_date": np.array(sales_qty['BJ_DATE']),
                            "real_sales": np.array(sales_qty['BJ_TOTAL_QTY']),
                            "pre_sales": np.array(predict_sales_agg['forecast_qty'][0:len(sales_qty)]),
                            "error": np.array(np.array(sales_qty['BJ_TOTAL_QTY']) - np.array(
                                predict_sales_agg['forecast_qty'][0:len(sales_qty)])),
                            "WEEK_DAY": np.array(DataClean.to_weekday(sales_qty['BJ_DATE']))
                        })
                        learn_error_join = pd.merge(learn_error, week_list, on='WEEK_DAY', how='left')
                        learn_error_join['prob_star'] = learn_error_join['PROB'] / (
                                sum(learn_error_join['PROB']) + 0.000001)
                        learn_error_join['prob_num'] = sum(learn_error_join['error']) * learn_error_join['prob_star']

                        learn_error_join_select = learn_error_join[['prob_num', 'WEEK_DAY']]
                        basic_error_num = learn_error_join_select.drop_duplicates(subset=['WEEK_DAY'], keep='first')
                        predict_sales_num = pd.merge(predict_sales, basic_error_num, on='WEEK_DAY', how="left")

                        ## deal with the predict data,put it as a list type,distinct the circulation
                        pre_list = DataClean.agg_to_frame(predict_sales, t3)
                        error_list = DataClean.agg_to_frame(predict_sales_num, t3)

                        # -------------------------------------------------------------------------------------
                        # 1 prediction sum for t2 days
                        pre_sum = DataClean.Pre_Error_sum_053(pre_list, 'forecast_qty', t2, t3)
                        error_sum = DataClean.Pre_Error_sum_053(error_list, 'prob_num', t2, t3)

                        # ------------------------------------------------------------------------------------

                        Error_sum_days_sigma = DataClean.Err_sigma_V1(error_sum, sigma)

                        Max_SS = np.array(pre_sum) + np.array(Error_sum_days_sigma)

                        # ------------------------------------------------------------------------------------------
                        ## if order or not
                        def Order_num():
                            Available_in_stock = usable_inv
                            if Available_in_stock < Max_SS[len(Max_SS) - 1]:
                                order_or_not = 1
                                Order_num = Max_SS[len(Max_SS) - 1] - Available_in_stock
                            else:
                                order_or_not = 2
                                Order_num = 0
                            return order_or_not, Order_num

                        # --------------------------------------------------------------------------------------------
                        order_or_not, Order_num = Order_num()
                        Order_frame_End = {'BARCODE': barcode,
                                           'Usable_Storage': usable_inv,
                                           'Max_SS': Max_SS[len(Max_SS) - 1],
                                           'Order_or_not': order_or_not,
                                           'Order_num': Order_num}
                        Order_frame_End = pd.DataFrame(Order_frame_End, index=[0])
                        Final_Ordering = pd.concat([Final_Ordering, Order_frame_End])
                        # Final_Ordering =Final_Ordering.append(Final_Ordering)

                    Base_Information = DataRead_0.Mysql_Data(base_information_sql)
                    Base_Information.columns = ['BARCODE', 'CATEGORY_CODE', 'SPACED',
                                                'BRAND_NUM', 'OUT_FACTOR', 'PRODUCT_NATURE', 'STATUS']

                    ## delete the duplicate number
                    Base_Information_Unique = Base_Information

                    ## through the sales of every month,calculate mean of sales number
                    sales_slow_selected = Sale_Data_Fill[
                        Sale_Data_Fill['BJ_DATE'].isin(DataRead_0.rq_range(srq_start, srq_end))]
                    sales_slow_selected_delete_group = sales_slow_selected.groupby(['BJ_BARCODE'], as_index=False)
                    sales_slow_selected_month = sales_slow_selected_delete_group['BJ_TOTAL_QTY'].mean()
                    sales_slow_selected_month['BJ_TOTAL_QTY'] = sales_slow_selected_month['BJ_TOTAL_QTY'] * 30
                    sales_slow_selected_month.columns = ['BJ_BARCODE', 'Month_Sale']
                    ## join the box factor
                    box_factor_frame = Base_Information_Unique[['BARCODE', 'OUT_FACTOR']]
                    slow_sales_month = pd.merge(sales_slow_selected_month, box_factor_frame, how='left',
                                                right_on='BARCODE',
                                                left_on='BJ_BARCODE')
                    slow_sales_month['OUT_FACTOR'] = pd.to_numeric(slow_sales_month['OUT_FACTOR'].fillna(1))
                    slow_sales_month['month_box_sale'] = slow_sales_month['Month_Sale'] / slow_sales_month['OUT_FACTOR']
                    slow_sales_month['slow_selling'] = list(
                        map(lambda x: 1 if x <= 1 else 2, slow_sales_month['month_box_sale']))

                    ## 1 mean slow good and 2 mean not slow goods
                    slow_barcode = slow_sales_month[['BJ_BARCODE', 'month_box_sale', 'slow_selling']]

                    ##=================================================================================================
                    ## calculate the usable quota
                    Promotion_Quato = DataRead_0.Mysql_Data(usable_promotion_quota_sql)
                    Promotion_Quato.columns = ['CNT_AT', 'BARCODE', 'QUOTA_QTY']
                    Promotion_Quato['CNT_AT'] = pd.to_datetime(Promotion_Quato['CNT_AT']).dt.normalize()
                    quota_frame = Promotion_Quato.groupby(['BARCODE'], as_index=False).sum()

                    ## though the order frame calculate the usable quota
                    Order_Frame = DataRead_0.Mysql_Data(order_sql)
                    Order_Frame.columns = ['AUDIT_DATE', 'BARCODE', 'QUANTITY']
                    Order_Frame['AUDIT_DATE'] = pd.to_datetime(Order_Frame['AUDIT_DATE']).dt.normalize()
                    Order_Frame_group = Order_Frame.groupby(['BARCODE'], as_index=False).sum()
                    Order_Frame_num = Order_Frame_group[['BARCODE', 'QUANTITY']]

                    quota_ordered_frame = pd.merge(quota_frame, Order_Frame_num, on='BARCODE', how='left')
                    quota_ordered_frame['QUANTITY'] = quota_ordered_frame['QUANTITY'].fillna(0)
                    quota_ordered_frame['USABLE_QUOTY'] = quota_ordered_frame['QUOTA_QTY'] - quota_ordered_frame[
                        'QUANTITY']
                    quota_ordered_frame['USABLE_QUOTY'][quota_ordered_frame['USABLE_QUOTY'] <= 0] = 0
                    quota_barcode = quota_ordered_frame.drop_duplicates(subset=['BARCODE'], keep='first')
                    quota_barcode = quota_barcode[['BARCODE', 'USABLE_QUOTY']]
                    ## join the Final_Ordering with the box factor
                    Final_Ordering_Box = pd.merge(Final_Ordering, box_factor_frame, how='left', on='BARCODE')
                    Final_Ordering_Box['OUT_FACTOR'] = pd.to_numeric(Final_Ordering_Box['OUT_FACTOR'].fillna(1))
                    Final_Ordering_Box['ORDER_NUM_BOX'] = Final_Ordering_Box['Order_num'] / Final_Ordering_Box[
                        'OUT_FACTOR']
                    Final_Ordering_Box['ORDER_NUM_BOX'] = list(
                        map(lambda x: math.floor(x), Final_Ordering_Box['ORDER_NUM_BOX']))

                    ## join the slow sales factor
                    Final_Ordering_Box_Slow = pd.merge(Final_Ordering_Box, slow_barcode, left_on='BARCODE',
                                                       right_on='BJ_BARCODE',
                                                       how='left')
                    Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'] = Final_Ordering_Box_Slow['ORDER_NUM_BOX']
                    Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'][Final_Ordering_Box_Slow['slow_selling'] == 1] = 0
                    Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'] = Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'].fillna(0)
                    ## join the quota information
                    Final_Ordering_Box_Quota = pd.merge(Final_Ordering_Box_Slow, quota_barcode, on='BARCODE',
                                                        how='left')
                    Rep_Base = Base_Information_Unique[
                        ['BARCODE', 'CATEGORY_CODE', 'SPACED', 'BRAND_NUM', 'OUT_FACTOR', 'PRODUCT_NATURE', 'STATUS']]
                    Final_Ordering_Box_Quota_Base = pd.merge(Final_Ordering_Box_Quota, Rep_Base, on='BARCODE',
                                                             how='left')
                    Final_Ordering_Box_Quota_Base['USABLE_QUOTY'] = Final_Ordering_Box_Quota_Base[
                        'USABLE_QUOTY'].fillna(0)
                    Final_Ordering_Box_Quota_Base['USABLE_QUOTY'][
                        Final_Ordering_Box_Quota_Base['PRODUCT_NATURE'] == 'Y'] = float(
                        'inf')

                    ## if the order number is bigger than quota,use the quota replace for it
                    Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'] = 0
                    Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'][
                        Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s'] > Final_Ordering_Box_Quota_Base[
                            'USABLE_QUOTY']] = \
                        Final_Ordering_Box_Quota_Base['USABLE_QUOTY']
                    Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'][
                        Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s'] <= Final_Ordering_Box_Quota_Base[
                            'USABLE_QUOTY']] = \
                        Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s']

                    ## join the product category
                    Final_Ordering_Box_V0 = Final_Ordering_Box_Quota_Base

                    ## join the objective information
                    FeedBack = DataRead_0.Mysql_Data(feedback_sql)
                    FeedBack.columns = ['BARCODE', 'REMARKS']
                    stop_barcode = list(set(FeedBack['BARCODE']))

                    Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback'] = list(
                        map(lambda x: math.floor(x), Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL']))
                    Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback'][
                        Final_Ordering_Box_V0['BARCODE'].isin(stop_barcode)] = 0
                    Final_Ordering_Box_V0['BJ_DATE'] = today
                    Final_Ordering_Box_V0['USABLE_BOX'] = Final_Ordering_Box_V0['Usable_Storage'] / \
                                                          Final_Ordering_Box_V0[
                                                              'OUT_FACTOR_x']
                    # 这一步是将原来考虑进入的装箱因子给反乘回来
                    Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback'] = Final_Ordering_Box_V0[
                                                                                 'ORDER_NUM_BOX_FINAL_causeback'] * \
                                                                             Final_Ordering_Box['OUT_FACTOR']
                    # 以下是对补货的输出进行修改，来满足仿真的需要输出格式
                    # ----------------------------------------------------------------------------------------------
                    Final_Ordering_Box_V0['interval_time'] = interval_time
                    Final_Ordering_Box_V0['account_date'] = today
                    Final_Ordering_Box_V0['account_date'] = Final_Ordering_Box_V0['account_date']. \
                        apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
                    # 加入判断，如果是尤妮佳，到货时间是3-5天随机
                    Final_Ordering_Box_V0['arrived_date'] = today + timedelta(interval_time)
                    Final_Ordering_Box_V0['arrived_date'] = Final_Ordering_Box_V0['arrived_date']. \
                        apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
                    Final_Ordering_AI = pd.DataFrame(
                        columns=['piece_bar_code', 'account_date', 'custom_stock_num', 'custom_business_num',
                                 'custom_terminal_num', 'interval_time', 'suggestion_qty', 'purchase_order_qty_AI',
                                 'arrived_date'])
                    Final_Ordering_AI['piece_bar_code'] = Final_Ordering_Box_V0['BARCODE']
                    Final_Ordering_AI['account_date'] = Final_Ordering_Box_V0['account_date']
                    Final_Ordering_AI['interval_time'] = Final_Ordering_Box_V0['interval_time']
                    Final_Ordering_AI['suggestion_qty'] = Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback']
                    Final_Ordering_AI['arrived_date'] = Final_Ordering_Box_V0['arrived_date']
                    Final_Ordering_AI['custom_stock_num'] = custom_stock_num
                    Final_Ordering_AI['custom_business_num'] = custom_business_num
                    Final_Ordering_AI['custom_terminal_num'] = custom_terminal_num
                    Final_Ordering_AI['purchase_order_qty_AI'] = Final_Ordering_AI['suggestion_qty']
                    replenishment = replenishment.append(Final_Ordering_AI)
                # 设置一个中间的dataframe来 计算第一天的补货行为来计算物权库存,做一个合并不需要有提前期的数值即可
            replenishment_AI = replenishment_AI.append(replenishment)
            #这一步操作是为了能够将补货信息进行合并到仿真数据内，显示在当天下单了多少的数量
            replenishment_AI_first_day = pd.DataFrame(columns=['piece_bar_code', 'custom_stock_num',
                                                               'custom_business_num', 'custom_terminal_num', 'account_date',
                                                               'purchase_order_qty_AI'])
            replenishment_AI_first_day_mid = replenishment.groupby(['piece_bar_code', 'custom_stock_num',
                                                                    'custom_business_num', 'custom_terminal_num',
                                                                    'account_date'
                                                                    ], as_index=False).agg(sum)
            replenishment_AI_first_day['piece_bar_code'] = replenishment_AI_first_day_mid['piece_bar_code']
            replenishment_AI_first_day['custom_stock_num'] = replenishment_AI_first_day_mid['custom_stock_num']
            replenishment_AI_first_day['custom_business_num'] = replenishment_AI_first_day_mid['custom_business_num']
            replenishment_AI_first_day['custom_terminal_num'] = replenishment_AI_first_day_mid['custom_terminal_num']
            replenishment_AI_first_day['account_date'] = replenishment_AI_first_day_mid['account_date']
            replenishment_AI_first_day['purchase_order_qty_AI'] = replenishment_AI_first_day_mid['purchase_order_qty_AI']
            basic_sheet = pd.merge(basic_sheet, replenishment_AI_first_day,
                                   on=['piece_bar_code', 'custom_stock_num',
                                       'custom_business_num', 'custom_terminal_num', 'account_date'], how='left')
            # 将第一天的发生的所有行为记录到对应的final_sheet的表中
            # 这里需要将最原始的数据放入replenishment_AI表中进行用于存储最原始的补货数据
            final_sheet = basic_sheet
        else:
            mid_get_replenishment = replenishment_AI_total[replenishment_AI_total['arrived_date'] == today.strftime('%Y-%m-%d')]
            #这步操作是，在仿真的前几天并不会有到货的数据出现，从而这个dataframe就是空集，这个是需要去除这个空集的存在
            if mid_get_replenishment.empty == True:
                basic_sheet_merge_Temporary = final_sheet.append(basic_sheet)
                basic_sheet_merge_Temporary['suggestion_qty'] = 0
                basic_sheet_merge_Temporary = basic_sheet_merge_Temporary.fillna(0)
            else:
                #先对数据进行分组求和，因为补货到货不分提前期
                mid_get_replenishment_mid = mid_get_replenishment.groupby(['piece_bar_code','custom_business_num',
                                                                           'custom_stock_num'], as_index=False).agg(sum)
                #设置一个中间的到货日期满足的表格，只保留基本维度信息和补货到货的补货量
                mid_replenishment = pd.DataFrame(columns =['piece_bar_code','custom_business_num','custom_stock_num','suggestion_qty'])
                mid_replenishment['piece_bar_code'] = mid_get_replenishment_mid['piece_bar_code']
                mid_replenishment['custom_business_num'] = mid_get_replenishment_mid['custom_business_num']
                mid_replenishment['custom_stock_num'] = mid_get_replenishment_mid['custom_stock_num']
                mid_replenishment['suggestion_qty'] = mid_get_replenishment_mid['suggestion_qty']
                # basic_sheet.to_csv(
                #     'D:/project/P&G/Code/output/evaluation-12/basic_sheet' + str(days) + '.csv', encoding="utf_8_sig")
                basic_sheet = pd.merge(basic_sheet,mid_replenishment,on=['piece_bar_code',
                                                                         'custom_stock_num', 'custom_business_num',
                                                  ], how='left')
                basic_sheet_merge_Temporary = final_sheet.append(basic_sheet)
                basic_sheet_merge_Temporary = basic_sheet_merge_Temporary.fillna(0)
            basic_sheet_merge_Temporary.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/'
                                               'v1---仿真/02_仿真输出/evaluation-18/basic_sheet_merge_Temporary' + str(
                days) + '.csv', encoding="utf_8_sig")

            # basic_sheet_merge_Temporary.to_csv(
            #     'D:/project/P&G/Code/output/evaluation-14/basic_sheet_merge_Temporary' + str(days) + '.csv', encoding="utf_8_sig")
            mid_date = today - timedelta(1)
            mid_date = mid_date.strftime('%Y-%m-%d')
            #使用一个中间的dataframe做计算
            basic_sheet_merge_01_yesterday = basic_sheet_merge_Temporary[basic_sheet_merge_Temporary['account_date'] == mid_date]
            basic_sheet_merge_01_today = basic_sheet_merge_Temporary[basic_sheet_merge_Temporary['account_date'] == today.strftime('%Y-%m-%d')]
            #将前一天的期末库存记为今天的期初库存
            basic_sheet_merge_01_today['start_interest_inv_AI'] = basic_sheet_merge_01_yesterday['end_interest_inv_AI']
            basic_sheet_merge_01_today['start_inv_qty_AI'] = basic_sheet_merge_01_yesterday['end_inv_qty_AI']
            basic_sheet_merge_01_today['shortage_qty_AI'] = 0
            basic_sheet_merge_01_today['delivery_qty_AI'] = 0
            basic_sheet_merge_01_today['shortage_qty_manual'] = 0
            basic_sheet_merge_01_today['storage_remove_AI'] = 0
            basic_sheet_merge_01_today['delivery_qty_AI'] = 0
            basic_sheet_merge_01_today['end_interest_inv_AI'] = 0
            basic_sheet_merge_01_today['end_inv_qty_AI'] = 0
                       #在当天的计算中还没有出现AI给出的订单量，避免后面出现两个列索引，索引在之前先将该索引删除
            basic_sheet_merge_01_today.drop(['purchase_order_qty_AI'], axis=1, inplace=True)
            basic_sheet_merge_01_today.to_csv(
                'D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/v1---仿真/'
                '02_仿真输出/evaluation-18/basic_sheet_merge_01_today' + str(
                    days) + '.csv', encoding="utf_8_sig")
            #这里加入补货模块，从第二天开始，每天的补货行为的库存情况受到仿真环境的库存情况影响
            parameter_condition_sql = """select * from mid_parameter_condition where status = 1 
                                          and custom_start_time <= %s and custom_end_time >= %s """ %(days,days)
            parameter_condition_read = DataRead_0.Mysql_Data(parameter_condition_sql)
            parameter_condition= parameter_condition_read[parameter_condition_read['custom_business_num'] == 3]
            print(len(parameter_condition))
            parameter_condition = parameter_condition.set_index('id')
            end_time = today
            start_time = end_time - timedelta(15 * 28 - 1)
            end_time_pre = today
            start_time_pre = end_time_pre - timedelta(30)
            srq_end = today
            srq_start = srq_end - timedelta(91)
            sale_sql_date = today - timedelta(400)
            start_sale_time = sale_sql_date.strftime('%Y%m%d')
            usable_inventory_time = "date'%s'" % today
            order_date = today - timedelta(today.day)
            order_time = order_date.strftime('%Y%m%d')
            quota_time = days
            print('order_time')
            print(order_time)
            t3 = 35  # for predictive cycle roll time
            sigma = 1.2  # for The uncertainty of prediction
            # 新建一个dataframe用于存储仿真环境中AI的补货量
            replenishment = pd.DataFrame(columns=['piece_bar_code', 'account_date', 'custom_stock_num', 'custom_business_num',
                                      'custom_terminal_num', 'interval_time', 'suggestion_qty',
                                      'purchase_order_qty_AI','arrived_date'])
            #以下操作是在仿真环境中run一次补货算法
            for i in range(len(parameter_condition)):
                # import the parameters of custom_business_num ...
                parameters = parameter_condition.iloc[i, :]
                # global manufacturer_num, custom_business_num, custom_stock_num, custom_terminal_num,t2
                manufacturer_num = parameters['manufacturer_num']
                custom_business_num = parameters['custom_business_num']
                custom_stock_num = parameters['custom_stock_num']
                custom_terminal_num = parameters['custom_terminal_num']
                interval_time = int(parameters['order_delivery_time']) + int(parameters['delivery_arrival_time'])
                mid_parameter_condition_id = parameter_condition.index[i]
                t2 = int(parameters['arrival_time'])
                print(manufacturer_num, custom_business_num, custom_stock_num, custom_terminal_num, t2)
                # Parameters.sql_read(manufacturer_num,custom_business_num,custom_stock_num,custom_terminal_num)
                predict_sql = """select cnt_at,piece_bar_code,forecast_qty from dm_cj_forecast 
                                                 where manufacturer_num = %s
                                                and custom_business_num = %s
                                                 and custom_stock_num = %s
                                                 and custom_terminal_num = %s
                                                 and belonged_date = %s
                                                 """ % (
                manufacturer_num, custom_business_num, custom_stock_num, custom_terminal_num, days)
                sale_sql = """select piece_bar_code,delivery_qty,account_date from mid_cj_sales
                                              where manufacturer_num = %s
                                              and custom_business_num = %s 
                                              and custom_stock_num = %s
                                              and custom_terminal_num = %s
                                              and account_date > %s""" % \
                           (manufacturer_num, custom_business_num, custom_stock_num, custom_terminal_num, start_sale_time)
                usable_inventory_sql_mid = basic_sheet_merge_01_today[
                    basic_sheet_merge_01_today['custom_stock_num'] == custom_stock_num]
                #用这种方式会出现依旧保留原索引的情况，因此需要对库存信息进行索引的重新排序
                usable_inventory_sql = pd.DataFrame(columns=['piece_bar_code', 'account_date', 'available_inv_qty'])
                usable_inventory_sql['piece_bar_code'] = usable_inventory_sql_mid['piece_bar_code']
                usable_inventory_sql['account_date'] = usable_inventory_sql_mid['account_date']
                usable_inventory_sql['available_inv_qty'] = usable_inventory_sql_mid['start_interest_inv_AI']
                #这里进行重置索引的操作
                usable_inventory_sql = usable_inventory_sql.reset_index(drop=True)
                ## the parameters for base information of sku:including brand ,box factor
                base_information_sql = """select piece_bar_code,category_num,spaced,
                                                                  brand_num,case_conversion_rate,segment5,status   
                                                          from mid_cj_goods """
                ## parameters for order time
                order_sql = """select order_date,piece_bar_code,max_uom_qty from mid_cj_order
                                                where manufacturer_num = %s
                                                and custom_business_num = %s 
                                                and custom_stock_num = %s
                                                and order_date>= %s """ % (
                    manufacturer_num, custom_business_num, custom_stock_num, order_time)
                ## the parameters for usable quota of promotional sku
                usable_promotion_quota_sql = """SELECT effective_start_date,piece_bar_code,quota  from mid_cj_quota a 
                                                               where custom_business_num = %s 
                                                               and effective_start_date < %s
                                                               and effective_start_date  > %s""" % (
                custom_business_num, quota_time,quota_time)
                ## the parameters for feedback frame
                feedback_sql = """select piece_bar_code, correct_cause_id from mid_cj_cause
                                where correct_cause_id in (1,2)"""
                ## sum BJ_TOTAL_QTY per day and per BJ_BARCODE
                if manufacturer_num == '000320':
                    def Deal_with_Sale_Data():
                        Sale_Data = DataRead_0.Mysql_Data(sale_sql)
                        Sale_Data.columns = ['BJ_BARCODE', 'BJ_TOTAL_QTY', 'BJ_DATE']
                        Sale_Data['BJ_DATE'] = pd.to_datetime(Sale_Data['BJ_DATE']).dt.normalize()
                        Sale_Data_group = Sale_Data.groupby(['BJ_BARCODE', 'BJ_DATE'], as_index=False)
                        Sale_Data_group_sum = Sale_Data_group.agg(sum)
                        ## fill  0 which has no number in that day
                        # the function use for fill the number which frame has no number of other days
                        Min_Date = min(Sale_Data_group_sum['BJ_DATE'])
                        Max_Date = max(Sale_Data_group_sum['BJ_DATE'])
                        Time_Series = pd.DataFrame(pd.date_range(Min_Date, Max_Date), columns=['BJ_DATE'])

                        ## the method for fill the dataframe
                        def fill_frame(df, TS=Time_Series, column='BJ_DATE'):
                            sale_merge = pd.merge(TS, df, on=column, how='left')
                            sale_merge['BJ_TOTAL_QTY'] = sale_merge['BJ_TOTAL_QTY'].fillna(0)
                            try:
                                barcode = list(set(sale_merge['BJ_BARCODE'][sale_merge['BJ_BARCODE'].notna()]))[0]
                            except IndexError:
                                barcode = 'unknow'
                            sale_merge['BJ_BARCODE'] = sale_merge['BJ_BARCODE'].fillna(barcode)
                            return sale_merge

                        ## select the data that use for calculate error between predict data and real data
                        ## use apply methon fullfill the split-apply-combine,group_keys=False can cancel the index of group
                        Sale_Data_Fill = Sale_Data_group_sum.groupby(['BJ_BARCODE'], group_keys=False).apply(fill_frame)

                        ## add week attribute information
                        Sale_Data_Fill['WEEK_DAY'] = DataClean.to_weekday(Sale_Data_Fill['BJ_DATE'])

                        ## select the date in a month
                        Sale_Selected_Date = Sale_Data_Fill[
                            Sale_Data_Fill['BJ_DATE'].isin(DataRead_0.rq_range(start_time_pre, end_time_pre))]
                        ## use the sales data to calculate week probablity and calculate the maximal storage
                        Sale_Data_Week = Sale_Data_Fill[
                            Sale_Data_Fill['BJ_DATE'].isin(DataRead_0.rq_range(start_time, end_time))]
                        Week_agg = Sale_Data_Week.groupby(['BJ_BARCODE', 'WEEK_DAY'], as_index=False).agg(sum)
                        Week_sum = Week_agg.groupby(['BJ_BARCODE'], as_index=False)
                        Week_sum = Week_sum['BJ_TOTAL_QTY'].agg(sum)
                        Week_agg_sum = pd.merge(Week_agg, Week_sum, on='BJ_BARCODE', how='left')
                        Week_agg_sum['PROB'] = Week_agg_sum['BJ_TOTAL_QTY_x'] / (Week_agg_sum['BJ_TOTAL_QTY_y'] + 0.0000001)

                        Sale_Barcode = set(Week_agg_sum['BJ_BARCODE'])
                        return Sale_Selected_Date, Week_agg_sum, Sale_Barcode, Sale_Data_Fill

                    ## deal with the storage of today:Instore_Road_Store
                    def Deal_with_Instorage():
                        Instore_Road_Store = usable_inventory_sql
                        Instore_Road_Store.columns = ['BARCODE', 'CNT_AT', 'PURCHASE_TOTAL_NUM']
                        Instore_Road_Store_Selected = Instore_Road_Store[['BARCODE', 'PURCHASE_TOTAL_NUM']]
                        Instore_Barcode = set(Instore_Road_Store_Selected['BARCODE'])
                        return Instore_Road_Store_Selected, Instore_Barcode

                    ## deal with the predict data
                    def Deal_With_Predict():
                        Pre_Data = DataRead_0.Mysql_Data(predict_sql)
                        Pre_Data = DataClean.missing_value(Pre_Data)
                        Pre_Data = DataClean.to_str(Pre_Data, 'piece_bar_code')
                        Pre_Barcode = set(Pre_Data['piece_bar_code'])
                        return Pre_Data, Pre_Barcode

                    Sale_Selected_Date, Week_agg_sum_group, Sale_Barcode, Sale_Data_Fill = Deal_with_Sale_Data()
                    Instore_Road_Store_Selected, Instore_Barcode = Deal_with_Instorage()
                    Pre_Data, Pre_Barcode = Deal_With_Predict()
                    Unique_Barcode = set(Sale_Barcode & Instore_Barcode & Pre_Barcode)
                    print('Unique_Barcode.empty')
                    print(len(Unique_Barcode))
                    # =========================================================================================================
                    Final_Ordering = pd.DataFrame(
                        columns=["BARCODE", "Usable_Storage", "Max_SS", "Order_or_not", "Order_num"])
                    for barcode in Unique_Barcode:
                        usable_storage = Instore_Road_Store_Selected[Instore_Road_Store_Selected['BARCODE'] == barcode]
                        # print(usable_storage)
                        if usable_storage.empty:
                            usable_inv = 0
                        else:
                            usable_inv = float(usable_storage['PURCHASE_TOTAL_NUM'])
                        # sales data of a sku
                        sales_qty = Sale_Selected_Date[Sale_Selected_Date['BJ_BARCODE'] == barcode]
                        predict_sales = Pre_Data[Pre_Data['piece_bar_code'] == barcode]
                        predict_sales['WEEK_DAY'] = DataClean.to_weekday(predict_sales['cnt_at'])
                        ## deal with predict sales
                        predict_sales_agg = predict_sales.groupby(['cnt_at'], as_index=False).mean()
                        ## get out the week list
                        week_list = Week_agg_sum_group[Week_agg_sum_group['BJ_BARCODE'] == barcode]
                        learn_error = pd.DataFrame({
                            "bj_date": np.array(sales_qty['BJ_DATE']),
                            "real_sales": np.array(sales_qty['BJ_TOTAL_QTY']),
                            "pre_sales": np.array(predict_sales_agg['forecast_qty'][0:len(sales_qty)]),
                            "error": np.array(np.array(sales_qty['BJ_TOTAL_QTY']) - np.array(
                                predict_sales_agg['forecast_qty'][0:len(sales_qty)])),
                            "WEEK_DAY": np.array(DataClean.to_weekday(sales_qty['BJ_DATE']))
                        })
                        learn_error_join = pd.merge(learn_error, week_list, on='WEEK_DAY', how='left')
                        learn_error_join['prob_star'] = learn_error_join['PROB'] / (
                                    sum(learn_error_join['PROB']) + 0.000001)
                        learn_error_join['prob_num'] = sum(learn_error_join['error']) * learn_error_join['prob_star']

                        learn_error_join_select = learn_error_join[['prob_num', 'WEEK_DAY']]
                        basic_error_num = learn_error_join_select.drop_duplicates(subset=['WEEK_DAY'], keep='first')
                        predict_sales_num = pd.merge(predict_sales, basic_error_num, on='WEEK_DAY', how="left")

                        ## deal with the predict data,put it as a list type,distinct the circulation
                        pre_list = DataClean.agg_to_frame(predict_sales, t3)
                        error_list = DataClean.agg_to_frame(predict_sales_num, t3)

                        # -------------------------------------------------------------------------------------
                        # 1 prediction sum for t2 days
                        pre_sum = DataClean.Pre_Error_sum_320(pre_list, 'forecast_qty', t2, t3)
                        error_sum = DataClean.Pre_Error_sum_320(error_list, 'prob_num', t2, t3)
                        # ------------------------------------------------------------------------------------
                        Error_sum_days_sigma = DataClean.Err_sigma_V1(error_sum, sigma)
                        Max_SS = np.array(pre_sum) + np.array(Error_sum_days_sigma)
                        # ------------------------------------------------------------------------------------------
                        ## if order or not
                        def Order_num():
                            Available_in_stock = usable_inv
                            if Available_in_stock < Max_SS[len(Max_SS) - 1]:
                                order_or_not = 1
                                Order_num = Max_SS[len(Max_SS) - 1] - Available_in_stock

                            else:
                                order_or_not = 2
                                Order_num = 0
                            return order_or_not, Order_num

                        # --------------------------------------------------------------------------------------------
                        order_or_not, Order_num = Order_num()
                        Order_frame_End = {'BARCODE': barcode,
                                           'Usable_Storage': usable_inv,
                                           'Max_SS': Max_SS[len(Max_SS) - 1],
                                           'Order_or_not': order_or_not,
                                           'Order_num': Order_num}
                        Order_frame_End = pd.DataFrame(Order_frame_End, index=[0])
                        Final_Ordering = pd.concat([Final_Ordering, Order_frame_End])
                    Final_Ordering.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/'
                                          'v1---仿真/02_仿真输出/evaluation-18/piece_bar_code/Final_Ordering' + str(t2) + str(
                        days) + '.csv', encoding="utf_8_sig")
                    Final_Ordering =Final_Ordering.append(Final_Ordering)
                    Final_Ordering['Order_num'] = list(
                        map(lambda x: math.floor(x), Final_Ordering['Order_num']))
                    Base_Information = DataRead_0.Mysql_Data(base_information_sql)
                    Base_Information.columns = ['BARCODE', 'CATEGORY_CODE', 'SPACED',
                                                'BRAND_NUM', 'OUT_FACTOR', 'PRODUCT_NATURE', 'STATUS']
                    ## delete the duplicate number
                    Base_Information_Unique = Base_Information
                    ## through the sales of every month,calculate mean of sales number
                    sales_slow_selected = Sale_Data_Fill[
                        Sale_Data_Fill['BJ_DATE'].isin(DataRead_0.rq_range(srq_start, srq_end))]
                    sales_slow_selected_delete_group = sales_slow_selected.groupby(['BJ_BARCODE'], as_index=False)
                    sales_slow_selected_month = sales_slow_selected_delete_group['BJ_TOTAL_QTY'].mean()
                    sales_slow_selected_month['BJ_TOTAL_QTY'] = sales_slow_selected_month['BJ_TOTAL_QTY'] * 30
                    sales_slow_selected_month.columns = ['BJ_BARCODE', 'Month_Sale']
                    ## join the box factor
                    box_factor_frame = Base_Information_Unique[['BARCODE', 'OUT_FACTOR']]
                    slow_sales_month = pd.merge(sales_slow_selected_month, box_factor_frame, how='left', right_on='BARCODE',
                                                left_on='BJ_BARCODE')
                    slow_sales_month['OUT_FACTOR'] = pd.to_numeric(slow_sales_month['OUT_FACTOR'].fillna(1))
                    slow_sales_month['month_box_sale'] = slow_sales_month['Month_Sale'] / slow_sales_month['OUT_FACTOR']
                    slow_sales_month['slow_selling'] = list(
                        map(lambda x: 1 if x <= 1 else 2, slow_sales_month['month_box_sale']))
                    ## 1 mean slow good and 2 mean not slow goods
                    slow_barcode = slow_sales_month[['BJ_BARCODE', 'month_box_sale', 'slow_selling']]
                    ##=================================================================================================
                    ## calculate the usable quota
                    # #这一步是将配额里面的最大数量单位转化为最小数量单位
                    # quota_frame = pd.merge(quota_frame_01,box_factor_frame,on='BARCODE',how='inner')
                    # quota_frame = quota_frame.fillna(1)
                    # quota_frame['QUOTA_QTY'] = quota_frame['QUOTA_QTY_max'] * quota_frame['OUT_FACTOR']
                    ## though the order frame calculate the usable quota
                    Promotion_Quato = DataRead_0.Mysql_Data(usable_promotion_quota_sql)
                    Promotion_Quato.columns = ['CNT_AT', 'BARCODE', 'QUOTA_QTY']
                    Promotion_Quato['CNT_AT'] = pd.to_datetime(Promotion_Quato['CNT_AT']).dt.normalize()
                    quota_frame = Promotion_Quato.groupby(['BARCODE'], as_index=False).sum()

                    ## though the order frame calculate the usable quota
                    Order_Frame = DataRead_0.Mysql_Data(order_sql)
                    Order_Frame.columns = ['AUDIT_DATE', 'BARCODE', 'QUANTITY']
                    Order_Frame['AUDIT_DATE'] = pd.to_datetime(Order_Frame['AUDIT_DATE']).dt.normalize()
                    Order_Frame_group = Order_Frame.groupby(['BARCODE'], as_index=False).sum()
                    Order_Frame_num = Order_Frame_group[['BARCODE', 'QUANTITY']]

                    quota_ordered_frame = pd.merge(quota_frame, Order_Frame_num, on='BARCODE', how='left')
                    quota_ordered_frame['QUANTITY'] = quota_ordered_frame['QUANTITY'].fillna(0)
                    quota_ordered_frame['USABLE_QUOTY'] = quota_ordered_frame['QUOTA_QTY'] - quota_ordered_frame[
                        'QUANTITY']
                    quota_ordered_frame['USABLE_QUOTY'][quota_ordered_frame['USABLE_QUOTY'] <= 0] = 0
                    quota_barcode = quota_ordered_frame.drop_duplicates(subset=['BARCODE'], keep='first')
                    quota_barcode = quota_barcode[['BARCODE', 'USABLE_QUOTY']]
                    ## join the Final_Ordering with the box factor
                    Final_Ordering_Box = pd.merge(Final_Ordering, box_factor_frame, how='left', on='BARCODE')
                    Final_Ordering_Box['OUT_FACTOR'] = pd.to_numeric(Final_Ordering_Box['OUT_FACTOR'].fillna(1))
                    Final_Ordering_Box['ORDER_NUM_BOX'] = Final_Ordering_Box['Order_num'] / Final_Ordering_Box['OUT_FACTOR']
                    Final_Ordering_Box['ORDER_NUM_BOX'] = list(
                        map(lambda x: math.floor(x), Final_Ordering_Box['ORDER_NUM_BOX']))

                    ## join the slow sales factor
                    Final_Ordering_Box_Slow = pd.merge(Final_Ordering_Box, slow_barcode, left_on='BARCODE',
                                                       right_on='BJ_BARCODE',
                                                       how='left')
                    Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'] = Final_Ordering_Box_Slow['ORDER_NUM_BOX']
                    Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'][Final_Ordering_Box_Slow['slow_selling'] == 1] = 0
                    Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'] = Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'].fillna(0)
                    Final_Ordering_Box_Slow.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/'
                                          'v1---仿真/02_仿真输出/evaluation-18/piece_bar_code/Final_Ordering_Box_Slow' + str(t2) + str(
                        days) + '.csv', encoding="utf_8_sig")
                    ## join the quota information
                    Final_Ordering_Box_Quota = pd.merge(Final_Ordering_Box_Slow, quota_barcode, on='BARCODE', how='left')
                    Rep_Base = Base_Information_Unique[
                        ['BARCODE', 'CATEGORY_CODE', 'SPACED', 'BRAND_NUM', 'OUT_FACTOR', 'PRODUCT_NATURE', 'STATUS']]
                    Final_Ordering_Box_Quota_Base = pd.merge(Final_Ordering_Box_Quota, Rep_Base, on='BARCODE', how='left')
                    Final_Ordering_Box_Quota_Base['USABLE_QUOTY'] = Final_Ordering_Box_Quota_Base['USABLE_QUOTY'].fillna(0)
                    Final_Ordering_Box_Quota_Base['USABLE_QUOTY'][
                        Final_Ordering_Box_Quota_Base['PRODUCT_NATURE'] == 'Y'] = float(
                        'inf')

                    ## if the order number is bigger than quota,use the quota replace for it
                    Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'] = 0
                    Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'][
                        Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s'] > Final_Ordering_Box_Quota_Base['USABLE_QUOTY']] = \
                        Final_Ordering_Box_Quota_Base['USABLE_QUOTY']
                    Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'][
                        Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s'] <= Final_Ordering_Box_Quota_Base['USABLE_QUOTY']] = \
                        Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s']

                    ## join the product category
                    Final_Ordering_Box_V0 = Final_Ordering_Box_Quota_Base
                    Final_Ordering_Box_V0.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/'
                                          'v1---仿真/02_仿真输出/evaluation-18/piece_bar_code/Final_Ordering_Box_V0' + str(t2) + str(
                        days) + '.csv', encoding="utf_8_sig")
                    ## join the objective information
                    FeedBack = DataRead_0.Mysql_Data(feedback_sql)
                    FeedBack.columns = ['BARCODE', 'REMARKS']
                    stop_barcode = list(set(FeedBack['BARCODE']))

                    Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback'] = list(
                        map(lambda x: math.floor(x), Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL']))
                    Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback'][
                        Final_Ordering_Box_V0['BARCODE'].isin(stop_barcode)] = 0
                    Final_Ordering_Box_V0['BJ_DATE'] = today

                    Final_Ordering_Box_V0['USABLE_BOX'] = Final_Ordering_Box_V0['Usable_Storage'] / Final_Ordering_Box_V0[
                        'OUT_FACTOR_x']
                    # 这一步是将原来考虑进入的装箱因子给反乘回来
                    Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback'] = Final_Ordering_Box_V0[
                                                                                 'ORDER_NUM_BOX_FINAL_causeback'] * \
                                                                             Final_Ordering_Box['OUT_FACTOR']
                    # 以下是对补货的输出进行修改，来满足仿真的需要输出格式
                    #-------------------------------------------------------------------------------------------------------------------
                    Final_Ordering_Box_V0['interval_time'] = interval_time
                    Final_Ordering_Box_V0['account_date'] = today
                    Final_Ordering_Box_V0['account_date'] = Final_Ordering_Box_V0['account_date'].\
                        apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
                    Final_Ordering_Box_V0['arrived_date'] = today + timedelta(interval_time)
                    Final_Ordering_Box_V0['arrived_date'] = Final_Ordering_Box_V0['arrived_date'].\
                        apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
                    Final_Ordering_AI = pd.DataFrame(
                        columns=['piece_bar_code', 'account_date','custom_stock_num','custom_business_num',
                                 'custom_terminal_num','interval_time', 'suggestion_qty','purchase_order_qty_AI', 'arrived_date'])
                    Final_Ordering_AI['piece_bar_code'] = Final_Ordering_Box_V0['BARCODE']
                    Final_Ordering_AI['account_date'] = Final_Ordering_Box_V0['account_date']
                    Final_Ordering_AI['interval_time'] = Final_Ordering_Box_V0['interval_time']
                    Final_Ordering_AI['suggestion_qty'] = Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback']
                    Final_Ordering_AI['arrived_date'] = Final_Ordering_Box_V0['arrived_date']
                    Final_Ordering_AI['custom_stock_num'] = custom_stock_num
                    Final_Ordering_AI['custom_business_num'] = custom_business_num
                    Final_Ordering_AI['custom_terminal_num'] =custom_terminal_num
                    Final_Ordering_AI['purchase_order_qty_AI'] = Final_Ordering_AI['suggestion_qty']

                    replenishment = replenishment.append(Final_Ordering_AI)
                elif manufacturer_num == '000323':
                    def Deal_with_Sale_Data():
                        Sale_Data = DataRead_0.Mysql_Data(sale_sql)
                        Sale_Data.columns = ['BJ_BARCODE', 'BJ_TOTAL_QTY', 'BJ_DATE']
                        Sale_Data['BJ_DATE'] = pd.to_datetime(Sale_Data['BJ_DATE']).dt.normalize()
                        Sale_Data_group = Sale_Data.groupby(['BJ_BARCODE', 'BJ_DATE'], as_index=False)
                        Sale_Data_group_sum = Sale_Data_group.agg(sum)
                        ## fill  0 which has no number in that day
                        # the function use for fill the number which frame has no number of other days
                        Min_Date = min(Sale_Data_group_sum['BJ_DATE'])
                        Max_Date = max(Sale_Data_group_sum['BJ_DATE'])
                        Time_Series = pd.DataFrame(pd.date_range(Min_Date, Max_Date), columns=['BJ_DATE'])

                        ## the method for fill the dataframe
                        def fill_frame(df, TS=Time_Series, column='BJ_DATE'):
                            sale_merge = pd.merge(TS, df, on=column, how='left')
                            sale_merge['BJ_TOTAL_QTY'] = sale_merge['BJ_TOTAL_QTY'].fillna(0)
                            try:
                                barcode = list(set(sale_merge['BJ_BARCODE'][sale_merge['BJ_BARCODE'].notna()]))[0]
                            except IndexError:
                                barcode = 'unknow'
                            sale_merge['BJ_BARCODE'] = sale_merge['BJ_BARCODE'].fillna(barcode)
                            return sale_merge

                        ## select the data that use for calculate error between predict data and real data
                        ## use apply methon fullfill the split-apply-combine,group_keys=False can cancel the index of group
                        Sale_Data_Fill = Sale_Data_group_sum.groupby(['BJ_BARCODE'], group_keys=False).apply(fill_frame)

                        ## add week attribute information
                        Sale_Data_Fill['WEEK_DAY'] = DataClean.to_weekday(Sale_Data_Fill['BJ_DATE'])

                        ## select the date in a month
                        Sale_Selected_Date = Sale_Data_Fill[
                            Sale_Data_Fill['BJ_DATE'].isin(DataRead_0.rq_range(start_time_pre, end_time_pre))]
                        ## use the sales data to calculate week probablity and calculate the maximal storage
                        Sale_Data_Week = Sale_Data_Fill[
                            Sale_Data_Fill['BJ_DATE'].isin(DataRead_0.rq_range(start_time, end_time))]
                        Week_agg = Sale_Data_Week.groupby(['BJ_BARCODE', 'WEEK_DAY'], as_index=False).agg(sum)
                        Week_sum = Week_agg.groupby(['BJ_BARCODE'], as_index=False)
                        Week_sum = Week_sum['BJ_TOTAL_QTY'].agg(sum)
                        Week_agg_sum = pd.merge(Week_agg, Week_sum, on='BJ_BARCODE', how='left')
                        Week_agg_sum['PROB'] = Week_agg_sum['BJ_TOTAL_QTY_x'] / (
                                    Week_agg_sum['BJ_TOTAL_QTY_y'] + 0.0000001)

                        Sale_Barcode = set(Week_agg_sum['BJ_BARCODE'])
                        return Sale_Selected_Date, Week_agg_sum, Sale_Barcode, Sale_Data_Fill

                    ## deal with the storage of today:Instore_Road_Store
                    def Deal_with_Instorage():
                        Instore_Road_Store = usable_inventory_sql
                        Instore_Road_Store.columns = ['BARCODE', 'CNT_AT', 'PURCHASE_TOTAL_NUM']

                        Instore_Road_Store_Selected = Instore_Road_Store[['BARCODE', 'PURCHASE_TOTAL_NUM']]
                        Instore_Barcode = set(Instore_Road_Store_Selected['BARCODE'])
                        return Instore_Road_Store_Selected, Instore_Barcode

                    ## deal with the predict data
                    def Deal_With_Predict():
                        Pre_Data = DataRead_0.Mysql_Data(predict_sql)
                        Pre_Data = DataClean.missing_value(Pre_Data)
                        Pre_Data = DataClean.to_str(Pre_Data, 'piece_bar_code')
                        Pre_Barcode = set(Pre_Data['piece_bar_code'])
                        return Pre_Data, Pre_Barcode

                    Sale_Selected_Date, Week_agg_sum_group, Sale_Barcode, Sale_Data_Fill = Deal_with_Sale_Data()
                    Instore_Road_Store_Selected, Instore_Barcode = Deal_with_Instorage()
                    Pre_Data, Pre_Barcode = Deal_With_Predict()
                    Unique_Barcode = set(Sale_Barcode & Instore_Barcode & Pre_Barcode)
                    # =========================================================================================================
                    Final_Ordering = pd.DataFrame(
                        columns=["BARCODE", "Usable_Storage", "Max_SS", "Order_or_not", "Order_num"])

                    for barcode in Unique_Barcode:

                        usable_storage = Instore_Road_Store_Selected[Instore_Road_Store_Selected['BARCODE'] == barcode]
                        if usable_storage.empty:
                            usable_inv = 0
                        else:
                            usable_inv = float(usable_storage['PURCHASE_TOTAL_NUM'])

                        # sales data of a sku
                        sales_qty = Sale_Selected_Date[Sale_Selected_Date['BJ_BARCODE'] == barcode]
                        predict_sales = Pre_Data[Pre_Data['piece_bar_code'] == barcode]
                        predict_sales['WEEK_DAY'] = DataClean.to_weekday(predict_sales['cnt_at'])

                        ## deal with predict sales
                        predict_sales_agg = predict_sales.groupby(['cnt_at'], as_index=False).mean()

                        ## get out the week list
                        week_list = Week_agg_sum_group[Week_agg_sum_group['BJ_BARCODE'] == barcode]

                        learn_error = pd.DataFrame({
                            "bj_date": np.array(sales_qty['BJ_DATE']),
                            "real_sales": np.array(sales_qty['BJ_TOTAL_QTY']),
                            "pre_sales": np.array(predict_sales_agg['forecast_qty'][0:len(sales_qty)]),
                            "error": np.array(np.array(sales_qty['BJ_TOTAL_QTY']) - np.array(
                                predict_sales_agg['forecast_qty'][0:len(sales_qty)])),
                            "WEEK_DAY": np.array(DataClean.to_weekday(sales_qty['BJ_DATE']))
                        })

                        learn_error_join = pd.merge(learn_error, week_list, on='WEEK_DAY', how='left')
                        learn_error_join['prob_star'] = learn_error_join['PROB'] / (
                                sum(learn_error_join['PROB']) + 0.000001)
                        learn_error_join['prob_num'] = sum(learn_error_join['error']) * learn_error_join['prob_star']

                        learn_error_join_select = learn_error_join[['prob_num', 'WEEK_DAY']]
                        basic_error_num = learn_error_join_select.drop_duplicates(subset=['WEEK_DAY'], keep='first')
                        predict_sales_num = pd.merge(predict_sales, basic_error_num, on='WEEK_DAY', how="left")

                        ## deal with the predict data,put it as a list type,distinct the circulation
                        pre_list = DataClean.agg_to_frame(predict_sales, t3)
                        error_list = DataClean.agg_to_frame(predict_sales_num, t3)

                        # -------------------------------------------------------------------------------------
                        # 1 prediction sum for t2 days
                        pre_sum = DataClean.Pre_Error_sum_323(pre_list, 'forecast_qty', t2, t3)
                        error_sum = DataClean.Pre_Error_sum_323(error_list, 'prob_num', t2, t3)
                        # ------------------------------------------------------------------------------------
                        Error_sum_days_sigma = DataClean.Err_sigma_V1(error_sum, sigma)
                        Max_SS = np.array(pre_sum) + np.array(Error_sum_days_sigma)
                        # ------------------------------------------------------------------------------------------
                        ## if order or not
                        def Order_num():
                            Available_in_stock = usable_inv
                            if Available_in_stock < Max_SS[len(Max_SS) - 1]:
                                order_or_not = 1
                                Order_num = Max_SS[len(Max_SS) - 1] - Available_in_stock

                            else:
                                order_or_not = 2
                                Order_num = 0
                            return order_or_not, Order_num

                        # --------------------------------------------------------------------------------------------
                        order_or_not, Order_num = Order_num()
                        Order_frame_End = {'BARCODE': barcode,
                                           'Usable_Storage': usable_inv,
                                           'Max_SS': Max_SS[len(Max_SS) - 1],
                                           'Order_or_not': order_or_not,
                                           'Order_num': Order_num}
                        Order_frame_End = pd.DataFrame(Order_frame_End, index=[0])
                        Final_Ordering = pd.concat([Final_Ordering, Order_frame_End])
                    # Final_Ordering.to_csv('D:/project/P&G/Code/output/evaluation-16/piece_bar_code/'+str(i)+str(days)+
                    #                           '.csv', encoding="utf_8_sig")
                    Final_Ordering = Final_Ordering.append(Final_Ordering)
                    Final_Ordering['Order_num'] = list(
                        map(lambda x: math.floor(x), Final_Ordering['Order_num']))
                    Base_Information = DataRead_0.Mysql_Data(base_information_sql)
                    Base_Information.columns = ['BARCODE', 'CATEGORY_CODE', 'SPACED',
                                                'BRAND_NUM', 'OUT_FACTOR', 'PRODUCT_NATURE', 'STATUS']
                    ## delete the duplicate number
                    Base_Information_Unique = Base_Information
                    ## through the sales of every month,calculate mean of sales number
                    sales_slow_selected = Sale_Data_Fill[
                        Sale_Data_Fill['BJ_DATE'].isin(DataRead_0.rq_range(srq_start, srq_end))]
                    sales_slow_selected_delete_group = sales_slow_selected.groupby(['BJ_BARCODE'], as_index=False)
                    sales_slow_selected_month = sales_slow_selected_delete_group['BJ_TOTAL_QTY'].mean()
                    sales_slow_selected_month['BJ_TOTAL_QTY'] = sales_slow_selected_month['BJ_TOTAL_QTY'] * 30
                    sales_slow_selected_month.columns = ['BJ_BARCODE', 'Month_Sale']
                    ## join the box factor
                    box_factor_frame = Base_Information_Unique[['BARCODE', 'OUT_FACTOR']]
                    slow_sales_month = pd.merge(sales_slow_selected_month, box_factor_frame, how='left',
                                                right_on='BARCODE',
                                                left_on='BJ_BARCODE')
                    slow_sales_month['OUT_FACTOR'] = pd.to_numeric(slow_sales_month['OUT_FACTOR'].fillna(1))
                    slow_sales_month['month_box_sale'] = slow_sales_month['Month_Sale'] / slow_sales_month['OUT_FACTOR']
                    slow_sales_month['slow_selling'] = list(
                        map(lambda x: 1 if x <= 1 else 2, slow_sales_month['month_box_sale']))
                    ## 1 mean slow good and 2 mean not slow goods
                    slow_barcode = slow_sales_month[['BJ_BARCODE', 'month_box_sale', 'slow_selling']]
                    ##=================================================================================================
                    ## calculate the usable quota
                    Promotion_Quato = DataRead_0.Mysql_Data(usable_promotion_quota_sql)
                    Promotion_Quato.columns = ['CNT_AT', 'BARCODE', 'QUOTA_QTY']
                    Promotion_Quato['CNT_AT'] = pd.to_datetime(Promotion_Quato['CNT_AT']).dt.normalize()
                    quota_frame = Promotion_Quato.groupby(['BARCODE'], as_index=False).sum()
                    ## though the order frame calculate the usable quota
                    Order_Frame = DataRead_0.Mysql_Data(order_sql)
                    Order_Frame.columns = ['AUDIT_DATE', 'BARCODE', 'QUANTITY']
                    Order_Frame['AUDIT_DATE'] = pd.to_datetime(Order_Frame['AUDIT_DATE']).dt.normalize()
                    Order_Frame_group = Order_Frame.groupby(['BARCODE'], as_index=False).sum()
                    Order_Frame_num = Order_Frame_group[['BARCODE', 'QUANTITY']]

                    quota_ordered_frame = pd.merge(quota_frame, Order_Frame_num, on='BARCODE', how='left')
                    quota_ordered_frame['QUANTITY'] = quota_ordered_frame['QUANTITY'].fillna(0)
                    quota_ordered_frame['USABLE_QUOTY'] = quota_ordered_frame['QUOTA_QTY'] - quota_ordered_frame[
                        'QUANTITY']
                    quota_ordered_frame['USABLE_QUOTY'][quota_ordered_frame['USABLE_QUOTY'] <= 0] = 0
                    quota_barcode = quota_ordered_frame.drop_duplicates(subset=['BARCODE'], keep='first')
                    quota_barcode = quota_barcode[['BARCODE', 'USABLE_QUOTY']]
                    ## join the Final_Ordering with the box factor
                    Final_Ordering_Box = pd.merge(Final_Ordering, box_factor_frame, how='left', on='BARCODE')
                    Final_Ordering_Box['OUT_FACTOR'] = pd.to_numeric(Final_Ordering_Box['OUT_FACTOR'].fillna(1))
                    Final_Ordering_Box['ORDER_NUM_BOX'] = Final_Ordering_Box['Order_num'] / Final_Ordering_Box[
                        'OUT_FACTOR']
                    Final_Ordering_Box['ORDER_NUM_BOX'] = list(
                        map(lambda x: math.floor(x), Final_Ordering_Box['ORDER_NUM_BOX']))

                    ## join the slow sales factor
                    Final_Ordering_Box_Slow = pd.merge(Final_Ordering_Box, slow_barcode, left_on='BARCODE',
                                                       right_on='BJ_BARCODE',
                                                       how='left')
                    Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'] = Final_Ordering_Box_Slow['ORDER_NUM_BOX']
                    Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'][Final_Ordering_Box_Slow['slow_selling'] == 1] = 0
                    Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'] = Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'].fillna(0)
                    ## join the quota information
                    Final_Ordering_Box_Quota = pd.merge(Final_Ordering_Box_Slow, quota_barcode, on='BARCODE',
                                                        how='left')
                    Rep_Base = Base_Information_Unique[
                        ['BARCODE', 'CATEGORY_CODE', 'SPACED', 'BRAND_NUM', 'OUT_FACTOR', 'PRODUCT_NATURE', 'STATUS']]
                    Final_Ordering_Box_Quota_Base = pd.merge(Final_Ordering_Box_Quota, Rep_Base, on='BARCODE',
                                                             how='left')
                    Final_Ordering_Box_Quota_Base['USABLE_QUOTY'] = Final_Ordering_Box_Quota_Base[
                        'USABLE_QUOTY'].fillna(0)
                    Final_Ordering_Box_Quota_Base['USABLE_QUOTY'][
                        Final_Ordering_Box_Quota_Base['PRODUCT_NATURE'] == 'Y'] = float(
                        'inf')

                    ## if the order number is bigger than quota,use the quota replace for it
                    Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'] = 0
                    Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'][
                        Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s'] > Final_Ordering_Box_Quota_Base[
                            'USABLE_QUOTY']] = \
                        Final_Ordering_Box_Quota_Base['USABLE_QUOTY']
                    Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'][
                        Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s'] <= Final_Ordering_Box_Quota_Base[
                            'USABLE_QUOTY']] = \
                        Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s']

                    ## join the product category
                    Final_Ordering_Box_V0 = Final_Ordering_Box_Quota_Base
                    Final_Ordering_Box_V0.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/'
                                          'v1---仿真/02_仿真输出/evaluation-17/piece_bar_code/Final_Ordering_Box_V0' + str(t2) + str(
                        days) + '.csv', encoding="utf_8_sig")
                    ## join the objective information
                    FeedBack = DataRead_0.Mysql_Data(feedback_sql)
                    FeedBack.columns = ['BARCODE', 'REMARKS']
                    stop_barcode = list(set(FeedBack['BARCODE']))

                    Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback'] = list(
                        map(lambda x: math.floor(x), Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL']))
                    Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback'][
                        Final_Ordering_Box_V0['BARCODE'].isin(stop_barcode)] = 0
                    Final_Ordering_Box_V0['BJ_DATE'] = today

                    Final_Ordering_Box_V0['USABLE_BOX'] = Final_Ordering_Box_V0['Usable_Storage'] / \
                                                          Final_Ordering_Box_V0[
                                                              'OUT_FACTOR_x']
                    # 这一步是将原来考虑进入的装箱因子给反乘回来
                    Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback'] = Final_Ordering_Box_V0[
                                                                                 'ORDER_NUM_BOX_FINAL_causeback'] * \
                                                                             Final_Ordering_Box['OUT_FACTOR']
                    # 以下是对补货的输出进行修改，来满足仿真的需要输出格式
                    # -------------------------------------------------------------------------------------------------------------------
                    Final_Ordering_Box_V0['interval_time'] = interval_time
                    Final_Ordering_Box_V0['account_date'] = today
                    Final_Ordering_Box_V0['account_date'] = Final_Ordering_Box_V0['account_date']. \
                        apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
                    # 加入判断，如果是尤妮佳，到货时间是3-5天随机
                    Final_Ordering_Box_V0['arrived_date'] = today + timedelta(random.randint(3, 5))
                    Final_Ordering_Box_V0['arrived_date'] = Final_Ordering_Box_V0['arrived_date']. \
                        apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
                    Final_Ordering_AI = pd.DataFrame(
                        columns=['piece_bar_code', 'account_date', 'custom_stock_num', 'custom_business_num',
                                 'custom_terminal_num', 'interval_time', 'suggestion_qty', 'purchase_order_qty_AI',
                                 'arrived_date'])
                    Final_Ordering_AI['piece_bar_code'] = Final_Ordering_Box_V0['BARCODE']
                    Final_Ordering_AI['account_date'] = Final_Ordering_Box_V0['account_date']
                    Final_Ordering_AI['interval_time'] = Final_Ordering_Box_V0['interval_time']
                    Final_Ordering_AI['suggestion_qty'] = Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback']
                    Final_Ordering_AI['arrived_date'] = Final_Ordering_Box_V0['arrived_date']
                    Final_Ordering_AI['custom_stock_num'] = custom_stock_num
                    Final_Ordering_AI['custom_business_num'] = custom_business_num
                    Final_Ordering_AI['custom_terminal_num'] = custom_terminal_num
                    Final_Ordering_AI['purchase_order_qty_AI'] = Final_Ordering_AI['suggestion_qty']
                    replenishment = replenishment.append(Final_Ordering_AI)
                elif manufacturer_num == '000053':
                    def Deal_with_Sale_Data():
                        Sale_Data = DataRead_0.Mysql_Data(sale_sql)
                        Sale_Data.columns = ['BJ_BARCODE', 'BJ_TOTAL_QTY', 'BJ_DATE']
                        Sale_Data['BJ_DATE'] = pd.to_datetime(Sale_Data['BJ_DATE']).dt.normalize()
                        Sale_Data_group = Sale_Data.groupby(['BJ_BARCODE', 'BJ_DATE'], as_index=False)
                        Sale_Data_group_sum = Sale_Data_group.agg(sum)
                        ## fill  0 which has no number in that day
                        # the function use for fill the number which frame has no number of other days
                        Min_Date = min(Sale_Data_group_sum['BJ_DATE'])
                        Max_Date = max(Sale_Data_group_sum['BJ_DATE'])
                        Time_Series = pd.DataFrame(pd.date_range(Min_Date, Max_Date), columns=['BJ_DATE'])

                        ## the method for fill the dataframe
                        def fill_frame(df, TS=Time_Series, column='BJ_DATE'):
                            sale_merge = pd.merge(TS, df, on=column, how='left')
                            sale_merge['BJ_TOTAL_QTY'] = sale_merge['BJ_TOTAL_QTY'].fillna(0)
                            try:
                                barcode = list(set(sale_merge['BJ_BARCODE'][sale_merge['BJ_BARCODE'].notna()]))[0]
                            except IndexError:
                                barcode = 'unknow'
                            sale_merge['BJ_BARCODE'] = sale_merge['BJ_BARCODE'].fillna(barcode)
                            return sale_merge

                        ## select the data that use for calculate error between predict data and real data
                        ## use apply methon fullfill the split-apply-combine,group_keys=False can cancel the index of group
                        Sale_Data_Fill = Sale_Data_group_sum.groupby(['BJ_BARCODE'], group_keys=False).apply(fill_frame)

                        ## add week attribute information
                        Sale_Data_Fill['WEEK_DAY'] = DataClean.to_weekday(Sale_Data_Fill['BJ_DATE'])

                        ## select the date in a month
                        Sale_Selected_Date = Sale_Data_Fill[
                            Sale_Data_Fill['BJ_DATE'].isin(DataRead_0.rq_range(start_time_pre, end_time_pre))]
                        ## use the sales data to calculate week probablity and calculate the maximal storage
                        Sale_Data_Week = Sale_Data_Fill[
                            Sale_Data_Fill['BJ_DATE'].isin(DataRead_0.rq_range(start_time, end_time))]
                        Week_agg = Sale_Data_Week.groupby(['BJ_BARCODE', 'WEEK_DAY'], as_index=False).agg(sum)
                        Week_sum = Week_agg.groupby(['BJ_BARCODE'], as_index=False)
                        Week_sum = Week_sum['BJ_TOTAL_QTY'].agg(sum)
                        Week_agg_sum = pd.merge(Week_agg, Week_sum, on='BJ_BARCODE', how='left')
                        Week_agg_sum['PROB'] = Week_agg_sum['BJ_TOTAL_QTY_x'] / (
                                    Week_agg_sum['BJ_TOTAL_QTY_y'] + 0.0000001)

                        Sale_Barcode = set(Week_agg_sum['BJ_BARCODE'])
                        return Sale_Selected_Date, Week_agg_sum, Sale_Barcode, Sale_Data_Fill

                    ## deal with the storage of today:Instore_Road_Store
                    def Deal_with_Instorage():
                        Instore_Road_Store = usable_inventory_sql
                        Instore_Road_Store.columns = ['BARCODE', 'CNT_AT', 'PURCHASE_TOTAL_NUM']

                        Instore_Road_Store_Selected = Instore_Road_Store[['BARCODE', 'PURCHASE_TOTAL_NUM']]
                        Instore_Barcode = set(Instore_Road_Store_Selected['BARCODE'])
                        return Instore_Road_Store_Selected, Instore_Barcode

                    ## deal with the predict data
                    def Deal_With_Predict():
                        Pre_Data = DataRead_0.Mysql_Data(predict_sql)
                        Pre_Data = DataClean.missing_value(Pre_Data)
                        Pre_Data = DataClean.to_str(Pre_Data, 'piece_bar_code')
                        Pre_Barcode = set(Pre_Data['piece_bar_code'])
                        return Pre_Data, Pre_Barcode

                    Sale_Selected_Date, Week_agg_sum_group, Sale_Barcode, Sale_Data_Fill = Deal_with_Sale_Data()
                    Instore_Road_Store_Selected, Instore_Barcode = Deal_with_Instorage()
                    Pre_Data, Pre_Barcode = Deal_With_Predict()
                    Unique_Barcode = set(Sale_Barcode & Instore_Barcode & Pre_Barcode)
                    # =========================================================================================================
                    Final_Ordering = pd.DataFrame(
                        columns=["BARCODE", "Usable_Storage", "Max_SS", "Order_or_not", "Order_num"])

                    for barcode in Unique_Barcode:
                        usable_storage = Instore_Road_Store_Selected[Instore_Road_Store_Selected['BARCODE'] == barcode]
                        if usable_storage.empty:
                            usable_inv = 0
                        else:
                            usable_inv = float(usable_storage['PURCHASE_TOTAL_NUM'])

                        # sales data of a sku
                        sales_qty = Sale_Selected_Date[Sale_Selected_Date['BJ_BARCODE'] == barcode]
                        predict_sales = Pre_Data[Pre_Data['piece_bar_code'] == barcode]
                        predict_sales['WEEK_DAY'] = DataClean.to_weekday(predict_sales['cnt_at'])

                        ## deal with predict sales
                        predict_sales_agg = predict_sales.groupby(['cnt_at'], as_index=False).mean()

                        ## get out the week list
                        week_list = Week_agg_sum_group[Week_agg_sum_group['BJ_BARCODE'] == barcode]

                        learn_error = pd.DataFrame({
                            "bj_date": np.array(sales_qty['BJ_DATE']),
                            "real_sales": np.array(sales_qty['BJ_TOTAL_QTY']),
                            "pre_sales": np.array(predict_sales_agg['forecast_qty'][0:len(sales_qty)]),
                            "error": np.array(np.array(sales_qty['BJ_TOTAL_QTY']) - np.array(
                                predict_sales_agg['forecast_qty'][0:len(sales_qty)])),
                            "WEEK_DAY": np.array(DataClean.to_weekday(sales_qty['BJ_DATE']))
                        })

                        learn_error_join = pd.merge(learn_error, week_list, on='WEEK_DAY', how='left')
                        learn_error_join['prob_star'] = learn_error_join['PROB'] / (
                                sum(learn_error_join['PROB']) + 0.000001)
                        learn_error_join['prob_num'] = sum(learn_error_join['error']) * learn_error_join['prob_star']

                        learn_error_join_select = learn_error_join[['prob_num', 'WEEK_DAY']]
                        basic_error_num = learn_error_join_select.drop_duplicates(subset=['WEEK_DAY'], keep='first')
                        predict_sales_num = pd.merge(predict_sales, basic_error_num, on='WEEK_DAY', how="left")

                        ## deal with the predict data,put it as a list type,distinct the circulation
                        pre_list = DataClean.agg_to_frame(predict_sales, t3)
                        error_list = DataClean.agg_to_frame(predict_sales_num, t3)

                        # -------------------------------------------------------------------------------------
                        # 1 prediction sum for t2 days
                        pre_sum = DataClean.pre_Error_sum_053(pre_list, 'forecast_qty', t2, t3)
                        error_sum = DataClean.pre_Error_sum_053(error_list, 'prob_num', t2, t3)
                        # ------------------------------------------------------------------------------------
                        Error_sum_days_sigma = DataClean.Err_sigma_V1(error_sum, sigma)
                        Max_SS = np.array(pre_sum) + np.array(Error_sum_days_sigma)

                        # ------------------------------------------------------------------------------------------
                        ## if order or not
                        def Order_num():
                            Available_in_stock = usable_inv
                            if Available_in_stock < Max_SS[len(Max_SS) - 1]:
                                order_or_not = 1
                                Order_num = Max_SS[len(Max_SS) - 1] - Available_in_stock

                            else:
                                order_or_not = 2
                                Order_num = 0
                            return order_or_not, Order_num

                        # --------------------------------------------------------------------------------------------
                        order_or_not, Order_num = Order_num()
                        Order_frame_End = {'BARCODE': barcode,
                                           'Usable_Storage': usable_inv,
                                           'Max_SS': Max_SS[len(Max_SS) - 1],
                                           'Order_or_not': order_or_not,
                                           'Order_num': Order_num}
                        Order_frame_End = pd.DataFrame(Order_frame_End, index=[0])
                        Final_Ordering = pd.concat([Final_Ordering, Order_frame_End])
                    # Final_Ordering.to_csv('D:/project/P&G/Code/output/evaluation-16/piece_bar_code/'+str(i)+str(days)+
                    #                           '.csv', encoding="utf_8_sig")
                    Final_Ordering = Final_Ordering.append(Final_Ordering)
                    Final_Ordering['Order_num'] = list(
                        map(lambda x: math.floor(x), Final_Ordering['Order_num']))
                    Base_Information = DataRead_0.Mysql_Data(base_information_sql)
                    Base_Information.columns = ['BARCODE', 'CATEGORY_CODE', 'SPACED',
                                                'BRAND_NUM', 'OUT_FACTOR', 'PRODUCT_NATURE', 'STATUS']
                    ## delete the duplicate number
                    Base_Information_Unique = Base_Information
                    ## through the sales of every month,calculate mean of sales number
                    sales_slow_selected = Sale_Data_Fill[
                        Sale_Data_Fill['BJ_DATE'].isin(DataRead_0.rq_range(srq_start, srq_end))]
                    sales_slow_selected_delete_group = sales_slow_selected.groupby(['BJ_BARCODE'], as_index=False)
                    sales_slow_selected_month = sales_slow_selected_delete_group['BJ_TOTAL_QTY'].mean()
                    sales_slow_selected_month['BJ_TOTAL_QTY'] = sales_slow_selected_month['BJ_TOTAL_QTY'] * 30
                    sales_slow_selected_month.columns = ['BJ_BARCODE', 'Month_Sale']
                    ## join the box factor
                    box_factor_frame = Base_Information_Unique[['BARCODE', 'OUT_FACTOR']]
                    slow_sales_month = pd.merge(sales_slow_selected_month, box_factor_frame, how='left',
                                                right_on='BARCODE',
                                                left_on='BJ_BARCODE')
                    slow_sales_month['OUT_FACTOR'] = pd.to_numeric(slow_sales_month['OUT_FACTOR'].fillna(1))
                    slow_sales_month['month_box_sale'] = slow_sales_month['Month_Sale'] / slow_sales_month['OUT_FACTOR']
                    slow_sales_month['slow_selling'] = list(
                        map(lambda x: 1 if x <= 1 else 2, slow_sales_month['month_box_sale']))
                    ## 1 mean slow good and 2 mean not slow goods
                    slow_barcode = slow_sales_month[['BJ_BARCODE', 'month_box_sale', 'slow_selling']]
                    ##=================================================================================================
                    ## calculate the usable quota
                    Promotion_Quato = DataRead_0.Mysql_Data(usable_promotion_quota_sql)
                    Promotion_Quato.columns = ['CNT_AT', 'BARCODE', 'QUOTA_QTY']
                    Promotion_Quato['CNT_AT'] = pd.to_datetime(Promotion_Quato['CNT_AT']).dt.normalize()
                    quota_frame = Promotion_Quato.groupby(['BARCODE'], as_index=False).sum()
                    ## though the order frame calculate the usable quota
                    Order_Frame = DataRead_0.Mysql_Data(order_sql)
                    Order_Frame.columns = ['AUDIT_DATE', 'BARCODE', 'QUANTITY']
                    Order_Frame['AUDIT_DATE'] = pd.to_datetime(Order_Frame['AUDIT_DATE']).dt.normalize()
                    Order_Frame_group = Order_Frame.groupby(['BARCODE'], as_index=False).sum()
                    Order_Frame_num = Order_Frame_group[['BARCODE', 'QUANTITY']]

                    quota_ordered_frame = pd.merge(quota_frame, Order_Frame_num, on='BARCODE', how='left')
                    quota_ordered_frame['QUANTITY'] = quota_ordered_frame['QUANTITY'].fillna(0)
                    quota_ordered_frame['USABLE_QUOTY'] = quota_ordered_frame['QUOTA_QTY'] - quota_ordered_frame[
                        'QUANTITY']
                    quota_ordered_frame['USABLE_QUOTY'][quota_ordered_frame['USABLE_QUOTY'] <= 0] = 0
                    quota_barcode = quota_ordered_frame.drop_duplicates(subset=['BARCODE'], keep='first')
                    quota_barcode = quota_barcode[['BARCODE', 'USABLE_QUOTY']]
                    ## join the Final_Ordering with the box factor
                    Final_Ordering_Box = pd.merge(Final_Ordering, box_factor_frame, how='left', on='BARCODE')
                    Final_Ordering_Box['OUT_FACTOR'] = pd.to_numeric(Final_Ordering_Box['OUT_FACTOR'].fillna(1))
                    Final_Ordering_Box['ORDER_NUM_BOX'] = Final_Ordering_Box['Order_num'] / Final_Ordering_Box[
                        'OUT_FACTOR']
                    Final_Ordering_Box['ORDER_NUM_BOX'] = list(
                        map(lambda x: math.floor(x), Final_Ordering_Box['ORDER_NUM_BOX']))

                    ## join the slow sales factor
                    Final_Ordering_Box_Slow = pd.merge(Final_Ordering_Box, slow_barcode, left_on='BARCODE',
                                                       right_on='BJ_BARCODE',
                                                       how='left')
                    Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'] = Final_Ordering_Box_Slow['ORDER_NUM_BOX']
                    Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'][Final_Ordering_Box_Slow['slow_selling'] == 1] = 0
                    Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'] = Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'].fillna(0)
                    ## join the quota information
                    Final_Ordering_Box_Quota = pd.merge(Final_Ordering_Box_Slow, quota_barcode, on='BARCODE',
                                                        how='left')
                    Rep_Base = Base_Information_Unique[
                        ['BARCODE', 'CATEGORY_CODE', 'SPACED', 'BRAND_NUM', 'OUT_FACTOR', 'PRODUCT_NATURE', 'STATUS']]
                    Final_Ordering_Box_Quota_Base = pd.merge(Final_Ordering_Box_Quota, Rep_Base, on='BARCODE',
                                                             how='left')
                    Final_Ordering_Box_Quota_Base['USABLE_QUOTY'] = Final_Ordering_Box_Quota_Base[
                        'USABLE_QUOTY'].fillna(0)
                    Final_Ordering_Box_Quota_Base['USABLE_QUOTY'][
                        Final_Ordering_Box_Quota_Base['PRODUCT_NATURE'] == 'Y'] = float(
                        'inf')

                    ## if the order number is bigger than quota,use the quota replace for it
                    Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'] = 0
                    Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'][
                        Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s'] > Final_Ordering_Box_Quota_Base[
                            'USABLE_QUOTY']] = \
                        Final_Ordering_Box_Quota_Base['USABLE_QUOTY']
                    Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'][
                        Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s'] <= Final_Ordering_Box_Quota_Base[
                            'USABLE_QUOTY']] = \
                        Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s']

                    ## join the product category
                    Final_Ordering_Box_V0 = Final_Ordering_Box_Quota_Base

                    ## join the objective information
                    FeedBack = DataRead_0.Mysql_Data(feedback_sql)
                    FeedBack.columns = ['BARCODE', 'REMARKS']
                    stop_barcode = list(set(FeedBack['BARCODE']))

                    Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback'] = list(
                        map(lambda x: math.floor(x), Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL']))
                    Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback'][
                        Final_Ordering_Box_V0['BARCODE'].isin(stop_barcode)] = 0
                    Final_Ordering_Box_V0['BJ_DATE'] = today

                    Final_Ordering_Box_V0['USABLE_BOX'] = Final_Ordering_Box_V0['Usable_Storage'] / \
                                                          Final_Ordering_Box_V0[
                                                              'OUT_FACTOR_x']
                    # 这一步是将原来考虑进入的装箱因子给反乘回来
                    Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback'] = Final_Ordering_Box_V0[
                                                                                 'ORDER_NUM_BOX_FINAL_causeback'] * \
                                                                             Final_Ordering_Box['OUT_FACTOR']
                    # 以下是对补货的输出进行修改，来满足仿真的需要输出格式
                    # -------------------------------------------------------------------------------------------------------------------
                    Final_Ordering_Box_V0['interval_time'] = interval_time
                    Final_Ordering_Box_V0['account_date'] = today
                    Final_Ordering_Box_V0['account_date'] = Final_Ordering_Box_V0['account_date']. \
                        apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
                    Final_Ordering_Box_V0['arrived_date'] = today + timedelta(interval_time)
                    Final_Ordering_Box_V0['arrived_date'] = Final_Ordering_Box_V0['arrived_date']. \
                        apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
                    Final_Ordering_AI = pd.DataFrame(
                        columns=['piece_bar_code', 'account_date', 'custom_stock_num', 'custom_business_num',
                                 'custom_terminal_num', 'interval_time', 'suggestion_qty', 'purchase_order_qty_AI',
                                 'arrived_date'])
                    Final_Ordering_AI['piece_bar_code'] = Final_Ordering_Box_V0['BARCODE']
                    Final_Ordering_AI['account_date'] = Final_Ordering_Box_V0['account_date']
                    Final_Ordering_AI['interval_time'] = Final_Ordering_Box_V0['interval_time']
                    Final_Ordering_AI['suggestion_qty'] = Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback']
                    Final_Ordering_AI['arrived_date'] = Final_Ordering_Box_V0['arrived_date']
                    Final_Ordering_AI['custom_stock_num'] = custom_stock_num
                    Final_Ordering_AI['custom_business_num'] = custom_business_num
                    Final_Ordering_AI['custom_terminal_num'] = custom_terminal_num
                    Final_Ordering_AI['purchase_order_qty_AI'] = Final_Ordering_AI['suggestion_qty']
                    replenishment = replenishment.append(Final_Ordering_AI)
            #这里需要将最原始的数据放入replenishment_AI表中进行用于存储最原始的补货数据
            replenishment_AI = replenishment_AI.append(replenishment)
            replenishment_2nd_mid =replenishment.groupby(['piece_bar_code','custom_stock_num',
                                         'custom_business_num','custom_terminal_num','account_date'], as_index=False).agg(sum)
            if replenishment_2nd_mid.empty == False:
                replenishment_2nd = pd.DataFrame(columns = ['piece_bar_code', 'custom_business_num', 'custom_stock_num',
                                                 'purchase_order_qty_AI','custom_terminal_num','account_date'])
                replenishment_2nd['piece_bar_code'] = replenishment_2nd_mid['piece_bar_code']
                replenishment_2nd['custom_business_num'] = replenishment_2nd_mid['custom_business_num']
                replenishment_2nd['custom_stock_num'] = replenishment_2nd_mid['custom_stock_num']
                replenishment_2nd['purchase_order_qty_AI'] = replenishment_2nd_mid['purchase_order_qty_AI']
                replenishment_2nd['custom_terminal_num'] = replenishment_2nd_mid['custom_terminal_num']
                replenishment_2nd['account_date'] = replenishment_2nd_mid['account_date']
                basic_sheet_merge_01_today = pd.merge(basic_sheet_merge_01_today,replenishment_2nd,on= \
                    ['piece_bar_code','account_date','custom_stock_num','custom_terminal_num','custom_business_num']
                                                      ,how='left')
                basic_sheet_merge_01_today = basic_sheet_merge_01_today.fillna(0)
                print('basic_sheet_merge_01_today')

            else:
                basic_sheet_merge_01_today['purchase_order_qty_AI'] = 0
            for i in range(len(basic_sheet_merge_01_today)):
                #MID_VALUE当天流转节点的AI状态下实有库存的总量
                MID_VALUE = basic_sheet_merge_01_today['start_inv_qty_AI'].iloc[i] + \
                            basic_sheet_merge_01_today['suggestion_qty'].iloc[i] + \
                            basic_sheet_merge_01_today['storge_move'].iloc[i] + \
                            basic_sheet_merge_01_today['refund_move'].iloc[i] + \
                            basic_sheet_merge_01_today['overflow_qty'].iloc[i]
                #MID_VALUE_interest当天交易节点AI状态下物权库存的总量
                MID_VALUE_interest = basic_sheet_merge_01_today['start_interest_inv_AI'].iloc[i] + \
                            basic_sheet_merge_01_today['purchase_order_qty_AI'].iloc[i] + \
                            basic_sheet_merge_01_today['storge_move'].iloc[i] + \
                            basic_sheet_merge_01_today['refund_move'].iloc[i] + \
                            basic_sheet_merge_01_today['overflow_qty'].iloc[i]
                #先判断出库量大于等于订单量的情况
                if basic_sheet_merge_01_today['delivery_qty'].iloc[i] >= basic_sheet_merge_01_today['order_qty'].iloc[i]:
                    basic_sheet_merge_01_today['shortage_qty_manual'].iloc[i] =0
                    if MID_VALUE >= basic_sheet_merge_01_today['delivery_qty'].iloc[i]:
                        basic_sheet_merge_01_today['delivery_qty_AI'].iloc[i] = basic_sheet_merge_01_today['delivery_qty'].iloc[i]
                        basic_sheet_merge_01_today['shortage_qty_AI'].iloc[i] =0
                        if MID_VALUE-basic_sheet_merge_01_today['delivery_qty_AI'].iloc[i] < basic_sheet_merge_01_today['storage_remove'].iloc[i]:
                            basic_sheet_merge_01_today['storage_remove_AI'].iloc[i] = MID_VALUE-basic_sheet_merge_01_today['delivery_qty_AI'].iloc[i]
                            basic_sheet_merge_01_today['end_inv_qty_AI'].iloc[i] = 0
                        else:
                            basic_sheet_merge_01_today['storage_remove_AI'].iloc[i] = basic_sheet_merge_01_today['storage_remove'].iloc[i]
                            basic_sheet_merge_01_today['end_inv_qty_AI'].iloc[i] = MID_VALUE-basic_sheet_merge_01_today['delivery_qty_AI'].iloc[i]- \
                                                                                  basic_sheet_merge_01_today[
                                                                                      'storage_remove'].iloc[i]
                    #加这一步的判断是说，如果当天的AI环境的物权库存和因为未得到消息而导致少补的数量的之和，大于现实环境的出库量
                    elif MID_VALUE + basic_sheet_merge_01_today['arrived_order_manual'].iloc[i] >basic_sheet_merge_01_today['delivery_qty'].iloc[i]:
                        basic_sheet_merge_01_today['delivery_qty_AI'].iloc[i] = \
                        basic_sheet_merge_01_today['delivery_qty'].iloc[i]
                        basic_sheet_merge_01_today['shortage_qty_AI'].iloc[i] = 0
                        if MID_VALUE-basic_sheet_merge_01_today['delivery_qty_AI'].iloc[i] < basic_sheet_merge_01_today['storage_remove'].iloc[i]:
                            basic_sheet_merge_01_today['storage_remove_AI'].iloc[i] = MID_VALUE+\
                                                                                      basic_sheet_merge_01_today['arrived_order_manual'].iloc[i]-\
                                                                                      basic_sheet_merge_01_today['delivery_qty_AI'].iloc[i]
                            basic_sheet_merge_01_today['end_inv_qty_AI'].iloc[i] = 0
                        else:
                            basic_sheet_merge_01_today['storage_remove_AI'].iloc[i] =basic_sheet_merge_01_today['storage_remove'].iloc[i]
                            basic_sheet_merge_01_today['end_inv_qty_AI'].iloc[i] =MID_VALUE +basic_sheet_merge_01_today['arrived_order_manual'].iloc[i]\
                                                                                  -basic_sheet_merge_01_today['delivery_qty_AI'].iloc[i]- \
                                                                                  basic_sheet_merge_01_today[
                                                                                      'storage_remove'].iloc[i]
                    else:
                        basic_sheet_merge_01_today['delivery_qty_AI'].iloc[i] = MID_VALUE
                        basic_sheet_merge_01_today['storage_remove_AI'].iloc[i] = 0
                        basic_sheet_merge_01_today['end_inv_qty_AI'].iloc[i] = 0
                        basic_sheet_merge_01_today['shortage_qty_AI'].iloc[i] = basic_sheet_merge_01_today['delivery_qty'].iloc[i] -\
                                                                                basic_sheet_merge_01_today['delivery_qty_AI'].iloc[i]
                else:
                    basic_sheet_merge_01_today['shortage_qty_manual'].iloc[i] = basic_sheet_merge_01_today['order_qty'].iloc[i]- \
                                                                                basic_sheet_merge_01_today['delivery_qty'].iloc[i]
                    if MID_VALUE >= basic_sheet_merge_01_today['order_qty'].iloc[i]:
                        basic_sheet_merge_01_today['delivery_qty_AI'].iloc[i] = basic_sheet_merge_01_today['order_qty'].iloc[i]
                        basic_sheet_merge_01_today['shortage_qty_AI'].iloc[i] = 0
                        if MID_VALUE-basic_sheet_merge_01_today['delivery_qty_AI'].iloc[i] >= basic_sheet_merge_01_today['storage_remove'].iloc[i]:
                            basic_sheet_merge_01_today['storage_remove_AI'].iloc[i] = basic_sheet_merge_01_today['storage_remove'].iloc[i]
                            basic_sheet_merge_01_today['end_inv_qty_AI'].iloc[i] = MID_VALUE -basic_sheet_merge_01_today['delivery_qty_AI'].iloc[i]\
                            -basic_sheet_merge_01_today['storage_remove_AI'].iloc[i]
                        else:
                            basic_sheet_merge_01_today['storage_remove_AI'].iloc[i] = MID_VALUE -basic_sheet_merge_01_today['delivery_qty_AI'].iloc[i]
                            basic_sheet_merge_01_today['end_inv_qty_AI'].iloc[i] = 0
                    else:
                        basic_sheet_merge_01_today['delivery_qty_AI'].iloc[i] = MID_VALUE
                        basic_sheet_merge_01_today['shortage_qty_AI'].iloc[i] = basic_sheet_merge_01_today['order_qty'].iloc[i]- \
                                                                                MID_VALUE
                        basic_sheet_merge_01_today['storage_remove_AI'].iloc[i] = 0
                        basic_sheet_merge_01_today['end_inv_qty_AI'].iloc[i] = 0
                if MID_VALUE_interest >= basic_sheet_merge_01_today['delivery_qty_AI'].iloc[i]:
                    if MID_VALUE_interest - basic_sheet_merge_01_today['delivery_qty_AI'].iloc[i] >= basic_sheet_merge_01_today['storage_remove_AI'].iloc[i]:
                        basic_sheet_merge_01_today['end_interest_inv_AI'].iloc[i] = MID_VALUE_interest -basic_sheet_merge_01_today['delivery_qty_AI'].iloc[i]- \
                                                                                      basic_sheet_merge_01_today[
                                                                                          'storage_remove_AI'].iloc[i]
                    else:
                        basic_sheet_merge_01_today['end_interest_inv_AI'].iloc[i] = 0
                elif MID_VALUE_interest+ basic_sheet_merge_01_today['arrived_order_manual'].iloc[i]>= basic_sheet_merge_01_today['delivery_qty_AI'].iloc[i]:
                    if MID_VALUE_interest+ basic_sheet_merge_01_today['arrived_order_manual'].iloc[i] - basic_sheet_merge_01_today['delivery_qty_AI'].iloc[i] >= \
                            basic_sheet_merge_01_today['storage_remove_AI'].iloc[i]:
                        basic_sheet_merge_01_today['end_interest_inv_AI'].iloc[i] = MID_VALUE_interest+ \
                                                                                    basic_sheet_merge_01_today['arrived_order_manual'].iloc[i] - \
                                                                                    basic_sheet_merge_01_today[
                                                                                        'delivery_qty_AI'].iloc[i] - \
                                                                                    basic_sheet_merge_01_today[
                                                                                        'storage_remove_AI'].iloc[i]
                    else:
                        basic_sheet_merge_01_today['end_interest_inv_AI'].iloc[i] = 0
                else:
                    basic_sheet_merge_01_today['end_interest_inv_AI'].iloc[i] = 0
            basic_today = basic_sheet_merge_01_today
            basic_today.to_csv(
                'D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/v1---仿真/02_仿真输出/evaluation-18/basic_today' + str(days) + '.csv',
                encoding="utf_8_sig")
            final_sheet = final_sheet.append(basic_today)
        #此部操作是将每次补货的结果都进行集中到replenishment_AI_totaldataframe中
        replenishment_AI_total = replenishment_AI_total.append(replenishment_AI)
        replenishment_AI.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/v1---仿真/02_仿真输出/evaluation-18/replenishment_AI.csv', encoding="utf_8_sig")
        final_sheet.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/v1---仿真/02_仿真输出/evaluation-18/final_sheet'+str(days)+'.csv', encoding="utf_8_sig")
    replenishment_AI_total.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/v1---仿真/02_仿真输出/evaluation-18/replenishment_AI_total.csv', encoding="utf_8_sig")
    final_sheet.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/v1---仿真/02_仿真输出/evaluation-18/final_sheet_revised.csv', encoding="utf_8_sig")
    calculate_finance = Calculate_finance(final_sheet,parameter)
    calculate_finance.to_csv('D:/project/P&G/数据及文档/01_相关解释文档/02_补货策略文档/v1---补货/v1---仿真/02_仿真输出/evaluation-18/calculate_finance_revised.csv',encoding="utf_8_sig")
start_end_date('20190210','20190328',0)


