# -*- coding = utf-8 -*-
'''
@Time: 2018/12/10 14:53
@Author: Ye Jinyu
'''
# -*- coding = utf-8 -*-
'''
@Time: 2018/11/6 10:28
@Author: Ye Jinyu
'''

from Parameters import *
import Data_cleaning
import numpy as np
import warnings
import math

warnings.filterwarnings("ignore")
import pymysql

# ---------------------------------------------------------------------------------------------
## parameters for clean data(combine_data)
DataRead_0 = Original_Data()
DataClean = Data_cleaning.CleanData()


# ------------------------------------------------------------------------

# def Data_Analyse():
    ## loop computation from
parameter_condition_sql = """select * from mid_parameter_condition
                             where status =1"""
parameter_condition = DataRead_0.Mysql_Data(parameter_condition_sql)
parameter_condition = parameter_condition.set_index('id')

for i in range(5):
    print(i)
    # import the parameters of custom_business_num ...

    parameters = parameter_condition.iloc[i, :]
    # global manufacturer_num, custom_business_num, custom_stock_num, custom_terminal_num,t2
    manufacturer_num = parameters['manufacturer_num']
    custom_business_num = parameters['custom_business_num']
    custom_stock_num = parameters['custom_stock_num']

    custom_terminal_num = parameters['custom_terminal_num']
    interval_time = str(parameters['order_delivery_time']) + str('+') + str(parameters['delivery_arrival_time'])
    mid_parameter_condition_id = parameter_condition.index[i]

    t2 = int(parameters['arrival_time'])
    # print(manufacturer_num, custom_business_num, custom_stock_num, custom_terminal_num, t2)
    # Parameters.sql_read(manufacturer_num,custom_business_num,custom_stock_num,custom_terminal_num)
    predict_sql = """select cnt_at,piece_bar_code,forecast_qty from dm_cj_forecast 
                     where mid_parameter_condition_id=%s
                     and belonged_date=%s
                     """ % (mid_parameter_condition_id,predict_time)

    sale_sql = """select piece_bar_code ,delivery_qty,account_date from mid_cj_sales
                  where manufacturer_num = %s
                  and custom_business_num = %s 
                  and custom_stock_num = %s
                  and custom_terminal_num = %s
                  and account_date > %s""" % (manufacturer_num, custom_business_num, custom_stock_num, custom_terminal_num, start_sale_time)
    ## parameters for new file that include usable inventory(include end inventory + road inventory + in storage)

    usable_inventory_sql = """select piece_bar_code,cnt_at,available_inv_qty  from mid_cj_inv a
                                              where custom_stock_num =%s
                                              and manufacturer_num = %s
                                              and cnt_at= %s  """ % (
    custom_stock_num, manufacturer_num, usable_inventory_time)

    ## the parameters for base information of sku:including brand ,box factor
    base_information_sql = """select piece_bar_code,category_num,spaced,
                                      brand_num,case_conversion_rate,segment5,status   
                              from mid_cj_goods c"""

    ## parameters for order time
    order_sql = """select order_date,piece_bar_code,max_uom_qty from mid_cj_order
                    where manufacturer_num = %s
                    and custom_business_num = %s 
                    and custom_stock_num = %s
                    and order_date>= %s """ % (manufacturer_num, custom_business_num, custom_stock_num, order_time)

    ## the parameters for usable quota of promotional sku
    usable_promotion_quota_sql = """SELECT effective_start_date,piece_bar_code,quota  from mid_cj_quota a 
                                   where custom_business_num = %s 
                                   and effective_start_date > %s""" % (custom_business_num, quota_time)

    ## the parameters for feedback frame
    feedback_sql = """select piece_bar_code, correct_cause_id from mid_cj_cause
    where correct_cause_id in (1,2)"""
    # print(sale_sql)
    ## create table for output
    # print(sale_sql,usable_inventory_sql,base_information_sql,order_sql,usable_promotion_quota_sql,feedback_sql )
    # ------------------------------------------------------------------------
    # --------------------------------------------------------------------
    ##  deal with the data : Sale Data
    ## sum BJ_TOTAL_QTY per day and per BJ_BARCODE
    print(mid_parameter_condition_id)
    def Deal_with_Sale_Data():
        Sale_Data = DataRead_0.Mysql_Data(sale_sql)
        Sale_Data.columns = ['BJ_BARCODE', 'BJ_TOTAL_QTY', 'BJ_DATE']
        # print(Sale_Data)
        Sale_Data['BJ_DATE'] = pd.to_datetime(Sale_Data['BJ_DATE']).dt.normalize()
        # print(Sale_Data)
        Sale_Data_group = Sale_Data.groupby(['BJ_BARCODE', 'BJ_DATE'], as_index=False)
        # print(Sale_Data_group)
        Sale_Data_group_sum = Sale_Data_group.agg(sum)
        # print(Sale_Data_group_sum)
        ## fill  0 which has no number in that day
        # the function use for fill the number which frame has no number of other days
        # print(Sale_Data_group_sum['BJ_DATE'])
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
        Sale_Data_Week = Sale_Data_Fill[Sale_Data_Fill['BJ_DATE'].isin(DataRead_0.rq_range(start_time, end_time))]
        Week_agg = Sale_Data_Week.groupby(['BJ_BARCODE', 'WEEK_DAY'], as_index=False).agg(sum)
        Week_sum = Week_agg.groupby(['BJ_BARCODE'], as_index=False)
        Week_sum = Week_sum['BJ_TOTAL_QTY'].agg(sum)
        Week_agg_sum = pd.merge(Week_agg, Week_sum, on='BJ_BARCODE', how='left')
        Week_agg_sum['PROB'] = Week_agg_sum['BJ_TOTAL_QTY_x'] / (Week_agg_sum['BJ_TOTAL_QTY_y'] + 0.0000001)
        Sale_Barcode = set(Week_agg_sum['BJ_BARCODE'])
        return Sale_Selected_Date, Week_agg_sum, Sale_Barcode, Sale_Data_Fill

    ## deal with the storage of today:Instore_Road_Store
    def Deal_with_Instorage():
        Instore_Road_Store = DataRead_0.Mysql_Data(usable_inventory_sql)
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

    ##=========================================================================================================
    Final_Ordering = pd.DataFrame(columns=["BARCODE", "Usable_Storage", "Max_SS", "Order_or_not", "Order_num"])
    for barcode in Unique_Barcode:
        # for test
        # barcode = '4902430680431'
        # usable_storage for a sku
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

        ## study the error of the all data
        learn_error = pd.DataFrame({
            "bj_date": np.array(sales_qty['BJ_DATE']),
            "real_sales": np.array(sales_qty['BJ_TOTAL_QTY']),
            "pre_sales": np.array(predict_sales_agg['forecast_qty'][0:len(sales_qty)]),
            "error": np.array(np.array(sales_qty['BJ_TOTAL_QTY']) - np.array(
                predict_sales_agg['forecast_qty'][0:len(sales_qty)])),
            "WEEK_DAY": np.array(DataClean.to_weekday(sales_qty['BJ_DATE']))
        })
        learn_error_join = pd.merge(learn_error, week_list, on='WEEK_DAY', how='left')
        learn_error_join['prob_star'] = learn_error_join['PROB'] / (sum(learn_error_join['PROB']) + 0.000001)
        learn_error_join['prob_num'] = sum(learn_error_join['error']) * learn_error_join['prob_star']
        learn_error_join_select = learn_error_join[['prob_num', 'WEEK_DAY']]
        basic_error_num = learn_error_join_select.drop_duplicates(subset=['WEEK_DAY'], keep='first')
        predict_sales_num = pd.merge(predict_sales, basic_error_num, on='WEEK_DAY', how="left")

        ## deal with the predict data,put it as a list type,distinct the circulation
        pre_list = DataClean.agg_to_frame(predict_sales, t3)
        error_list = DataClean.agg_to_frame(predict_sales_num, t3)

        # -------------------------------------------------------------------------------------
        # 1 prediction sum for t2 days
        pre_sum = DataClean.Pre_Error_sum(pre_list, 'forecast_qty', t2, t3)
        error_sum = DataClean.Pre_Error_sum(error_list, 'prob_num', t2, t3)

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
        Final_Ordering = Final_Ordering.append(Final_Ordering,ignore_index=True)
        print(Final_Ordering)
    # Final_Ordering = Final_Ordering.append(Final_Ordering, ignore_index=True)
    # print(Final_Ordering)
        ## make the data complete
        # Base_Information = DataRead_0.Mysql_Data(base_information_sql)
        # Base_Information.columns = ['BARCODE', 'CATEGORY_CODE', 'SPACED',
        #                             'BRAND_NUM', 'OUT_FACTOR', 'PRODUCT_NATURE', 'STATUS']
        # ## delete the duplicate number
        # Base_Information_Unique = Base_Information
        #
        # ## through the sales of every month,calculate mean of sales number
        # sales_slow_selected = Sale_Data_Fill[Sale_Data_Fill['BJ_DATE'].isin(DataRead_0.rq_range(srq_start, srq_end))]
        # sales_slow_selected_delete_group = sales_slow_selected.groupby(['BJ_BARCODE'], as_index=False)
        # sales_slow_selected_month = sales_slow_selected_delete_group['BJ_TOTAL_QTY'].mean()
        # sales_slow_selected_month['BJ_TOTAL_QTY'] = sales_slow_selected_month['BJ_TOTAL_QTY'] * 30
        # sales_slow_selected_month.columns = ['BJ_BARCODE', 'Month_Sale']
        # ## join the box factor
        # box_factor_frame = Base_Information_Unique[['BARCODE', 'OUT_FACTOR']]
        # slow_sales_month = pd.merge(sales_slow_selected_month, box_factor_frame, how='left', right_on='BARCODE',
        #                             left_on='BJ_BARCODE')
        # slow_sales_month['OUT_FACTOR'] = pd.to_numeric(slow_sales_month['OUT_FACTOR'].fillna(1))
        # slow_sales_month['month_box_sale'] = slow_sales_month['Month_Sale'] / slow_sales_month['OUT_FACTOR']
        # slow_sales_month['slow_selling'] = list(map(lambda x: 1 if x <= 1 else 2, slow_sales_month['month_box_sale']))
        #
        # ## 1 mean slow good and 2 mean not slow goods
        # slow_barcode = slow_sales_month[['BJ_BARCODE', 'month_box_sale', 'slow_selling']]
        #
        # ##=================================================================================================
        # ## calculate the usable quota
        # Promotion_Quato = DataRead_0.Mysql_Data(usable_promotion_quota_sql)
        # Promotion_Quato.columns = ['CNT_AT', 'BARCODE', 'QUOTA_QTY']
        # Promotion_Quato['CNT_AT'] = pd.to_datetime(Promotion_Quato['CNT_AT']).dt.normalize()
        # quota_frame = Promotion_Quato.groupby(['BARCODE'], as_index=False).sum()
        #
        # ## though the order frame calculate the usable quota
        # Order_Frame = DataRead_0.Mysql_Data(order_sql)
        # Order_Frame.columns = ['AUDIT_DATE', 'BARCODE', 'QUANTITY']
        # Order_Frame['AUDIT_DATE'] = pd.to_datetime(Order_Frame['AUDIT_DATE']).dt.normalize()
        # Order_Frame_group = Order_Frame.groupby(['BARCODE'], as_index=False).sum()
        # Order_Frame_num = Order_Frame_group[['BARCODE', 'QUANTITY']]
        #
        # quota_ordered_frame = pd.merge(quota_frame, Order_Frame_num, on='BARCODE', how='left')
        # quota_ordered_frame['QUANTITY'] = quota_ordered_frame['QUANTITY'].fillna(0)
        # quota_ordered_frame['USABLE_QUOTY'] = quota_ordered_frame['QUOTA_QTY'] - quota_ordered_frame['QUANTITY']
        # quota_ordered_frame['USABLE_QUOTY'][quota_ordered_frame['USABLE_QUOTY'] <= 0] = 0
        # quota_barcode = quota_ordered_frame.drop_duplicates(subset=['BARCODE'], keep='first')
        # quota_barcode = quota_barcode[['BARCODE', 'USABLE_QUOTY']]
        # ## join the Final_Ordering with the box factor
        # Final_Ordering_Box = pd.merge(Final_Ordering, box_factor_frame, how='left', on='BARCODE')
        # Final_Ordering_Box['OUT_FACTOR'] = pd.to_numeric(Final_Ordering_Box['OUT_FACTOR'].fillna(1))
        # Final_Ordering_Box['ORDER_NUM_BOX'] = Final_Ordering_Box['Order_num'] / Final_Ordering_Box['OUT_FACTOR']
        # Final_Ordering_Box['ORDER_NUM_BOX'] = list(map(lambda x: math.floor(x), Final_Ordering_Box['ORDER_NUM_BOX']))
        #
        # ## join the slow sales factor
        # Final_Ordering_Box_Slow = pd.merge(Final_Ordering_Box, slow_barcode, left_on='BARCODE', right_on='BJ_BARCODE',
        #                                    how='left')
        # Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'] = Final_Ordering_Box_Slow['ORDER_NUM_BOX']
        # Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'][Final_Ordering_Box_Slow['slow_selling'] == 1] = 0
        # Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'] = Final_Ordering_Box_Slow['ORDER_NUM_BOX_s'].fillna(0)
        # ## join the quota information
        # Final_Ordering_Box_Quota = pd.merge(Final_Ordering_Box_Slow, quota_barcode, on='BARCODE', how='left')
        # Rep_Base = Base_Information_Unique[
        #     ['BARCODE', 'CATEGORY_CODE', 'SPACED', 'BRAND_NUM', 'OUT_FACTOR', 'PRODUCT_NATURE', 'STATUS']]
        # Final_Ordering_Box_Quota_Base = pd.merge(Final_Ordering_Box_Quota, Rep_Base, on='BARCODE', how='left')
        # Final_Ordering_Box_Quota_Base['USABLE_QUOTY'] = Final_Ordering_Box_Quota_Base['USABLE_QUOTY'].fillna(0)
        # Final_Ordering_Box_Quota_Base['USABLE_QUOTY'][Final_Ordering_Box_Quota_Base['PRODUCT_NATURE'] == 'Y'] = float(
        #     'inf')
        #
        # ## if the order number is bigger than quota,use the quota replace for it
        # Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'] = 0
        # Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'][
        #     Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s'] > Final_Ordering_Box_Quota_Base['USABLE_QUOTY']] = \
        # Final_Ordering_Box_Quota_Base['USABLE_QUOTY']
        # Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_FINAL'][
        #     Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s'] <= Final_Ordering_Box_Quota_Base['USABLE_QUOTY']] = \
        # Final_Ordering_Box_Quota_Base['ORDER_NUM_BOX_s']
        #
        # ## join the product category
        # Final_Ordering_Box_V0 = Final_Ordering_Box_Quota_Base
        #
        # ## join the objective information
        # FeedBack = DataRead_0.Mysql_Data(feedback_sql)
        # FeedBack.columns = ['BARCODE', 'REMARKS']
        # stop_barcode = list(set(FeedBack['BARCODE']))
        #
        # Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback'] = list(
        #     map(lambda x: math.floor(x), Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL']))
        # Final_Ordering_Box_V0['ORDER_NUM_BOX_FINAL_causeback'][Final_Ordering_Box_V0['BARCODE'].isin(stop_barcode)] = 0
        # Final_Ordering_Box_V0['BJ_DATE'] = Today_date
        #
        # Final_Ordering_Box_V0['USABLE_BOX'] = Final_Ordering_Box_V0['Usable_Storage'] / Final_Ordering_Box_V0[
        #     'OUT_FACTOR_x']
        #
        # for row in range(len(Final_Ordering_Box_V0)):
        #     insert_row = Final_Ordering_Box_V0.iloc[row, :]
        #     cnt_at = Parameter_cnt
        #
        #     piece_bar_code = insert_row['BARCODE']
        #     use_inv_qty = int(insert_row['USABLE_BOX'])
        #     replenish_qty = insert_row['ORDER_NUM_BOX']
        #     suggestion_qty = insert_row['ORDER_NUM_BOX_FINAL_causeback']
        #     plan_qty = insert_row['ORDER_NUM_BOX_FINAL_causeback']
        #
        #     cur.execute("""insert into dm_cj_replenishment_new (custom_business_num,custom_stock_num,custom_terminal_num,
        #                                cnt_at,manufacturer_num,interval_time,piece_bar_code, use_inv_qty,replenish_qty,suggestion_qty,
        #                                plan_qty,mid_parameter_condition_id)
        #                     values('%s','%s','%s',str_to_date(\'%s\','%%Y-%%m-%%d %%H:%%i:%%s'),'%s','%s','%s','%s','%s','%s'
        #                     ,'%s','%s')"""
        #                 % (custom_business_num, custom_stock_num, custom_terminal_num,
        #                    cnt_at.strftime("%Y-%m-%d %H:%M:%S"),
        #                    manufacturer_num, interval_time, piece_bar_code, use_inv_qty, replenish_qty, suggestion_qty,
        #                    plan_qty, mid_parameter_condition_id))
        #
        #     conn.commit()


# if __name__ == "__main__":
#     try:
#         # connect mysql
#         conn = pymysql.connect(**parame_mysql)
#         cur = conn.cursor()
#         cur.execute("""delete from dm_cj_replenishment_new where date(cnt_at) = %s""" % predict_time)
#         Data_Analyse()
#         conn.close()
#         print("result:1")
#     except Exception as err:
#         conn.close()
#         print('repr(err):\t', repr(err))
#         print("result:0")

    # Final_Ordering_Box_V1 = Final_Ordering_Box_V0[['BJ_DATE','BARCODE','Max_SS','BRAND_NUM','CATEGORY_CODE',
    #                                 'PRODUCT_NATURE','OUT_FACTOR_x','SPACED','STATUS',
    #                                  'slow_selling','USABLE_QUOTY','ORDER_NUM_BOX_FINAL_causeback','USABLE_BOX','month_box_sale']]

    # final_test = Final_Ordering_Box_V1[['BJ_DATE','BARCODE','USABLE_QUOTY','ORDER_NUM_BOX_FINAL_causeback','USABLE_BOX','month_box_sale']]

    # final_test = Final_Ordering_Box_V1[['BJ_DATE','BARCODE','USABLE_QUOTY','ORDER_NUM_BOX_FINAL_causeback','USABLE_BOX']]
    # Final_Ordering_Box_V1.to_csv('F:/data_sales/result/single_day/Final_Ordering_Box_V1_9_18.csv',encoding="utf_8_sig") #绝对位置
