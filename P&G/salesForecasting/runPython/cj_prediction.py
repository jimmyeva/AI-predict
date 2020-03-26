# -*- coding: utf-8 -*-
"""
Created on 2018-12-27
@author: yuan.lin
"""
#读取得到数据
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from tqdm import *
import itertools
import datetime
import os
import pymysql
import copy
import sys
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import math
import warnings
# from chinese_calendar import is_workday, is_holiday
# import chinese_calendar as calendar  #
import time
warnings.filterwarnings("ignore")
#<----------------------------------------------------------------------------------------------------------------------读取外部参数
def date_parameter_read():
    date_parameter = sys.argv[1]
    date_parameter_intercept = date_parameter[0:8]
    date_parameter_transform = pd.to_datetime(datetime. datetime.strptime(date_parameter_intercept,'%Y%m%d'))
    return date_parameter_transform


#<---------------------------------------------------------------------------------------------------------------------—读取数据库的数据
def database_read():
    # dbconn = pymysql.connect(host="192.168.6.122",database="data_cjgm",user="root",password="Rpt@123456",port=3306,charset='utf8')
    dbconn = pymysql.connect(host="172.16.4.7", database="supply_chain", user="bi_user", password="RL9FCS4@QTrmOsRk",port=3310, charset='utf8')
    sql_sales = """select account_date,manufacturer_num,custom_business_num,custom_stock_num,custom_terminal_num,
             piece_bar_code,delivery_type_name,delivery_qty,transaction_unit_price_tax from mid_cj_sales"""
    sql_check = """SELECT piece_bar_code,item_name,category_num,
                    category_name,brand_num,brand_name,segment5,
                    goods_allow,basic_price FROM mid_cj_goods"""
    sql_parameter = """SELECT custom_business_num,
                        custom_stock_num,custom_terminal_num,
                        manufacturer_num,order_delivery_time FROM mid_parameter_condition"""
    data_check = pd.read_sql(sql_check,dbconn)
    data_bj_sales= pd.read_sql(sql_sales,dbconn)
    data_parameter = pd.read_sql(sql_parameter,dbconn)
    dbconn.close()
    return data_bj_sales,data_check,data_parameter

#<----------------------------------------------------------------------------------------------------------------------获取最终要预测的原始数据
def finally_data_obtain(data_bj_new,data_check,data_parameter):
    data_bj_new_sub = data_bj_new[(data_bj_new["custom_business_num"]==data_parameter["custom_business_num"])&
                                  (data_bj_new["custom_stock_num"]==data_parameter["custom_stock_num"])&
                                (data_bj_new["custom_terminal_num"]==data_parameter["custom_terminal_num"])&
                                  (data_bj_new["manufacturer_num"] == data_parameter["manufacturer_num"])]
    data_bj_new_sub = data_bj_new_sub[data_bj_new_sub["delivery_type_name"]=="销售"]
    data_finally = data_check[data_check["goods_allow"] == "Y"]
    sku_code = list(data_finally["piece_bar_code"].drop_duplicates())
    data_bj_new_sub_fina = data_bj_new_sub[data_bj_new_sub["piece_bar_code"].isin(sku_code)==True]
    return data_bj_new_sub_fina

#<----------------------------------------------------------------------------------------------------------------------对日期转化为标准的格式
def date_transform(data):
    data = data.sort_values(["account_date"],ascending = 1)
    data["account_date"]= pd.to_datetime(data["account_date"].apply(lambda x : x.strftime("%Y-%m-%d")))
    return data

#<----------------------------------------------------------------------------------------------------------------------进行日期的截取
def date_intercept(data,date_parameter):
    date_data = data[data["account_date"]<date_parameter]
    return date_data

#<------------------------------------------------------------------------首先对相同的日期进行合并
def date_rbind(data):
    data_daterbind = pd.DataFrame(columns = ["account_date","piece_bar_code","delivery_qty","transaction_unit_price_tax"])
    data_daterbind["delivery_qty"]= data.groupby(["account_date"]).sum()["delivery_qty"]
    data_daterbind["transaction_unit_price_tax"] = data.groupby(["account_date"]).mean()["transaction_unit_price_tax"]
    data_daterbind["account_date"]=data.groupby(["account_date"]).sum().index
    data_daterbind["piece_bar_code"] = [data["piece_bar_code"].iloc[0]]*len(data_daterbind["delivery_qty"])
    return data_daterbind

#<----------------------------------------------------------------------------------------------------------------------对空缺的日期进行填补
def date_fill(data_check,data,date_parameter):#
    date_range_sku = pd.date_range(start='20170101', end=date_parameter - datetime.timedelta(days=1))
    sales = [0] * len(date_range_sku)
    price = [0] * len(date_range_sku)
    data_sku = pd.DataFrame({'account_date': date_range_sku, 'delivery': sales, "transaction_price": price})
    result = pd.merge(data, data_sku, how='right', on=['account_date'])
    del result["delivery"]
    del result["transaction_price"]
    result["delivery_qty"].iloc[np.where(np.isnan(result["delivery_qty"]))] = 0
    basic_price = list(data_check[data_check["piece_bar_code"] == data["piece_bar_code"].iloc[0]]["basic_price"])
    list_index = [x for x in basic_price if not x is None]
    if len(list_index) == False:
        result["transaction_unit_price_tax"] = result["transaction_unit_price_tax"].fillna(result["transaction_unit_price_tax"].mean())
    else:
        result["transaction_unit_price_tax"] = result["transaction_unit_price_tax"].fillna(basic_price[0])
    fill_piece_bar_code = np.unique(data["piece_bar_code"])[0]
    result["piece_bar_code"] = result["piece_bar_code"].fillna(fill_piece_bar_code)
    result = result.sort_values(["account_date"], ascending=1)
    return result

#<----------------------------------------------------------------------------------------------------------------------获取各种类型所需要预测数据的sku组成的list集合
def all_manufacturer_prediction_data(**type_parameter):
    date_parameter = date_parameter_read()
    manufacturer_data_all = database_read()
    data_manufacturer_sales,data_check,data_parameter_set =manufacturer_data_all[0],manufacturer_data_all[1],manufacturer_data_all[2]
    data_manufacturer_parameter = data_parameter_set[(data_parameter_set["manufacturer_num"]==type_parameter.get('manufacturer_num'))&
                                       (data_parameter_set["custom_business_num"] == type_parameter.get('custom_business_num'))&
                                       (data_parameter_set["custom_stock_num"] == type_parameter.get('custom_stock_num'))&
                                       (data_parameter_set["custom_terminal_num"] == type_parameter.get('custom_terminal_num'))].iloc[0]
    sku_data_finally =finally_data_obtain(data_manufacturer_sales,data_check,data_manufacturer_parameter)
    commodity_code = np.unique(sku_data_finally["piece_bar_code"])#
    commodity_code_data = [sku_data_finally[sku_data_finally["piece_bar_code"]==x] for x in commodity_code]
    date_transform_data = [date_transform(x) for x in commodity_code_data]
    data_datefill = []
    for x in tqdm(range(len(date_transform_data))):
        data_datefill_i = date_fill(data_check, date_rbind(date_transform_data[x]),date_parameter)
        data_datefill_intercept = date_intercept(data_datefill_i,date_parameter)
        data_datefill.append(data_datefill_intercept)
    print("预测数据获取成功")
    return data_datefill,commodity_code,data_parameter_set,date_parameter,data_manufacturer_parameter,commodity_code_data,date_transform_data


#<----------------------------------------------------------------------------------------------------------------------对所有的sku进行分类处理
#<----------------------------------------------------------------------------------------------------------------------筛选sku出库量为0的sku
def Outgoing_data_zero(Outgoing_data_all_sku):
    data_all_sales_zero = [np.sum(x["delivery_qty"]) for x in Outgoing_data_all_sku]
    code_zero_index = [x for x in range(len(data_all_sales_zero)) if data_all_sales_zero[x]==0]
    code_zero = []
    for i in range(len(code_zero_index)):
        code_name = Outgoing_data_all_sku[code_zero_index[i]]
        code_zero.append(code_name)
    return code_zero

#<----------------------------------------------------------------------------------------------------------------------筛选sku出库量比较频繁的sku
def Outgoing_data_many(Outgoing_data_all_sku):
    data_all_sales_zero = [np.sum(x["delivery_qty"]) for x in Outgoing_data_all_sku]
    code_zero_index = [x for x in range(len(data_all_sales_zero)) if data_all_sales_zero[x]==0]
    code_zero = []
    for i in range(len(code_zero_index)):
        code_name = Outgoing_data_all_sku[code_zero_index[i]]
        code_zero.append(code_name)
    sku_sales_len_zero = [x[x["delivery_qty"]==0].shape[0] for x in Outgoing_data_all_sku]
    sku_sales_len_all = [x.shape[0] for x in Outgoing_data_all_sku]
    sku_rate = []
    for i in range(len(sku_sales_len_zero)):
        result = sku_sales_len_zero[i]/sku_sales_len_all[i]
        sku_rate.append(result)
    sku_rate_all = pd.DataFrame({'rate':sku_rate})
    code_many_index =sku_rate_all[sku_rate_all["rate"]<0.6].index
    data_sku_error_many = []
    for i in code_many_index:
        data_error = Outgoing_data_all_sku[i]
        data_sku_error_many.append(data_error)
    return  data_sku_error_many

#<----------------------------------------------------------------------------------------------------------------------筛选后三个月基本没什么销量的sku
#定义一个小函数，来得到后三月的剩下的sku的所有的销量
def last_threemonth_skudata(data,date_parameter):
    now_time = date_parameter-datetime.timedelta(days = 1)-datetime.timedelta(days=100)
    data_last_threemonth_data_one = data[data["account_date"]>=now_time]
    data_last_threemonth_data = data_last_threemonth_data_one[data_last_threemonth_data_one["account_date"]<=date_parameter]
    return data_last_threemonth_data

#<----------------------------------------------------------------------------------------------------------------------计算后三个月的销量占比很少的sku
def last_threemonth_skuofsales_ratio(data):
    ratio = len(data[data["delivery_qty"]==0])/len(data)
    return ratio

#<----------------------------------------------------------------------------------------------------------------------得到重复数值的索引
def repeat_sku_index(data):
    dic = []
    for i in data:
        dic_i = [x for x in range(len(data)) if data[x]==i]
        dic.append([i,dic_i])
    index_dic = dict(dic)
    return index_dic

#<----------------------------------------------------------------------------------------------------------------------得到后三个月销量很少的sku
def Outgoing_data_other(Outgoing_data_all_sku,commodity_code,date_parameter):
    data_all_sales_zero = [np.sum(x["delivery_qty"]) for x in Outgoing_data_all_sku]
    code_zero_index = [x for x in range(len(data_all_sales_zero)) if data_all_sales_zero[x]==0]
    code_zero = []
    for i in range(len(code_zero_index)):
        code_name = Outgoing_data_all_sku[code_zero_index[i]]
        code_zero.append(code_name)
    sku_sales_len_zero = [x[x["delivery_qty"]==0].shape[0] for x in Outgoing_data_all_sku]
    sku_sales_len_all = [x.shape[0] for x in Outgoing_data_all_sku]
    sku_rate = []
    for i in range(len(sku_sales_len_zero)):
        result = sku_sales_len_zero[i]/sku_sales_len_all[i]
        sku_rate.append(result)
    sku_rate_all = pd.DataFrame({'rate':sku_rate})
    code_many_index =sku_rate_all[sku_rate_all["rate"]<0.6].index
    data_sku_error_many = []
    for i in code_many_index:
        data_error = Outgoing_data_all_sku[i]
        data_sku_error_many.append(data_error)
    code_zero_many = list(code_many_index)+list(code_zero_index)
    code = range(0,len(commodity_code))
    code_others = set(code)-set(code_zero_many)
    data_sku_others = []
    for i in code_others:
        data_error = Outgoing_data_all_sku[i]
        data_sku_others.append(data_error)
    data_allsku = [last_threemonth_skudata(x,date_parameter) for x in data_sku_others]
    data_allsku_ratio = [last_threemonth_skuofsales_ratio(x) for x in data_allsku]
    data_sku_few_index = repeat_sku_index(data_allsku_ratio)
    index_few = [x for x in data_sku_few_index.keys() if x>0.98]
    data_sku_few_index = [data_sku_few_index[x] for x in index_few]
    data_few_index = [list(code_others)[x] for x in list(itertools.chain.from_iterable(data_sku_few_index))]
    data_other_index = set(code_others)-set(data_few_index)
    data_few_sku = [Outgoing_data_all_sku[x] for x in data_few_index]
    data_other_sku = [Outgoing_data_all_sku[x] for x in data_other_index]
    return data_few_sku,data_other_sku

#<----------------------------------------------------------------------------------------------------------------------对宝洁的数据进行划分
def data_bj_obtain():
    bj_prediction_data_all = all_manufacturer_prediction_data(manufacturer_num="000320",
                                                                    custom_business_num=3,
                                                                    custom_stock_num=1,
                                                                    custom_terminal_num=1)
    data_bj_datefill, commodity_bj_code, data_parameter_set, date_bj_parameter,data_bj_parameter = bj_prediction_data_all[0],\
                                                                                      bj_prediction_data_all[1],\
                                                                                      bj_prediction_data_all[2],\
                                                                                      bj_prediction_data_all[3],\
                                                                                      bj_prediction_data_all[4]
    return data_parameter_set, date_bj_parameter,data_bj_parameter,data_bj_datefill,commodity_bj_code

#<----------------------------------------------------------------------------------------------------------------------对宝洁的数据进行分类处理
def data_bj_partition(data_bj_datefill,commodity_bj_code,bj_date_para):
    date_bj_parameter = bj_date_para
    bj_sku_sales_zero =  Outgoing_data_zero(data_bj_datefill)
    bj_sku_sales_many = Outgoing_data_many(data_bj_datefill)
    bj_sku_sales_other = Outgoing_data_other(data_bj_datefill,commodity_bj_code,date_bj_parameter)
    bj_sku_sales_median = bj_sku_sales_other[1]
    bj_sku_sales_less = bj_sku_sales_other[0]
    return bj_sku_sales_zero,bj_sku_sales_many,bj_sku_sales_median,bj_sku_sales_less

def period_of_month(day):
    if day in range(1, 11): return 1
    if day in range(11, 21): return 2
    if day in range(21, 32): return 3

def period2_of_month(day):
    if day in range(1, 16): return 1
    if day in range(16, 32): return 2

def week_of_month(day):
    if day in range(1, 8): return 1
    if day in range(8, 15): return 2
    if day in range(15, 22): return 3
    if day in range(22, 32): return 4

def quarter(month):
    if month in range(1, 4): return 1
    if month in range(4, 7): return 2
    if month in range(7, 10): return 3
    if month in range(10, 13): return 4

#<----------------------------------------------------------------------------------------------------------------------添加节假日期的特征因素
def holiday_day(date):
    on_holiday, holiday_name = calendar.get_holiday_detail(date)
    return on_holiday

def time_subset(x):
    x["dayofweek"] = x['account_date'].apply(lambda x: x.dayofweek)
    x["weekofyear"] = x["account_date"].apply(lambda x: x.weekofyear)
    x['month'] = x['account_date'].apply(lambda x: x.month)
    x['day'] = x['account_date'].apply(lambda x: x.day)
    x['year'] = x['account_date'].apply(lambda x: x.year)
    x['period_of_month'] = x['day'].apply(lambda x: period_of_month(x))
    x['period2_of_month'] = x['day'].apply(lambda x: period2_of_month(x))
    x['week_of_month'] = x['day'].apply(lambda x: week_of_month(x))
    x['quarter'] = x['month'].apply(lambda x: quarter(x))
    # x["holiday"] = x["account_date"].apply(lambda x: holiday_day(x))
    # x["holiday_index"] = [1 if a == True else 0 for a in x["holiday"]]
    # del x["holiday"]
    return x

# <----------------------------------------------------------------------------------------------------------------------构建测试集
def creat_test():
    date_parameter = date_parameter_read()
    date_prediction = pd.date_range(start=date_parameter,periods=35)
    date_dataframe = pd.DataFrame({"account_date":date_prediction})
    test = time_subset(date_dataframe)
    return test

# <----------------------------------------------------------------------------------------------------------------------处理异常值
def outli(ou):
    per_4 = np.percentile(ou['销售'], 25)
    per_7 = np.percentile(ou['销售'], 75)
    data_dian = per_7 + 1.5 * (per_7 - per_4)
    return data_dian

# <----------------------------------------------------------------------------------------------------------------------用nan值替换其中的异常值
def nan_subset(sku):
    sku.loc[sku["销售"] > outli(sku),'销售'] = np.nan
    return (sku)

#<----------------------------------------------------------------------------------------------------------------------计算峰度
def kurtosis_compute(data):
    data_mean = np.mean(data)
    data_var = np.var(data)+0.1
    data_sc = np.mean((data - data_mean) ** 3)
    data_ku = np.mean((data - data_mean) ** 4) / pow(data_var, 2)  # 计算峰度
    return data_ku

# <---------------------------------------------------------------------------------------------------------------------构建每一个时间分布上的特征
def time_agg(train,test_df,vars_to_agg,vars_be_agg):  # 构建比较多的特征
    for var in vars_to_agg:
        agg = train.groupby(var)[vars_be_agg].agg(["sum", "mean", "std", "skew", "median", "min", "max","count",kurtosis_compute])
        if isinstance(var, list):
            agg.columns = pd.Index(["fare_by_" + "_".join(var) + "_" + str(e) for e in agg.columns.tolist()])
        else:
            agg.columns = pd.Index(["fare_by_" + var + "_" + str(e) for e in agg.columns.tolist()])
        train = pd.merge(train, agg.reset_index(), on=var, how="left")
        test_df = pd.merge(test_df, agg.reset_index(), on=var, how="left")
    return train, test_df

#<-----------------------------------------------------------------------------------添加所要所要预测的距离该sku第一次售卖的时间的长度,权重
def add_time_diff(train_data,test_data):
    min_date = train_data["account_date"].min()
    date_train_diff = train_data["account_date"]-min_date
    date_test_diff = test_data["account_date"]-min_date
    train_data["date_diff"] = date_train_diff.apply(lambda x:x.days)
    train_data = train_data[train_data["date_diff"]>0]
    train_data["date_diff"] = train_data["date_diff"].apply(lambda x : np.exp(1/x))
    test_data["date_diff"] = date_test_diff.apply(lambda x:x.days)
    test_data = test_data[test_data["date_diff"] > 0]
    test_data["date_diff"] = test_data["date_diff"].apply(lambda x :np.exp(1/x))
    return train_data,test_data

#<----------------------------------------------------------------------------------------------------------------------构建随机森林模型
def construct_randomforest_model(train_feature,train_targe,test):
    rf = RandomForestRegressor(n_estimators=300, max_features=10)#<-----------------------------------------------------参数有待调整
    rf.fit(train_feature,train_targe)
    result = rf.predict(test)
    return result

#<----------------------------------------------------------------------------------------------------------------------构建KNN模型
def construct_knn_model(train_feature,train_targe,test):
    knn_model = KNeighborsRegressor(n_neighbors=10, leaf_size=13, n_jobs=-1)
    knn_model.fit(train_feature,train_targe)#<--------------------------------------------------------------------------参数有待调整
    knn_test_pre = knn_model.predict(test)
    return  knn_test_pre

#<----------------------------------------------------------------------------------------------------------------------对输出的结果进行规范化
def prediction_result_Regularization(train,result,data_para,test,date_parameter):
    prediction_sales = result
    prediction_date = test["account_date"]
    prediction_code = [np.unique(train["piece_bar_code"])[0]]*35
    prediction_manufacture = [data_para["manufacturer_num"]]*35
    custom_business_num_data = [data_para["custom_business_num"]]*35
    custom_stock_num_data = [data_para["custom_stock_num"]]*35
    custom_terminal_num_data = [data_para["custom_terminal_num"]]*35
    prediction_date_para = [date_parameter]*35
    prediction_df = pd.DataFrame({"manufacturer_num":prediction_manufacture,
                                  "custom_business_num":custom_business_num_data,
                                  "custom_stock_num":custom_stock_num_data,
                                 "custom_terminal_num":custom_terminal_num_data,
                                  "piece_bar_code":prediction_code,
                                 "cnt_at":prediction_date,
                                  "forecast_qty":prediction_sales,
                                  "belonged_date":prediction_date_para})
    return prediction_df

#<----------------------------------------------------------------------------------------------------------------------得到特征输出结果
def prediction_feature_targe(sku_i):
    train = time_subset(sku_i)
    test = creat_test()
    vars_be_agg = "delivery_qty"
    vars_to_agg = ["dayofweek", "weekofyear", "month", "day", "year", "period_of_month", "period2_of_month",
                   "week_of_month", "quarter", ["month", "dayofweek"], ["quarter", "month"]]
    data = time_agg(train, test, vars_to_agg, vars_be_agg)
    train_feature_data = data[0].fillna(0)
    test_feature_data = data[1].fillna(0)
    train_test_data = add_time_diff(train_feature_data,test_feature_data)
    train_feature_data_result = train_test_data[0]
    test_feature_data_result = train_test_data[1]
    train_feature_data = train_feature_data_result.drop(["account_date","piece_bar_code","delivery_qty","transaction_unit_price_tax"],axis=1)
    prediction_feature = list(train_feature_data.columns)
    prediction_target = ["delivery_qty"]
    return train,test,train_feature_data_result,test_feature_data_result,prediction_feature,prediction_target

#<----------------------------------------------------------------------------------------------------------------------构建的随机森林模型得到最终的结果
def prediction_sku_sales_RandomForestRegressor_model(sku_i,data_para,date_parameter):
    prediction_feature_targe_data = prediction_feature_targe(sku_i)
    train = prediction_feature_targe_data[0]
    test = prediction_feature_targe_data[1]
    train_feature_data_result = prediction_feature_targe_data[2]
    test_feature_data_result = prediction_feature_targe_data[3]
    prediction_feature = prediction_feature_targe_data[4]
    prediction_target = prediction_feature_targe_data[5]
    prediction_result = list(construct_randomforest_model(train_feature_data_result[prediction_feature],
                                                          train_feature_data_result[prediction_target],
                                                          test_feature_data_result[prediction_feature]))
    result = prediction_result_Regularization(train,prediction_result,data_para,test,date_parameter)
    return result

#<----------------------------------------------------------------------------------------------------------------------构建的KNN模型得到最终的结果
def prediction_sku_sales_knn_model(sku_i,data_para,date_parameter):
    prediction_feature_targe_data = prediction_feature_targe(sku_i)
    train = prediction_feature_targe_data[0]
    test = prediction_feature_targe_data[1]
    train_feature_data_result = prediction_feature_targe_data[2]
    test_feature_data_result = prediction_feature_targe_data[3]
    prediction_feature = prediction_feature_targe_data[4]
    prediction_target = prediction_feature_targe_data[5]
    prediction_result = list(itertools.chain.from_iterable(construct_knn_model(train_feature_data_result[prediction_feature],
                                                train_feature_data_result[prediction_target],
                                                test_feature_data_result[prediction_feature])))
    result = prediction_result_Regularization(train,prediction_result,data_para,test,date_parameter)
    return result

#<----------------------------------------------------------------------------------------------------------------------进行模型的融合
def RF_KNN_model_merge(sku_i,data_para,date_parameter):
    RF_result = prediction_sku_sales_RandomForestRegressor_model(sku_i,data_para,date_parameter)
    knn_result = prediction_sku_sales_knn_model(sku_i,data_para,date_parameter)
    RF_knn_merge_result = RF_result.copy()
    RF_knn_merge_result["forecast_qty"] = 0.5*(RF_result["forecast_qty"]+knn_result["forecast_qty"])
    return RF_knn_merge_result

#<----------------------------------------------------------------------------------------------------------------------对销量比较小的sku进行建模
def data_sku_sales_few_model(sku_i,data_para,date_parameter):
    sku_slales_mean = np.mean(sku_i["delivery_qty"])
    date_sku = pd.date_range(start=date_parameter,periods = 35)
    data_sku = [sku_slales_mean]*35
    code =[np.unique(sku_i["piece_bar_code"])[0]]*35
    prediction_manufacture = [data_para["manufacturer_num"]] * 35
    custom_business_num_data = [data_para["custom_business_num"]] * 35
    custom_stock_num_data = [data_para["custom_stock_num"]] * 35
    custom_terminal_num_data = [data_para["custom_terminal_num"]] * 35
    date_para = [date_parameter]*35
    sku_sales = pd.DataFrame({"manufacturer_num":prediction_manufacture,"custom_business_num":custom_business_num_data,
                              "custom_stock_num":custom_stock_num_data,"custom_terminal_num":custom_terminal_num_data,
                              "piece_bar_code":code,"cnt_at":date_sku,"forecast_qty":data_sku,"belonged_date":date_para})
    return sku_sales

#<----------------------------------------------------------------------------------------------------------------------定义一个空的数据框
def empty_dataframe():
    data = pd.DataFrame(columns = ["cnt_at","custom_business_num","custom_stock_num","custom_terminal_num","forecast_qty","manufacturer_num","piece_bar_code","belonged_date"])
    return data

#<----------------------------------------------------------------------------------------------------------------------当宝洁的可下单sku数量非常小的时候，我们自定义的分类结果会出现错误，运行下面函数
#<----------------------------------------------------------------------------------------------------------------------预测宝洁的结果
def bj_prediction_result():
    bj_data_obtain = data_bj_obtain()
    data_parameter_set = bj_data_obtain[0]
    bj_date_para = bj_data_obtain[1]
    bj_data_para = bj_data_obtain[2]
    data_bj_datefill = bj_data_obtain[3]
    commodity_bj_code = bj_data_obtain[4]
    if len(data_bj_datefill)<800 and len(data_bj_datefill)>0:
        bj_data_sku_sales_result = [RF_KNN_model_merge(x, bj_data_para, bj_date_para) for x in data_bj_datefill]
        bj_prediction_all_sku = pd.concat(bj_data_sku_sales_result).reset_index(drop=True)
    elif len(data_bj_datefill)==0:
        bj_prediction_all_sku = empty_dataframe()
    else:
        bj_data_prediction = data_bj_partition(data_bj_datefill,commodity_bj_code,bj_date_para)#<--------------------------------------------------------------------对宝洁的数据进行切分
        bj_data_sku_sales_many = bj_data_prediction[1]
        bj_data_sku_sales_median = bj_data_prediction[2]
        bj_data_sku_sales_few = bj_data_prediction[3]
        bj_data_sku_sales_many_result = [RF_KNN_model_merge(x,bj_data_para,bj_date_para)for x in bj_data_sku_sales_many]
        print("宝洁销量多的sku预测成功")
        bj_data_sku_sales_median_result = [RF_KNN_model_merge(x,bj_data_para,bj_date_para) for x in bj_data_sku_sales_median]
        print("宝洁销量适中的sku预测成功")
        bj_data_sku_sales_few_result = [data_sku_sales_few_model(x,bj_data_para,bj_date_para) for x in bj_data_sku_sales_few]
        print("宝洁销量小的sku预测成功")
        bj_data_sku_sales_many_result_df = pd.concat(bj_data_sku_sales_many_result).reset_index(drop=True)
        bj_data_sku_sales_median_result_df = pd.concat(bj_data_sku_sales_median_result).reset_index(drop=True)
        bj_data_sku_sales_few_result_df = pd.concat(bj_data_sku_sales_few_result).reset_index(drop=True)
        bj_prediction_all_sku = pd.concat([bj_data_sku_sales_many_result_df,bj_data_sku_sales_median_result_df,bj_data_sku_sales_few_result_df]).reset_index(drop=True)
        print("宝洁所有sku预测成功并且数据合并成功")
    return bj_prediction_all_sku

#<----------------------------------------------------------------------------------------------------------------------预测尤妮佳的结果
def ynj_prediction_result():
    ynj_prediction_data_all = all_manufacturer_prediction_data(manufacturer_num="000323", custom_business_num=4,
                                                custom_stock_num=2, custom_terminal_num=1)
    ynj_data_datefill, ynj_commodity_code, data_parameter_set, ynj_date_parameter,ynj_data_parameter = ynj_prediction_data_all[0], \
                                                                                                   ynj_prediction_data_all[1], \
                                                                                                   ynj_prediction_data_all[2], \
                                                                                                   ynj_prediction_data_all[3], \
                                                                                                   ynj_prediction_data_all[4]
    if len(ynj_data_datefill) == 0:
        data_sku_ynj_prediction = empty_dataframe()
    else:
        data_sku_sales_ynj_result = [RF_KNN_model_merge(x,ynj_data_parameter,ynj_date_parameter) for x in ynj_data_datefill]
        print("尤妮佳sku预测成功")
        data_sku_ynj_prediction = pd.concat(data_sku_sales_ynj_result).reset_index(drop=True)
        print("尤妮佳预测数据合并成功")
    return data_sku_ynj_prediction

#<----------------------------------------------------------------------------------------------------------------------预测联合利华的结果
def lianhelihua_prediction_result():
    lianhelihua_prediction_data_all = all_manufacturer_prediction_data(manufacturer_num="000053", custom_business_num=9,
                                               custom_stock_num=6, custom_terminal_num=1)
    lianhelihua_data_datefill, lianheliahua_commodity_code, data_parameter_set, lianhelihua_date_parameter,lianhelihua_data_parameter = lianhelihua_prediction_data_all[0], \
                                                                                                                   lianhelihua_prediction_data_all[1], \
                                                                                                                   lianhelihua_prediction_data_all[2], \
                                                                                                                   lianhelihua_prediction_data_all[3], \
                                                                                                                     lianhelihua_prediction_data_all[4]
    if len(lianhelihua_data_datefill) == 0:
        data_sku_lianhelihua_prediction = empty_dataframe()
    else:
        data_sku_sales_lianhelihua_result = [RF_KNN_model_merge(x,lianhelihua_data_parameter,lianhelihua_date_parameter) for x in lianhelihua_data_datefill]
        print("联合利华sku预测成功")
        data_sku_lianhelihua_prediction = pd.concat(data_sku_sales_lianhelihua_result).reset_index(drop=True)
        print("联合利华预测数据合并成功")
    return data_sku_lianhelihua_prediction

#<----------------------------------------------------------------------------------------------------------------------预测宜昌宝洁的结果
def yi_chang_bj_prediction_result():
    yichangbj_prediction_data_all = all_manufacturer_prediction_data(manufacturer_num="000320", custom_business_num=10,
                                               custom_stock_num=7, custom_terminal_num=1)
    yichangbj_data_datefill, yichangbj_commodity_code, data_parameter_set, yichangbj_date_parameter,yichangbj_data_parameter = yichangbj_prediction_data_all[0], \
                                                                                                                               yichangbj_prediction_data_all[1], \
                                                                                                                               yichangbj_prediction_data_all[2], \
                                                                                                                               yichangbj_prediction_data_all[3], \
                                                                                                                               yichangbj_prediction_data_all[4]
    if len(yichangbj_data_datefill) == 0:
        data_sku_yichangbj_prediction = empty_dataframe()
    else:

        data_sku_sales_yichangbj_result = [RF_KNN_model_merge(x,yichangbj_data_parameter,yichangbj_date_parameter) for x in yichangbj_data_datefill]
        print("宜昌宝洁sku预测成功")
        data_sku_yichangbj_prediction = pd.concat(data_sku_sales_yichangbj_result).reset_index(drop=True)
        print("宜昌宝洁预测数据合并成功")
    return data_sku_yichangbj_prediction

#<----------------------------------------------------------------------------------------------------------------------预测零售通的结果
def lingshoutong_bj_prediction_result():
    lingshoutong_prediction_data_all = all_manufacturer_prediction_data(manufacturer_num="000320", custom_business_num=8,
                                               custom_stock_num=4, custom_terminal_num=4)
    lingshoutong_data_datefill, lingshoutong_commodity_code, data_parameter_set, lingshoutong_date_parameter,lingshoutong_data_parameter = lingshoutong_prediction_data_all[0], \
                                                                                                                                           lingshoutong_prediction_data_all[1], \
                                                                                                                                           lingshoutong_prediction_data_all[2], \
                                                                                                                                           lingshoutong_prediction_data_all[3], \
                                                                                                                                           lingshoutong_prediction_data_all[4]
    if len(lingshoutong_data_datefill) == 0:
        data_sku_lingshoutong_prediction = empty_dataframe()
    else:
        data_sku_sales_lingshoutong_result = [RF_KNN_model_merge(x,lingshoutong_data_parameter,lingshoutong_date_parameter) for x in lingshoutong_data_datefill]
        print("零售通sku预测成功")
        data_sku_lingshoutong_prediction = pd.concat(data_sku_sales_lingshoutong_result).reset_index(drop=True)
        print("零售通预测数据合并成功")
    return data_sku_lingshoutong_prediction

#<----------------------------------------------------------------------------------------------------------------------合并所有预测品牌的结果
def all_result_concat():
    bj_prediction = bj_prediction_result()
    ynj_prediction = ynj_prediction_result()
    lianhelihua_prediction = lianhelihua_prediction_result()
    yichangbj_prediction = yi_chang_bj_prediction_result()
    lingshoutong_prediction = lingshoutong_bj_prediction_result()
    contact_data = [bj_prediction,ynj_prediction,lianhelihua_prediction,yichangbj_prediction,lingshoutong_prediction]
    if len(contact_data) == 0:
        all_prediction_result = empty_dataframe()
    else:
        all_prediction_result = pd.concat(contact_data).reset_index(drop=True)
        print("所有品牌的sku预测成功并且数据合并成功")
    return all_prediction_result
#<----------------------------------------------------------------------------------------------------------------------规整化数据
def Consolidation_data(data):
    data_result = pd.DataFrame({"belonged_date":data["belonged_date"],
                                "cnt_at":data["cnt_at"],
                                "custom_business_num":data["custom_business_num"],
                                "custom_stock_num":data["custom_stock_num"],
                                "custom_terminal_num":data["custom_terminal_num"],
                                "forecast_qty":data["forecast_qty"],
                                "manufacturer_num":data["manufacturer_num"],
                                "piece_bar_code":data["piece_bar_code"]})
    data_result["belonged_date"] = data_result["belonged_date"].apply(lambda x: x.strftime("%Y-%m-%d"))
    data_result["cnt_at"] = data_result["cnt_at"].apply(lambda x: x.strftime("%Y-%m-%d"))
    return data_result

def connectdb():
    print('连接到mysql服务器...')
    # db = pymysql.connect(host="192.168.6.122",database="data_cjgm",user="root",password="Rpt@123456",port=3306,charset='utf8')
    db = pymysql.connect(host="172.16.4.7", database="supply_chain", user="bi_user", password="RL9FCS4@QTrmOsRk",port=3310, charset='utf8')
    print('连接上了!')
    return db

#《---------------------------------------------------------------------------------------------------------------------删除重复日期数据
def drop_data(db):
    cursor = db.cursor()
    date_parameter = date_parameter_read()
    date_parameter = date_parameter.strftime("%Y-%m-%d")
    sql = """delete from dm_cj_forecast where belonged_date = str_to_date(\'%s\','%%Y-%%m-%%d')"""%(date_parameter)
    cursor.execute(sql)
#<======================================================================================================================
def insertdb(db,data):
    cursor = db.cursor()
    param = list(map(tuple, np.array(data).tolist()))
    sql = """INSERT INTO dm_cj_forecast (belonged_date,cnt_at,custom_business_num,
    custom_stock_num,custom_terminal_num,forecast_qty,manufacturer_num,piece_bar_code)
     VALUES (str_to_date(%s,'%%Y-%%m-%%d'), 
     str_to_date(%s,'%%Y-%%m-%%d'),'%s','%s','%s','%s',%s,%s)"""
    try:
        cursor.executemany(sql, param)
        print("所有品牌的sku数据插入数据库成功")
        db.commit()
    except OSError as reason:
        print('出错原因是%s' % str(reason))
        db.rollback()
#<=============================================================================
def closedb(db):
    db.close()
#<=============================================================================
def main():
    all_result_brand = all_result_concat()
    all_result_brand = Consolidation_data(all_result_brand)
    db = connectdb()
    drop_data(db)
    if all_result_brand.empty:
        print("The data frame is empty")
        print("result:1")
        closedb(db)
    else:
        insertdb(db,all_result_brand)
        closedb(db)
        print("result:1")
#《============================================================================主函数入口
if __name__ == '__main__':
    try:
        main()
    except OSError as reason:
        print('出错原因是%s'%str(reason))
        print ("result:0")
#<================================================================================结束
