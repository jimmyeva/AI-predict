# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 13:52:31 2018
@author: yuan.lin
"""
#读取得到数据
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
import cx_Oracle as cx
import matplotlib.pyplot as plt
from tqdm import *
import itertools
import datetime
from tqdm import *
import os
import pymysql
import copy
import sys
#<-----------------------------------------------------------------------------读取外部参数
def date_parameter_read():  
    date_parameter = sys.argv[1]
    date_parameter_intercept = date_parameter[0:8]
    date_parameter_transform = pd.to_datetime(datetime.datetime.strptime(date_parameter_intercept,'%Y%m%d'))
    return date_parameter_transform
#<-----------------------------------------------------------------------------以下是连接mysql,读取我们想要的数据
def data_procurement():
    dbconn=pymysql.connect(host="192.168.6.122",database="data_cjgm",user="root",password="Rpt@123456",port=3306,charset='utf8')
    sql = """select account_date,manufacturer_num,custom_business_num,custom_stock_num,custom_terminal_num,
             piece_bar_code,delivery_type_name,delivery_qty from mid_cj_sales"""
    sql_check = """SELECT piece_bar_code,item_name,category_num,
                    category_name,brand_num,brand_name,segment5,
                    goods_allow FROM mid_cj_goods"""
    sql_parameter = """SELECT id,custom_business_num,
                        custom_stock_num,custom_terminal_num,
                        manufacturer_num FROM mid_parameter_condition"""
    data_check = pd.read_sql(sql_check,dbconn)
    data_bj= pd.read_sql(sql,dbconn)
    data_parameter = pd.read_sql(sql_parameter,dbconn)
    return data_bj,data_check,data_parameter
#<-----------------------------------------------------------------------------数据筛选
def finally_data_obtain(data_bj_new,data_check,data_parameter):
    data_bj_new_sub = data_bj_new[(data_bj_new["custom_business_num"]==data_parameter[1])&(data_bj_new["custom_stock_num"]==data_parameter[2])&(data_bj_new["custom_terminal_num"]==data_parameter[3])]
    data_bj_new_sub = data_bj_new_sub[data_bj_new_sub["delivery_type_name"]=="销售"]
    data_finally = data_check[data_check["goods_allow"] == "Y"]
    sku_code = list(data_finally["piece_bar_code"].drop_duplicates())
    sku_data = [data_bj_new_sub[data_bj_new_sub["piece_bar_code"]==x] for x in sku_code]
    code = [list(pd.unique(x["piece_bar_code"])) for x in sku_data]
    code_final = [x for x in code if x]
    code_final_list = list(itertools.chain.from_iterable(code_final))
    code_final_list =  code_final_list
    data_bj_new_sub_fina = data_bj_new_sub[data_bj_new_sub["piece_bar_code"].isin(code_final_list)==True]
    return(data_bj_new_sub_fina)
#《----------------------------------------------------------------------------转化为年月日的日期格式
def date_transform(data):
    data = data.sort_values(["account_date"],ascending = 1)
    data["account_date"]= pd.to_datetime(data["account_date"].apply(lambda x : x.strftime("%Y-%m-%d")))
    return data
#<------------------------------------------------------------------------------进行日期的截取
def date_intercept(data,date_parameter):
    date_data = data[data["account_date"]<date_parameter]
    return date_data
#<----------------------------------------------------------------------------对每一个sku前后的日期进行相减
def sku_time_subtract(data):
    data["account_date"]=pd.to_datetime(data["account_date"],format='%Y%m%d')
    time_value = []
    if len(data["account_date"])<5:
        time_value=[20]
    else:
        for i in range(len(data["account_date"])-2):
            value = (data["account_date"].iloc[i+1]-data["account_date"].iloc[i]).days
            time_value.append(value)
    return time_value
#<-----------------------------------------------------------------------------对得到的补货频次进行处理和统计
def Frequencyofreplenishment(data):
    data.sort()
    data = [x for x in data if x!=0]
    dic = {}
    for i in set(data):
        dic[i]=data.count(i)
    dic_serise = pd.Series(dic)
    dic_serise = dic_serise.sort_values(ascending=False)
    return list(dic_serise.index)
#<=============================================================================对sku的预测进行建模
#<-----------------------------------------------------------------------------首先对相同的日期进行合并
def daterbind(data):
    #定义一个空的数据框
    data_daterbind = pd.DataFrame(columns = ["account_date","piece_bar_code","delivery_qty"])
    data_daterbind["delivery_qty"]=data.groupby(["account_date"]).sum()["delivery_qty"]
    data_daterbind["account_date"]=data.groupby(["account_date"]).sum().index
    data_daterbind["piece_bar_code"] = [data["piece_bar_code"].iloc[0]]*len(data_daterbind["delivery_qty"])
    return data_daterbind
#<-----------------------------------------------------------------------------对空缺的日期进行填补
def date_fill(data,date_parameter):#
    date_range_sku = pd.date_range(start='20170101',end=date_parameter-datetime.timedelta(days = 1))
    sales = [0]*len(date_range_sku)
    data_sku = pd.DataFrame({'account_date':date_range_sku,'delivery':sales})
    result = pd.merge(data, data_sku, how='right', on=['account_date'])
    result["delivery_qty"].iloc[np.where(np.isnan(result["delivery_qty"]))] = 0
    result.sort_values(['account_date'], ascending=[1])
    result["piece_bar_code"]=[data["piece_bar_code"].iloc[0]]*len(result)
    result = result.sort_values(["account_date"],ascending = 1)
    return result
#<-----------------------------------------------------------------------------这是最终得到商品的补货频次函数
def frequency_replenishment():
    date_parameter = date_parameter_read()
    bj_data_all = data_procurement()
    sku_data_finally =finally_data_obtain(bj_data_all[0],bj_data_all[1],bj_data_all[2].loc[0])
    commodity_code = np.unique(sku_data_finally["piece_bar_code"])#
    commodity_code_data = [sku_data_finally[sku_data_finally["piece_bar_code"]==x] for x in commodity_code]
    a = [date_transform(x) for x in commodity_code_data]
    f = [date_intercept(x,date_parameter) for x in a]
    b = [sku_time_subtract(x) for x in f]
    c = [Frequencyofreplenishment(x) for x in b]
    data_daterbind=[]
    for x in tqdm(range(len(a))):
        mn = date_fill(daterbind(a[x]),date_parameter)
        data_daterbind.append(mn)
    sku_frequency = {}
    for i in range(len(commodity_code)):
        sku_frequency.setdefault(commodity_code[i],c[i])
    return data_daterbind,sku_frequency,commodity_code,bj_data_all[2],date_parameter
#<=============================================================================对所有的sku进行分类处理
#筛选sku出库量为0的sku
def segmentationdata_0(data_all_finally_sku_fill_del_f):
    data_allsales_0 = [np.sum(x["delivery_qty"]) for x in data_all_finally_sku_fill_del_f]
    code_0_index = [x for x in range(len(data_allsales_0)) if data_allsales_0[x]==0]
    #if code_0_index
    code_0 = []
    for i in range(len(code_0_index)):
        code_name = data_all_finally_sku_fill_del_f[code_0_index[i]]
        code_0.append(code_name)
    return code_0
#<=============================================================================筛选sku出库量比较频繁的sku
def segmentationdata_07(data_all_finally_sku_fill_del_f):
    data_allsales_0 = [np.sum(x["delivery_qty"]) for x in data_all_finally_sku_fill_del_f]
    code_0_index = [x for x in range(len(data_allsales_0)) if data_allsales_0[x]==0]
    code_0 = []
    for i in range(len(code_0_index)):
        code_name = data_all_finally_sku_fill_del_f[code_0_index[i]]
        code_0.append(code_name)
    sku_sales_len0 = [x[x["delivery_qty"]==0].shape[0] for x in data_all_finally_sku_fill_del_f]
    sku_sales_lenall = [x.shape[0] for x in data_all_finally_sku_fill_del_f]
    sku_rate = []
    for i in range(len(sku_sales_len0)):
        result = sku_sales_len0[i]/sku_sales_lenall[i]
        sku_rate.append(result)
    sku_rate_all = pd.DataFrame({'rate':sku_rate}) 
    code_07_index =sku_rate_all[sku_rate_all["rate"]<0.6].index
    data_sku_error07 = []
    for i in code_07_index:
        data_error = data_all_finally_sku_fill_del_f[i]
        data_sku_error07.append(data_error)
    return data_sku_error07
#<=============================================================================筛选后三个月基本没什么销量的sku
#定义一个小函数，来得到后三月的剩下的sku的所有的销量
def last_threemonth_skudata(data,date_parameter):
    now_time = date_parameter-datetime.timedelta(days = 1)-datetime.timedelta(days=100)
    data_last_threemonth_data_one = data[data["account_date"]>=now_time]
    data_last_threemonth_data = data_last_threemonth_data_one[data_last_threemonth_data_one["account_date"]<=date_parameter]
    return data_last_threemonth_data
#计算后三个月的销量占比很少的sku
def last_threemonth_skuofsales_ratio(data):
    ratio = len(data[data["delivery_qty"]==0])/len(data)
    return ratio
#得到重复数值的索引
def repeat_sku_index(data):
    dic = []
    for i in data:
        dic_i = [x for x in range(len(data)) if data[x]==i]
        dic.append([i,dic_i])
    index_dic = dict(dic)
    return index_dic
#得到后三个月销量很少的sku
def segmentationdata_other(data_all_finally_sku_fill_del_f,commodity_code,date_parameter):
    data_allsales_0 = [np.sum(x["delivery_qty"]) for x in data_all_finally_sku_fill_del_f]
    code_0_index = [x for x in range(len(data_allsales_0)) if data_allsales_0[x]==0]
    code_0 = []
    for i in range(len(code_0_index)):
        code_name = data_all_finally_sku_fill_del_f[code_0_index[i]]
        code_0.append(code_name)
    sku_sales_len0 = [x[x["delivery_qty"]==0].shape[0] for x in data_all_finally_sku_fill_del_f]
    sku_sales_lenall = [x.shape[0] for x in data_all_finally_sku_fill_del_f]
    sku_rate = []
    for i in range(len(sku_sales_len0)):
        result = sku_sales_len0[i]/sku_sales_lenall[i]
        sku_rate.append(result)
    sku_rate_all = pd.DataFrame({'rate':sku_rate}) 
    code_07_index = sku_rate_all[sku_rate_all["rate"]<0.6].index
    data_sku_error07 = []
    for i in code_07_index:
        data_error = data_all_finally_sku_fill_del_f[i]
        data_sku_error07.append(data_error)
    code_0_07 = list(code_07_index)+list(code_0_index)
    code = range(0,len(commodity_code))
    code_others = set(code)-set(code_0_07)
    data_sku_others = []
    for i in code_others:
        data_error = data_all_finally_sku_fill_del_f[i]
        data_sku_others.append(data_error)
    data_allsku = [last_threemonth_skudata(x,date_parameter) for x in data_sku_others]
    data_allsku_ratio = [last_threemonth_skuofsales_ratio(x) for x in data_allsku]
    data_sku_98_index = repeat_sku_index(data_allsku_ratio)
    index_98 = [x for x in data_sku_98_index.keys() if x>0.98]
    data_sku_98_index = [data_sku_98_index[x] for x in index_98]
    data_98_index = [list(code_others)[x] for x in list(itertools.chain.from_iterable(data_sku_98_index))]
    data_other_index = set(code_others)-set(data_98_index)
    data_98_sku = [data_all_finally_sku_fill_del_f[x] for x in data_98_index]
    data_other_sku = [data_all_finally_sku_fill_del_f[x] for x in data_other_index]
    return data_98_sku,data_other_sku
#<-----------------------------------------------------------------------------对数据进行划分
def data_partition():
    frequency_replenishment_data= frequency_replenishment()
    sku_0 =  segmentationdata_0((frequency_replenishment_data[0]))
    sku_05 = segmentationdata_07(frequency_replenishment_data[0])
    sku_98 = segmentationdata_other(frequency_replenishment_data[0],frequency_replenishment_data[2],frequency_replenishment_data[4])[0]
    sku_other = segmentationdata_other(frequency_replenishment_data[0],frequency_replenishment_data[2],frequency_replenishment_data[4])[1]
    return sku_0,sku_05,sku_98,sku_other,frequency_replenishment_data[1],frequency_replenishment_data[3],frequency_replenishment_data[4]
#<=============================================================================构建模型
#<-----------------------------------------------------------------------------对出库量全部为0的sku进行销量建模
def sales_0(x,data_para,date_parameter):
    id_para = [data_para[0]]*35
    date_prediction = pd.date_range(start=date_parameter,periods = 35)
    data_sales = [0]*35
    sku_code = x["piece_bar_code"][0:35]
    custom_business_num_data = [data_para[1]]*35
    custom_stock_num_data = [data_para[2]]*35
    custom_terminal_num_data = [data_para[3]]*35
    prediction_0 = pd.DataFrame({"mid_parameter_condition_id":id_para,"custom_business_num":custom_business_num_data,"custom_stock_num":custom_stock_num_data,
                                 "custom_terminal_num":custom_terminal_num_data,"piece_bar_code":sku_code,
                                 "cnt_at":date_prediction,"forecast_qty":data_sales})
    return(prediction_0)
def result_sku_sales_0(data_daterbind,data_para,date_parameter):
    sku_sales_0 = [sales_0(x,data_para,date_parameter) for x in segmentationdata_0(data_daterbind)]
    if len(sku_sales_0)==0:
        sku_sales_0_concat = pd.DataFrame(columns=["custom_business_num","custom_stock_num","custom_terminal_num","piece_bar_code","cnt_at","forecast_qty"])
    else:      
        sku_sales_0_concat = pd.concat(sku_sales_0).reset_index(drop=True)
    return sku_sales_0_concat
#<-----------------------------------------------------------------------------对出库比较频繁的sku进行构建模型
def period_of_month(day):
    if day in range(1,11):return 1
    if day in range(11,21):return 2
    if day in range(21,32):return 3    
def period2_of_month(day):
    if day in range(1,16):return 1
    if day in range(16,32):return 2
def week_of_month(day):
    if day in range(1,8):return 1
    if day in range(8,15):return 2
    if day in range(15,22):return 3
    if day in range(22,32):return 4
def quarter(month):
    if month in range(1,4):return 1     
    if month in range(4,7):return 2
    if month in range(7,10):return 3
    if month in range(10,13):return 4 
def time_subset(x):
    x["dayofweek"]=x['account_date'].apply(lambda x:x.dayofweek)
    x["weekofyear"] = x["account_date"].apply(lambda x :x.weekofyear)
    x['month']=x['account_date'].apply(lambda x:x.month)
    x['day']=x['account_date'].apply(lambda x:x.day)
    x['year']=x['account_date'].apply(lambda x:x.year)
    x['period_of_month'] = x['day'].apply(lambda x:period_of_month(x))
    x['period2_of_month'] = x['day'].apply(lambda x:period2_of_month(x))
    x['week_of_month'] = x['day'].apply(lambda x:week_of_month(x))
    x['quarter'] = x['month'].apply(lambda x:quarter(x))
    return(x)
#处理异常值
def outli(ou):
    per_4 = np.percentile(ou['销售'],25)
    per_7 = np.percentile(ou['销售'],75)
    data_dian = per_7+1.5*(per_7-per_4)
    return data_dian
#用nan值替换其中的异常值
def nan_subset(sku):
    sku.loc[sku["销售"]>outli(sku),'销售'] = np.nan
    return(sku)   
#构建测试集
def creat_test(date):
    date_prediction = pd.date_range(start=date,periods = 35)
    date_prediction = pd.DataFrame({"account_date":date_prediction})
    date_prediction = time_subset(date_prediction)
    test = date_prediction
    return(test)
#构建随机森林的模型
def RF(data,test):
    rf=RandomForestRegressor(n_estimators=400,max_features=3) 
    rf.fit(data.iloc[:,4:12],data["delivery_qty"])
    result = rf.predict(test)
    return result
#得到最终的结果
def RF_sku_sales(data_daterbind,result_sku_RF,i,data_para,date_parameter):
    rf_sales = list(result_sku_RF[i])
    rf_date = pd.date_range(start=date_parameter,periods = 35)
    rf_code = [np.unique(data_daterbind[i]["piece_bar_code"])[0]]*35
    id_para = [data_para[0]]*35
    rf_custom_business_num_data = [data_para[1]]*35
    rf_custom_stock_num_data = [data_para[2]]*35
    rf_custom_terminal_num_data = [data_para[3]]*35
    rf_date_para = [date_parameter]*35
    rf_df = pd.DataFrame({"mid_parameter_condition_id":id_para,"custom_business_num":rf_custom_business_num_data,"custom_stock_num":rf_custom_stock_num_data,
                                 "custom_terminal_num":rf_custom_terminal_num_data,"piece_bar_code":rf_code,
                                 "cnt_at":rf_date,"forecast_qty":rf_sales,"belonged_date":rf_date_para})
    return rf_df
def RF_sales_predict(data_daterbind,data_para,date_parameter):
    test = creat_test(date_parameter)
    result_sku_RF = []
    for i in tqdm(range(len(data_daterbind))):
        a = RF(time_subset(data_daterbind[i]),test.iloc[:,1:9])
        result_sku_RF.append(a)
    sku_sales_rf = [RF_sku_sales(data_daterbind,result_sku_RF,x,data_para,date_parameter) for x in range(len(result_sku_RF))]
    sku_sales_rf_concat = pd.concat(sku_sales_rf).reset_index(drop=True)
    return sku_sales_rf_concat
#<=============================================================================对后三个月销量中销量为0的sku占比其后三个月销量大于百分之98的建模
def skuofsales_98(data,data_para,date_parameter):
    index_sku = []
    for i in range(0,len(data),35):
        index_sku.append(i)
    data_mean = []
    for i in range(len(index_sku)-2):
        data_mean.append(np.mean(data["delivery_qty"][index_sku[i]:index_sku[i+2]]))
    #计算得到除去零之后的均值
    sku_slales_mean = np.mean([x for x in data_mean if x>0])
    date_sku = pd.date_range(start=date_parameter,periods = 35)
    data_sku = [sku_slales_mean]*35
    code =[np.unique(data["piece_bar_code"])[0]]*35
    id_para = [data_para[0]]*35
    custom_business_num_data = [data_para[1]]*35
    custom_stock_num_data = [data_para[2]]*35
    custom_terminal_num_data = [data_para[3]]*35
    date_para = [date_parameter]*35
    sku_sales = pd.DataFrame({"mid_parameter_condition_id":id_para,"custom_business_num":custom_business_num_data,"custom_stock_num":custom_stock_num_data,
                                 "custom_terminal_num":custom_terminal_num_data,"piece_bar_code":code,
                                 "cnt_at":date_sku,"forecast_qty":data_sku,"belonged_date":date_para})
    return sku_sales
def sku_sales_98_predict(data_daterbind,data_para,date_parameter):
    sku_sales_98 = []
    for x in tqdm(data_daterbind):
        sku_sales_98.append(skuofsales_98(x,data_para,date_parameter))
    sku_sales_98_concat = pd.concat(sku_sales_98).reset_index(drop=True)
    return sku_sales_98_concat
#<=============================================================================对销量适中的sku进行建模
#《----------------------------------------------------------------------------计算出库频次的众数
def frequency_mode(data):
    counts = np.bincount(data)
    return np.argmax(counts)
#《----------------------------------------------------------------------------除去销量为0的sku
def sales_not0(data):
    sales_data = [x[x["delivery_qty"]>0] for x in data]
    return(sales_data)
#《----------------------------------------------------------------------------计算得到最后一次补货的时间
def replement_date(data):
    date_data = [x["account_date"].iloc[len(x["account_date"])-1] for x in data]
    return(date_data)
#《----------------------------------------------------------------------------计算每一个sku出货的众数
def sales_all(data):
    median_data = [frequency_mode(list(x["delivery_qty"])) for x in data]
    return(median_data)
def median_sku(data_fre,date_re,date_parameter):
    date_sku = pd.date_range(start=date_parameter,periods = 35)
    date_i = []
    if date_re==date_sku[0]:
        date_i.append(date_re)
        while True:
            date_re = date_re+datetime.timedelta(days=int(data_fre))
            date_i.append(date_re)
            if date_re>date_sku[34]:
                break
    else:
        while True:
            date_re = date_re+datetime.timedelta(days=int(data_fre))
            date_i.append(date_re)
            if date_re>date_sku[34]:
                break
    return(date_i)
#《----------------------------------------------------------------------------得到计算的结果
def result_sku(sale_mode,data_fre,date_re,date_parameter):
    date_sku = pd.date_range(start=date_parameter,periods = 35)
    sales = [0]*35
    sku_pd_0 = pd.DataFrame({"date":date_sku,"sales":sales})
    date_two = median_sku(data_fre,date_re,date_parameter)
    date_two = [x for x in date_two if x>date_sku[0]]
    sale = sale_mode*len(date_two)
    sku_pd_predict = pd.DataFrame({"date":date_two,"sale":sale})
    result = pd.merge(sku_pd_predict, sku_pd_0, how='right', on=['date'])
    result["sale"].iloc[np.where(np.isnan(result["sale"]))] = 0
    result = result.sort_values(["date"],ascending = 1)
    return result
def other_sku_sales(sales_median,result_data,i,data_para,date_parameter):
    other_sales = result_data[i]["sale"]
    other_date = result_data[i]["date"]
    other_code = [sales_median[i]]*35
    id_para = [data_para[0]]*35
    custom_business_num_data = [data_para[1]]*35
    custom_stock_num_data = [data_para[2]]*35
    custom_terminal_num_data = [data_para[3]]*35
    date_para = [date_parameter]*35
    other_df = pd.DataFrame({"mid_parameter_condition_id":id_para,"custom_business_num":custom_business_num_data,"custom_stock_num":custom_stock_num_data,
                                 "custom_terminal_num":custom_terminal_num_data,"piece_bar_code":other_code,
                                 "cnt_at":other_date,"forecast_qty":other_sales,"belonged_date":date_para})
    return other_df
def other_sku_sales_predict(sku_frequency,data_daterbind,data_para,date_parameter):
    #首先得到销量适中的sku的编码
    sales_median = [np.unique(x["piece_bar_code"])[0] for x in data_daterbind]
    sku_replenishment_frequency = [sku_frequency[x] for x in sales_median]
    #得到补货频次的众数
    sales_mezzo = sales_not0(data_daterbind)
    replementofdate = replement_date(sales_mezzo)
    sale_mode = sales_all(sales_mezzo)
    result_data = []
    for i in tqdm(range(len(sku_replenishment_frequency))):
        a= result_sku(sale_mode[i],np.median(sku_replenishment_frequency[i]),replementofdate[i],date_parameter)
        result_data.append(a)
    sku_sales_other = [other_sku_sales(sales_median,result_data,x,data_para,date_parameter) for x in range(len(result_data))]
    sku_sales_other_concat = pd.concat(sku_sales_other).reset_index(drop=True)
    return sku_sales_other_concat
#填补缺失值
def fill_null(data):
    data["forecast_qty"] = data["forecast_qty"].fillna(0)
    return data
#《============================================================================得到最终的结果，并进行建模
def main_data_modeling():
    sku_type_data = data_partition()
    #首先得到销售量为0的销售预测量
    #在得到出货量比较大的sku的销售预测
    sku_sales_RF = RF_sales_predict(sku_type_data[1],sku_type_data[5].loc[0],sku_type_data[6])
    sku_sales_98 = sku_sales_98_predict(sku_type_data[2],sku_type_data[5].loc[0],sku_type_data[6])
    sku_sales_98["forecast_qty"] = sku_sales_98["forecast_qty"].fillna(0)
    sku_sales_other = other_sku_sales_predict(sku_type_data[4],sku_type_data[3],sku_type_data[5].loc[0],sku_type_data[6])
    list_sku_sales = [sku_sales_RF,sku_sales_98,sku_sales_other]
    all_sku_prediction = pd.concat(list_sku_sales).reset_index(drop=True)
    return all_sku_prediction
#<=============================================================================尤妮佳销量预测模型
def unicharm_data_prediction():
    bj_data_all = data_procurement()
    date_parameter = date_parameter_read()
    sku_data_finally =finally_data_obtain(bj_data_all[0],bj_data_all[1],bj_data_all[2].loc[2])
    commodity_code = np.unique(sku_data_finally["piece_bar_code"])#
    commodity_code_data = [sku_data_finally[sku_data_finally["piece_bar_code"]==x] for x in commodity_code]
    a = [date_transform(x) for x in commodity_code_data]
    data_daterbind=[]
    for x in tqdm(range(len(a))):
        mn = date_fill(daterbind(a[x]),date_parameter)
        data_daterbind.append(mn)
    unicharm_data = RF_sales_predict(data_daterbind,bj_data_all[2].loc[2],date_parameter)
    return bj_data_all,unicharm_data
#<=============================================================================得到最终的结果(包含尤妮佳的销量预测)
def all_type_prediction():
    bj_rule_one = main_data_modeling()
    bj_rule_two = copy.deepcopy(bj_rule_one)
    ynj_rule_one = unicharm_data_prediction()
    bj_rule_two["mid_parameter_condition_id"] = ynj_rule_one[0][2].loc[1][0]
    all_sku_sales = [bj_rule_one,bj_rule_two,ynj_rule_one[1]]
    all_sku_prediction = pd.concat(all_sku_sales).reset_index(drop=True)
    all_sku_prediction["belonged_date"] = all_sku_prediction["belonged_date"].apply(lambda x : x.strftime("%Y-%m-%d"))
    all_sku_prediction["cnt_at"] = all_sku_prediction["cnt_at"].apply(lambda x : x.strftime("%Y-%m-%d"))
    return all_sku_prediction
#当可下单的数量非常小的时候
def small_Number_orders():
    bj_data_all = data_procurement()
    date_parameter = date_parameter_read()
    sku_data_finally =finally_data_obtain(bj_data_all[0],bj_data_all[1],bj_data_all[2].loc[0])
    commodity_code = np.unique(sku_data_finally["piece_bar_code"])#
    commodity_code_data = [sku_data_finally[sku_data_finally["piece_bar_code"]==x] for x in commodity_code]
    a = [date_transform(x) for x in commodity_code_data]
    data_daterbind=[]
    for x in tqdm(range(len(a))):
        mn = date_fill(daterbind(a[x]),date_parameter)
        data_daterbind.append(mn)
    f = [date_intercept(x,date_parameter) for x in data_daterbind]
    unicharm_data = RF_sales_predict(f,bj_data_all[2].loc[2],date_parameter)
    unicharm_data["belonged_date"] = unicharm_data["belonged_date"].apply(lambda x : x.strftime("%Y-%m-%d"))
    unicharm_data["cnt_at"] = unicharm_data["cnt_at"].apply(lambda x : x.strftime("%Y-%m-%d"))
    return unicharm_data
#当所有的数据都为空的时候返回一个空的数据框
def empty_dataframe():
    data = pd.DataFrame(columns = ["cnt_at","custom_business_num","custom_stock_num","custom_terminal_num","forecast_qty","mid_parameter_condition_id","piece_bar_code","belonged_date"])
    return data
#判断条件,假如可下单的数量非常少的时候，运行small_Number_order函数
def Analyzing_conditions():
    #date_parameter = date_parameter_read()
    bj_data_all = data_procurement()
    sku_data_finally =finally_data_obtain(bj_data_all[0],bj_data_all[1],bj_data_all[2].loc[0])
    if len(np.unique(sku_data_finally["piece_bar_code"]))<800 and len(np.unique(sku_data_finally["piece_bar_code"]))>0:
        return small_Number_orders()
    elif len(np.unique(sku_data_finally["piece_bar_code"]))==0:
        return empty_dataframe()
    else:
        return all_type_prediction()
#<=============================================================================写入数据库
def connectdb():
    print('连接到mysql服务器...')
    db = pymysql.connect(host="192.168.6.122",database="data_cjgm",user="root",password="Rpt@123456",port=3306,charset='utf8')
    print('连接上了!')
    return db
#《----------------------------------------------------------------------------删除重复日期数据
def drop_data(db):
    cursor = db.cursor()
    date_parameter = date_parameter_read()
    date_parameter = date_parameter.strftime("%Y-%m-%d")
    sql = """delete from dm_cj_forecast where belonged_date = str_to_date(\'%s\','%%Y-%%m-%%d')"""%(date_parameter)
    cursor.execute(sql)
def insertdb(db,data):
    cursor = db.cursor()
    try:
        for i in range(len(data)):
            cnt_at=data.iloc[i]["cnt_at"]
            custom_business_num = data.iloc[i]["custom_business_num"]
            custom_stock_num= data.iloc[i]["custom_stock_num"]
            custom_terminal_num = data.iloc[i]["custom_terminal_num"]
            forecast_qty = data.iloc[i]["forecast_qty"]
            mid_parameter_condition_id = data.iloc[i]["mid_parameter_condition_id"]
            piece_bar_code = data.iloc[i]["piece_bar_code"]
            belonged_date = data.iloc[i]["belonged_date"]
            #print(cnt_at,custom_business_num,custom_stock_num,custom_terminal_num,forecast_qty,mid_parameter_condition_id,piece_bar_code,belonged_date)
            cursor.execute("""INSERT INTO dm_cj_forecast (custom_business_num,
                                                          custom_stock_num,
                                                          custom_terminal_num,
                                                          piece_bar_code,
                                                          cnt_at,
                                                          forecast_qty,
                                                          belonged_date,
                                                          mid_parameter_condition_id
                                                          ) VALUES ('%s','%s','%s','%s',str_to_date(\'%s\','%%Y-%%m-%%d'),
                                                          '%s',str_to_date(\'%s\','%%Y-%%m-%%d'),'%s')""" % (custom_business_num,custom_stock_num,
                                                          custom_terminal_num,piece_bar_code,cnt_at,forecast_qty,belonged_date,mid_parameter_condition_id))
            db.commit()
    except OSError as reason:
        print('出错原因是%s'%str(reason))
        db.rollback()
def closedb(db):
    db.close()
def main():
    db = connectdb()
    data = Analyzing_conditions()
    drop_data(db)
    if data.empty:
        print("The data frame is empty")
        closedb(db)
    else:
        insertdb(db,data)
        closedb(db)
        print("result:1")
#《============================================================================主函数入口
if __name__ == '__main__':
    try:
        main()
    except OSError as reason:
        print('出错原因是%s'%str(reason))
        print ("result:0")