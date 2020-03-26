# -*- coding: utf-8 -*-
# @Time    : 2019/7/9 9:07
# @Author  : Ye Jinyu__jimmy
# @File    : SALES_FORECAST.py
import pandas as pd
import cx_Oracle
import os
import numpy as np
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
# 注：设置环境编码方式，可解决读取数据库乱码问题
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
from matplotlib import pyplot as plt

#parser是根据字符串解析成datetime,字符串可以很随意，可以用时间日期的英文单词，
# 可以用横线、逗号、空格等做分隔符。没指定时间默认是0点，没指定日期默认是今天，没指定年份默认是今年。
# from pylab import *
plt.switch_backend('agg')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
#如下是支持中文数字
# mpl.rcParams['font.sans-serif'] = ['SimHei']
#读取得到数据
from sklearn.ensemble import RandomForestRegressor
from tqdm import *
import itertools
import datetime
from sklearn.neighbors import KNeighborsRegressor
import warnings

# import chinese_calendar as calendar  #
warnings.filterwarnings("ignore")
import get_holiday_inf_2019
import get_holiday_inf_2018

holiday_2019 = get_holiday_inf_2019.main()
holiday_2019 = holiday_2019[1].fillna(0)
holiday_2018 = get_holiday_inf_2018.main()
holiday_2018 = holiday_2018[1].fillna(0)


#<---------------------------------------------------------------------------------->确认外部参数的计算日期
def date_parameter_read(end_date):
    # date_parameter = sys.argv[1]
    # date_parameter_intercept = date_parameter[0:8]
    end = pd.to_datetime(datetime. datetime.strptime(end_date,'%Y%m%d'))
    return end


#--------------------------------------------------------------------------------->对日期进行转化返回string前一天的日期
def date_convert(end):
    datetime_forma= datetime.datetime.strptime(end, "%Y%m%d")
    yesterday = datetime_forma - datetime.timedelta(days=1)
    yesterday = yesterday.strftime("%Y%m%d")
    return yesterday


#-------------------------------------------------------------------------->函数读取近两年总销量前50名的SKU的资源id
def read_oracle_data(start_date,end_date):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    #读取的数据包括销量的时间序列，天气和活动信息
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", parameters)
    #查看详细的出库数据，进行了日期的筛选，查看销量签50名的SKU
    stkout_detail_sql = """SELECT sum(s.qty),s.GDGID FROM STKOUTDTL s INNER JOIN(select *
                                from STKOUT s
                                INNER JOIN STORE s1  ON s.sender = s1.gid 
                                INNER JOIN STORE s2 ON s.CLIENT = s2.gid 
                                WHERE bitand(s1.property,32)=32 
                                AND bitand(s2.property,32)<>32 
                                AND substr(s2.AREA,2,3)<'8000' 
                                AND s.CLS='统配出')b ON s.NUM = b.NUM AND s.CLS='统配出' 
                                AND s.TOTAL> 0  and b.OCRDATE >= to_date('%s','yyyy-mm-dd') 
                                AND b.OCRDATE <= to_date('%s','yyyy-mm-dd') 
                                GROUP BY s.GDGID order by sum(s.QTY) DESC""" % (start_date,end_date,)
    GDGID_sales = pd.read_sql(stkout_detail_sql, conn)
    #将SKU的的iD转成list，并保存前50个，再返回值
    conn.close
    sku_id = GDGID_sales['GDGID'].tolist()
    sku_id = sku_id[0:2]
    return sku_id


#------------------------------------------------------------------>根据SKU 的id来获取每个SKU的具体的销售明细数据

def get_detail_sales_data(sku_id,start_date,end_date):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    dsn = cx_Oracle.makedsn(host, port, sid)

    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码)
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", dsn)
    # 查看出货详细单的数据
    stkout_detail_sql = """SELECT s.GDGID,b.NUM,s.RTOTAL,b.OCRDATE,s.CRTOTAL,s.MUNIT,s.QTY,s.QTYSTR,
                            s.TOTAL,s.PRICE,s.QPC FROM STKOUTDTL s INNER JOIN(
                            select *
                            from STKOUT s
                            INNER JOIN STORE s1  ON s.sender = s1.gid
                            INNER JOIN STORE s2 ON s.CLIENT = s2.gid 
                            WHERE bitand(s1.property,32)=32 
                            AND bitand(s2.property,32)<>32 
                            AND substr(s2.AREA,2,3)<'8000' 
                            AND s.CLS='统配出')b ON s.NUM = b.NUM AND s.CLS='统配出' 
                            and s.GDGID = %s and  s.TOTAL> 0 and b.OCRDATE >= to_date('%s','yyyy-mm-dd') 
                                and b.OCRDATE <= to_date('%s','yyyy-mm-dd')""" % (sku_id,start_date,end_date)
    stkout_detail = pd.read_sql(stkout_detail_sql, conn)
    conn.close
    return stkout_detail

#------------------------------------------------------------------------------>读取数据库中历史
#设置一个函数用来选择链接oracle拿到对应的某款SKU的售卖门店数量的时间序列,GROUPBY在sql里面运行的话会耗费好久的时间，每个SKU需要至少5-8分钟，因此把合并的操作放在
def sku_shops_count(sku_id,start_date,end_date):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    dsn = cx_Oracle.makedsn(host, port, sid)

    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", dsn)
    # 查看出货详细单的数据
    sku_shops_count = """SELECT b.OCRDATE,COUNT(b.CLIENT) as ClIENT  FROM STKOUTDTL s INNER JOIN(select *
                            from STKOUT s
                            INNER JOIN STORE s1  ON s.sender = s1.gid  
                            INNER JOIN STORE s2 ON s.CLIENT = s2.gid 
                            WHERE bitand(s1.property,32)=32
                            AND bitand(s2.property,32)<>32
                            AND substr(s2.AREA,2,3)<'8000'
                            AND s.CLS='统配出')b ON s.NUM = b.NUM AND s.CLS='统配出' 
                            AND s.GDGID= %s AND s.TOTAL> 0 and b.OCRDATE >= to_date
                            ('%s','yyyy-mm-dd') and b.OCRDATE <= to_date('%s','yyyy-mm-dd') 
                            GROUP BY b.OCRDATE""" % (sku_id,start_date,end_date)
    GDGID_Count = pd.read_sql(sku_shops_count, conn)
    conn.close
    return GDGID_Count


#--------------------------------------------------------------------------------->日的标准化转化
def date_normalize(data_frame):
    data_frame_sort = data_frame.sort_values(by = ['OCRDATE'],ascending=False )
    data_frame_sort['OCRDATE'] = pd.to_datetime(data_frame_sort['OCRDATE']).dt.normalize()
    return data_frame_sort

#<------------------------------------------------------------------------->在列表阶段对日期转化为标准的格式
def date_transform(data):
    data = data.sort_values(["account_date"], ascending=1)
    data["account_date"]= pd.to_datetime(data["account_date"].apply(lambda x : x.strftime("%Y-%m-%d")))
    return data

# def date_transform_02(data):
#     data = data.sort_values(["account_date"],ascending = 1)
#     data["account_date"]= pd.to_datetime(data["account_date"].apply(lambda x : x.strftime("%Y-%m-%d")))
#     return data

#-------------------------------------------------------------------------------->门店数量信息需要再分组求和
def shop_cout_agg(data):
    shop_count_data = data.groupby(['OCRDATE'],as_index = False).agg(sum)
    return shop_count_data

#------------------------------------------------------------------------------->以日期作为分组内容查看每天每个SKU的具体的销量
def data_group(data):
    #这里的毛利是门店卖出的总金额与仓库进货的总金额的差值比
    data['gross_profit_rate'] = (data['RTOTAL'] - data['TOTAL']) / data['TOTAL']
    #计算仓库销售的正确单价
    data['price'] = data['PRICE']/ data['QTY']
    #以下是用来保存分组后的数据
    sales_data = pd.DataFrame(columns = ["OCRDATE","GDGID","QTY","price",'gross_profit_rate'])
    sales_data["QTY"]=data.groupby(["OCRDATE"],as_index = False).sum()["QTY"]
    sales_data["price"] = data.groupby(["OCRDATE"],as_index = False).mean()["price"]
    sales_data["gross_profit_rate"] = data.groupby(["OCRDATE"],as_index = False).mean()["gross_profit_rate"]
    sales_data["OCRDATE"]= data.groupby(['OCRDATE']).sum().index
    sales_data["GDGID"] = [data["GDGID"].iloc[0]]*len(sales_data["QTY"])
    sales_data = sales_data.sort_values( by = ['OCRDATE'], ascending = False)
    return sales_data

#---------------------------------------------------------------------------->对日期没有销量和价格等信息进行补齐操作
def date_fill(data,end):#
    yesterday = date_convert(end)
    date_range_sku = pd.date_range(start='20190601', end = yesterday)
    data_sku = pd.DataFrame({'OCRDATE': date_range_sku})
    result = pd.merge(data, data_sku,on=['OCRDATE'],how='right')
    #如果在某一天没有销量的话，采取补零的操作
    result["QTY"].iloc[np.where(np.isnan(result["QTY"]))] = 0
    result = result.fillna(method='ffill')
    result = result.sort_values(["OCRDATE"], ascending=1)
    return result


#----------------------------------------------------------------------->合并含有节假日对应信息的数据到数据集中
def holiday_merge(data,holiday_01,holiday_02):
    merge_data = pd.merge(data,holiday_01,on=['account_date'],how='inner')
    merge_data = pd.merge(merge_data,holiday_02,on=['account_date'],how='inner')
    return merge_data

#---------------------------------------------------------------------------->销量明细与门店数量的合并
def sales_shop(sales_data,shop_data):
    merge_data = pd.merge(sales_data,shop_data,on=['OCRDATE'],how='left')
    return merge_data

#-------------------------------------------------------------------------->统合操作数据清洗后的所有可用的需求预测数据
def all_pre_data(start_date,end_date):
    sku_id = read_oracle_data(start_date,end_date)
    good_id = set(sku_id)
    final_forecast = []
    for i in tqdm(good_id):
        sales_data = get_detail_sales_data(i,start_date,end_date)
        shop_count = sku_shops_count(i,start_date,end_date)
        # sales_data = date_normalize(sales_data)
        # shop_count = date_normalize(shop_count)
        shop_count = shop_cout_agg(shop_count)
        sales_group = data_group(sales_data)
        merge_data = sales_shop(sales_group,shop_count)
        sales_shop_data =  date_fill(merge_data,end_date)
        sales_shop_data = sales_shop_data.reset_index(drop=True, inplace=False)

        sales_shop_data = sales_shop_data.rename(columns = {'OCRDATE':'account_date','GDGID':'sku_id',
                                              'QTY':'sales_qty','PRICE':'price','CLIENT':'shop_num'})
        final_forecast.append(sales_shop_data)

    return final_forecast


#------------------------------------------------------------------->构建时间特征，定义打标签
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
#
# def count_num(num):


#<--------------------------------------------------------------------------------------添加节假日期的特征因素
#holiday的包只支持到2018年，等待最新可用的包
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
    # x['count_num'] = x['shop_num'].apply(lambda x: )
    # x["holiday"] = x["account_date"].apply(lambda x: holiday_day(x))
    # x["holiday_index"] = [1 if a == True else 0 for a in x["holiday"]]
    # del x["holiday"]
    return x


#--------------------------------------------------------------------->构建测试时间范围与基本特征
def creat_test(end):
    #先将string日期转成pd的时间格式的日期
    start = pd.to_datetime(datetime. datetime.strptime(end,'%Y%m%d'))
    date_prediction = pd.date_range(start=start,periods=35)
    date_dataframe = pd.DataFrame({"account_date":date_prediction})
    test = time_subset(date_dataframe)
    return test

#--------------------------------------------------------------------->计算数据的峰度
def kurtosis_compute(data):
    data_mean = np.mean(data)
    data_var = np.var(data)+0.1
    data_sc = np.mean((data - data_mean) ** 3)
    data_ku = np.mean((data - data_mean) ** 4) / pow(data_var, 2)  # 计算峰度
    period2_of_month(data_ku)
    return data_ku

#-------------------------------------------------------------------->构建数据进货门店数量的特征
def shop_count(data):
    shop_num = np.mean(data[['shop_num']])
    print(shop_num)
    return shop_num

# <-------------------------------------------------------------->构建每一个时间分布上的特征
def time_agg(train,test_df,vars_to_agg,vars_be_agg):  # 构建时间特征与峰度
    for var in vars_to_agg:
        agg = train.groupby(var)[vars_be_agg].agg(["sum", "mean", "std", "skew", "median", "min", "max","count",
                                                   kurtosis_compute])
        if isinstance(var, list):
            agg.columns = pd.Index(["fare_by_" + "_".join(var) + "_" + str(e) for e in agg.columns.tolist()])
        else:
            agg.columns = pd.Index(["fare_by_" + var + "_" + str(e) for e in agg.columns.tolist()])
        train = pd.merge(train, agg.reset_index(), on=var, how = "left")
        test_df = pd.merge(test_df, agg.reset_index(), on=var, how = "left")
    return train, test_df


#<---------------------------------------------------------->添加所要所要预测的距离该sku第一次售卖的时间的长度,权重
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


#<--------------------------------------------------------------------------->构建随机森林模型
def construct_randomforest_model(train_feature,train_targe,test):
    rf = RandomForestRegressor(n_estimators=300, max_features=10, max_depth=8, oob_score=True)#<---------参数有待调整
    rf.fit(train_feature,train_targe)
    result = rf.predict(test)
    return result

#------------------------------------------------------------------------------>构建KNN模型
def construct_KNN_model(train_feature,train_targe,test):
    knn_model = KNeighborsRegressor(n_neighbors=10, leaf_size=13, n_jobs=-1)#<--------------参数有待调整
    knn_model.fit(train_feature,train_targe)
    knn_test_pre = knn_model.predict(test)
    return  knn_test_pre


#------------------------------------------------------------------------------->对输出的结果进行规范化
def prediction_result_Regularization(train,result,test,date_parameter):
    prediction_sales = result
    prediction_date = test["account_date"]
    sku_id = [np.unique(train["sku_id"])[0]]*35
    prediction_date_algorithm = [date_parameter]*35
    prediction_df = pd.DataFrame({"sku_id":sku_id,
                                 "account_date":prediction_date,
                                  "forecast_qty":prediction_sales,
                                  "belonged_date":prediction_date_algorithm})
    return prediction_df

#------------------------------------------------------------------------------->得到特征的输出结果
def prediction_feature_targe(sku_i,end):
    train = time_subset(sku_i)
    train = holiday_merge(train,holiday_2018,holiday_2019)
    test = creat_test(end)
    test = holiday_merge(test, holiday_2018, holiday_2019)
    vars_be_agg = "sales_qty"
    vars_to_agg = ["dayofweek", "weekofyear", "month", "day", "year", "period_of_month", "period2_of_month",
                   "week_of_month", "quarter",["month", "dayofweek"], ["quarter", "month"],
                   'weekday','chinese_festival','solar_festival','term_festival','lunar_festival']
    data = time_agg(train, test, vars_to_agg, vars_be_agg)
    train_feature_data = data[0].fillna(0)
    test_feature_data = data[1].fillna(0)
    train_test_data = add_time_diff(train_feature_data,test_feature_data)
    train_feature_data_result = train_test_data[0]
    print(train_feature_data_result)
    test_feature_data_result = train_test_data[1]
    train_feature_data = train_feature_data_result.drop(["account_date","sku_id","sales_qty",'price',"gross_profit_rate",'shop_num'],axis=1)
    prediction_feature = list(train_feature_data.columns)
    prediction_target = ["sales_qty"]
    return train,test,train_feature_data_result,test_feature_data_result,prediction_feature,prediction_target


#<------------------------------------------------------------------------>构建的随机森林模型得到最终的结果
def prediction_sku_sales_RandomForestRegressor_model(sku_i,date_parameter,end):
    prediction_feature_targe_data = prediction_feature_targe(sku_i,end)
    train = prediction_feature_targe_data[0]
    test = prediction_feature_targe_data[1]
    train_feature_data_result = prediction_feature_targe_data[2]
    test_feature_data_result = prediction_feature_targe_data[3]
    prediction_feature = prediction_feature_targe_data[4]
    prediction_target = prediction_feature_targe_data[5]
    # print('--------------------')
    # print('test_feature_data_result')
    # print(test_feature_data_result)
    # print('--------------------')
    # print('prediction_feature')
    # print(prediction_feature)
    prediction_result = list(construct_randomforest_model(train_feature_data_result[prediction_feature],
                                                          train_feature_data_result[prediction_target],
                                                          test_feature_data_result[prediction_feature]))
    result = prediction_result_Regularization(train,prediction_result,test,date_parameter)
    return result


#<------------------------------------------------------->构建的KNN模型得到最终的结果
def prediction_sku_sales_knn_model(sku_i,date_parameter,end):
    prediction_feature_targe_data = prediction_feature_targe(sku_i,end)
    train = prediction_feature_targe_data[0]
    test = prediction_feature_targe_data[1]
    train_feature_data_result = prediction_feature_targe_data[2]
    test_feature_data_result = prediction_feature_targe_data[3]
    prediction_feature = prediction_feature_targe_data[4]
    prediction_target = prediction_feature_targe_data[5]
    prediction_result = list(itertools.chain.from_iterable(construct_KNN_model(train_feature_data_result[prediction_feature],
                                                train_feature_data_result[prediction_target],
                                                test_feature_data_result[prediction_feature])))
    result = prediction_result_Regularization(train,prediction_result,test,date_parameter)
    return result

#<------------------------------------------------------------------>进行模型的融合
def RF_KNN_model_merge(sku_i,date_parameter,end):
    RF_result = prediction_sku_sales_RandomForestRegressor_model(sku_i,date_parameter,end)
    knn_result = prediction_sku_sales_knn_model(sku_i,date_parameter,end)
    RF_knn_merge_result = RF_result.copy()
    RF_knn_merge_result["forecast_qty"] = 0.5*(RF_result["forecast_qty"]+knn_result["forecast_qty"])
    return RF_knn_merge_result


#<--------------------------------------------------------------------------->定义一个空的数据框
def empty_dataframe():
    data = pd.DataFrame(columns = ["account_date",
                                   "forecast_qty","sku_id","belonged_date"])
    return data

#<------------------------------------------------------------------>最后预测
def prediction_result(end,data):
    date_parameter = date_parameter_read(end)
    final = pd.DataFrame(columns=['belonged_date',])
    result_forecast = [RF_KNN_model_merge(x,date_parameter,end) for x in data]
    result_forecast = pd.concat(result_forecast).reset_index(drop=True)
    print("鲜丰水果预测成功")

    return result_forecast

#<----------------------------------------------------------------------->规整化数据
def Consolidation_data(data):
    data_result = pd.DataFrame({"belonged_date":data["belonged_date"],
                                "account_date":data["account_date"],
                                "forecast_qty":data["forecast_qty"],
                                "sku_id":data["sku_id"]})
    data_result["belonged_date"] = data_result["belonged_date"].apply(lambda x: x.strftime("%Y-%m-%d"))
    data_result["account_date"] = data_result["account_date"].apply(lambda x: x.strftime("%Y-%m-%d"))
    return data_result


#------------------------------------------------------------------------>设置总函数
def main_function(start_date,end_date):
    sku_list = read_oracle_data(start_date,end_date)
    data = all_pre_data(start_date,end_date)
    final_data = prediction_result(end_date,data)
    print(final_data)
    final_data = Consolidation_data(final_data)
    return final_data


    # result_forecast = prediction_result(end_date,data)
    # print(result_forecast)
    # final_data =  Consolidation_data(result_forecast)
    # print(final_data)


start_date = '20190601'
end_date = '20190701'
# final_forecast = all_pre_data(start_date,end_date)
# print(final_forecast)
final = main_function(start_date,end_date)
# final.to_csv('D:/jimmy-ye/AI_supply_chain/forecast/final.csv',encoding='utf_8_sig')
print(final)
# 保存
# filename = open('D:/jimmy-ye/AI_supply_chain/forecast/forecast_1st.txt', 'w')
# for value in final:
#      filename.write(str(value))
# filename.close()









