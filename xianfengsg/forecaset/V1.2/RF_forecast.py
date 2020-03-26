# -*- coding: utf-8 -*-
# @Time    : 2019/7/22 15:58
# @Author  : Ye Jinyu__jimmy
# @File    : RF_forecast.py

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
import time

# import chinese_calendar as calendar  #
warnings.filterwarnings("ignore")
import get_holiday_inf_2019
import get_holiday_inf_2018

holiday_2019 = get_holiday_inf_2019.main()
holiday_2019 = holiday_2019[1].fillna(0).reindex()
holiday_2018 = get_holiday_inf_2018.main()
holiday_2018 = holiday_2018[1].fillna(0).reindex()
holiday_2019['Account_date'] = pd.to_datetime(holiday_2019['Account_date'], format='%Y-%m-%d', errors='ignore')
holiday_2018['Account_date'] = pd.to_datetime(holiday_2018['Account_date'], format='%Y-%m-%d', errors='ignore')

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
def read_oracle_data(start_date,end_date,i):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    #读取的数据包括销量的时间序列，天气和活动信息
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", parameters)
    #查看详细的出库数据，进行了日期的筛选，查看销量签50名的SKU
    stkout_detail_sql = """  SELECT sum(b.qty),b.GDGID
                            FROM stkout ss ,stkoutdtl b, stkoutlog c,store s ,goods g,warehouseh wrh
                            where ss.num = b.num  and ss.num =c.num and b.gdgid=g.gid and ss.sender =s.gid
                            and ss.cls='统配出' and ss.cls=c.cls and ss.cls=b.cls and ss.wrh = wrh.gid
                            and c.stat IN ('700','720','740','320','340')
                            AND wrh.NAME LIKE'%%商品仓%%'  
                            AND ss.SENDER= %s
                            and c.time>=  to_date('%s','yyyy-mm-dd')
                            and c.time <  to_date('%s','yyyy-mm-dd')
                            GROUP BY b.GDGID order by sum(b.QTY) DESC""" %(i,start_date,end_date)
    GDGID_sales = pd.read_sql(stkout_detail_sql, conn)
    #将SKU的的iD转成list，并保存前50个，再返回值
    conn.close
    sku_id = GDGID_sales['GDGID'].tolist()
    return sku_id


#------------------------------------------------------------------>根据SKU 的id来获取每个SKU的具体的销售明细数据
def get_detail_sales_data(sku_id,start_date,end_date,DC_CODE):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    dsn = cx_Oracle.makedsn(host, port, sid)

    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码)
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", dsn)
    # 查看出货详细单的数据
    stkout_detail_sql = """SELECT ss.sender,s.name Dc_name      
                                ,wrh.GID as WRH,wrh.NAME warehouse_name
                                ,ss.num ,b.gdgid,G.NAME sku_name
                                ,trunc(c.time) OCRDATE,b.CRTOTAL,b.munit,b.qty,b.QTYSTR
                                ,b.TOTAL,b.price,b.qpc,b.RTOTAL
                               FROM stkout ss ,stkoutdtl b, stkoutlog c,store s ,goods g,warehouseh wrh
                                where ss.num = b.num  and ss.num =c.num 
                                and b.gdgid=g.gid and ss.sender =s.gid
                                and ss.cls='统配出' and ss.cls=c.cls and ss.cls=b.cls and ss.wrh = wrh.gid
                                and c.stat IN ('700','720','740','320','340')
                                and c.time>=  to_date('%s','yyyy-mm-dd')
                                and c.time <  to_date('%s','yyyy-mm-dd')
                                and b.GDGID = %s 
                                AND wrh.NAME LIKE'%%商品仓%%'  AND ss.SENDER= %s""" % \
                        (start_date,end_date,sku_id,DC_CODE)
    stkout_detail = pd.read_sql(stkout_detail_sql, conn)
    conn.close
    return stkout_detail

# #---------------------------------------------------------------------------->按照不同的
def diff_DC(n):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    dsn = cx_Oracle.makedsn(host, port, sid)

    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码)
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", dsn)
    # 查看出货详细单的数据
    DC = """select s.SENDER,COUNT(s.SENDER) from STKOUT s INNER JOIN STORE s1  ON s.sender = s1.gid
                            INNER JOIN(SELECT * FROM WAREHOUSE w WHERE w.NAME LIKE'%%商品仓%%' )w
                            ON w.STOREGID = s1.gid
                            INNER JOIN STORE s2 ON s.CLIENT = s2.gid
                            WHERE bitand(s1.property,32)=32
                            AND bitand(s2.property,32)<>32
                            AND substr(s2.AREA,2,3)<'8000'
                            AND s.CLS='统配出'
                            GROUP BY s.SENDER order by count(s.SENDER) DESC"""
    DC_detail = pd.read_sql(DC, conn)
    conn.close
    DC_detail = DC_detail['SENDER'].tolist()
    DC_detail = DC_detail[0:n]
#     return DC_detail




#--------------------------------------------------------------------------------->日的标准化转化
def date_normalize(data_frame):
    data_frame_sort = data_frame.sort_values(by = ['OCRDATE'],ascending=False )
    data_frame_sort['OCRDATE'] = pd.to_datetime(data_frame_sort['OCRDATE']).dt.normalize()
    return data_frame_sort

#<------------------------------------------------------------------------->在列表阶段对日期转化为标准的格式
def date_transform(data):
    data = data.sort_values(["Account_date"], ascending=1)
    data["Account_date"]= pd.to_datetime(data["Account_date"].apply(lambda x : x.strftime("%Y-%m-%d")))
    return data

#------------------------------------------------------------------------------->以日期作为分组内容查看每天每个SKU的具体的销量
def data_group(data):
    #这里的毛利是门店卖出的总金额与仓库进货的总金额的差值比
    data['GROSS_PROFIT_RATE'] = (data['RTOTAL'] - data['TOTAL']) / data['TOTAL']
    #计算仓库销售的正确单价
    data['PRICE'] = data['PRICE']/ data['QTY']
    #以下是用来保存分组后的数据
    sales_data = pd.DataFrame(columns = ["Account_date","Sku_id",'Dc_name',"Sales_qty","Price",'Gross_profit_rate','Dc_code',
                                         'Wrh','Warehouse_name','Sku_name','Munit'])
    sales_data["Sales_qty"]=data.groupby(["OCRDATE"],as_index = False).sum()["QTY"]
    sales_data["Price"] = data.groupby(["OCRDATE"],as_index = False).mean()["PRICE"]
    sales_data["Gross_profit_rate"] = data.groupby(["OCRDATE"],as_index = False).mean()["GROSS_PROFIT_RATE"]
    sales_data["Account_date"]= data.groupby(['OCRDATE']).sum().index
    sales_data["Sku_id"] = [data["GDGID"].iloc[0]]*len(sales_data["Sales_qty"])
    sales_data["Dc_name"] = [data["DC_NAME"].iloc[0]] * len(sales_data["Sku_id"])
    sales_data["Dc_code"] = [data["SENDER"].iloc[0]] * len(sales_data["Sku_id"])
    sales_data["Munit"] = [data["MUNIT"].iloc[0]] * len(sales_data["Sales_qty"])
    sales_data["Wrh"] = [data["WRH"].iloc[0]] * len(sales_data["Sales_qty"])
    sales_data["Warehouse_name"] = [data["WAREHOUSE_NAME"].iloc[0]] * len(sales_data["Sales_qty"])
    sales_data["Sku_name"] = [data["SKU_NAME"].iloc[0]] * len(sales_data["Sales_qty"])
    sales_data = sales_data.sort_values( by = ['Account_date'], ascending = False)
    return sales_data

#----------------------------------------------------------------------->合并含有节假日对应信息的数据到数据集中
def holiday_merge(data,holiday_01,holiday_02):
    holiday = pd.concat([holiday_02,holiday_01],join='outer',axis=0)
    merge_data = pd.merge(data,holiday,on=['Account_date'],how='inner')
    return merge_data


#---------------------------------------------------------------------------->对日期没有销量和价格等信息进行补齐操作
def date_fill(data,end):#
    yesterday = date_convert(end)
    date_range_sku = pd.date_range(start='20180101', end = yesterday)
    data_sku = pd.DataFrame({'Account_date': date_range_sku})
    result = pd.merge(data, data_sku,on=['Account_date'],how='right')
    #如果在某一天没有销量的话，采取补零的操作
    result["Sales_qty"].iloc[np.where(np.isnan(result["Sales_qty"]))] = 0
    result = result.fillna(method='ffill')
    result = result.sort_values(["Account_date"], ascending=1)
    return result


def one_hot(data,features):
    # 把带中文的标称属性转换为数值型，因为one-hot编码也需要先转换成数值型，用简单整数代替即可
    data = data[[features]]
    listUniq = data.ix[:, features].unique()
    for j in range(len(listUniq)):
        data.ix[:, features] = data.ix[:, features].apply(lambda x: j if x == listUniq[j] else x)
    # 进行one-hot编码
    # tempdata = data[[features]]
    # enc = preprocessing.OneHotEncoder()
    # enc.fit(tempdata)
    #
    # # one-hot编码的结果是比较奇怪的，最好是先转换成二维数组
    # tempdata = enc.transform(tempdata).toarray()
    # print('取值范围整数个数：', enc.n_values_)
    #
    # # 再将二维数组转换为DataFrame，记得这里会变成多列
    # tempdata = pd.DataFrame(tempdata, columns=[features] * len(tempdata[0]))
    return data


def holiday_features(sales_shop_data):
    sales_shop_data = holiday_merge(sales_shop_data, holiday_2018, holiday_2019)
    tempdata_weekday = one_hot(sales_shop_data, 'Weekday')
    tempdata_solar_festival = one_hot(sales_shop_data, 'Solar_festival')
    tempdata_term_festival = one_hot(sales_shop_data, 'Term_festival')
    tempdata_lunar_festival = one_hot(sales_shop_data, 'Lunar_festival')
    tempdata_chinese_festival = one_hot(sales_shop_data, 'Chinese_festival')
    sales_data = sales_shop_data.drop(
        ['Weekday', 'Chinese_festival', 'Solar_festival', 'Term_festival', 'Lunar_festival'], axis=1)
    original_data = pd.concat(
        [sales_data, tempdata_weekday, tempdata_chinese_festival, tempdata_solar_festival, tempdata_term_festival
            , tempdata_lunar_festival], join='outer', axis=1)
    original_data.to_csv('D:\jimmy-ye\AI_supply_chain\data\original_data.csv',encoding='utf_8_sig')
    return original_data


#-------------------------------------------------------------------------->统合操作数据清洗后的所有可用的需求预测数据
def all_pre_data(good_id,DC_CODE,start_date,end_date):

    final_forecast = []
    for i in tqdm(good_id):
        sales_data = get_detail_sales_data(i,start_date,end_date,DC_CODE)
        #-----------------------------------存在某个仓库的sku在某个时间段并没有销售记录
        if sales_data.empty==True:
            pass
        else:
            sales_group = data_group(sales_data)
            sales_shop_data =  date_fill(sales_group,end_date)
            sales_shop_data = sales_shop_data.reset_index(drop=True, inplace=False)
            sales_shop_data = date_transform(sales_shop_data)
            print('sales_shop_data')
            print(sales_shop_data)
            original_data = holiday_features(sales_shop_data)

            final_forecast.append(original_data)
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


#<---------------------------------------------------------------------->以年月日日期在进行特征构建
def time_subset(x):
    x["dayofweek"] = x['Account_date'].apply(lambda x: x.dayofweek)
    x["weekofyear"] = x["Account_date"].apply(lambda x: x.weekofyear)
    x['month'] = x['Account_date'].apply(lambda x: x.month)
    x['day'] = x['Account_date'].apply(lambda x: x.day)
    x['year'] = x['Account_date'].apply(lambda x: x.year)
    x['period_of_month'] = x['day'].apply(lambda x: period_of_month(x))
    x['period2_of_month'] = x['day'].apply(lambda x: period2_of_month(x))
    x['week_of_month'] = x['day'].apply(lambda x: week_of_month(x))
    x['quarter'] = x['month'].apply(lambda x: quarter(x))
    return x

#--------------------------------------------------------------------->构建测试时间范围与基本特征
def creat_test(end):
    #先将string日期转成pd的时间格式的日期
    start = pd.to_datetime(datetime. datetime.strptime(end,'%Y%m%d'))
    date_prediction = pd.date_range(start=start,periods=35)
    date_dataframe = pd.DataFrame({"Account_date":date_prediction})
    test = time_subset(date_dataframe)
    test = holiday_features(test)
    return test

#--------------------------------------------------------------------->计算数据的峰度
def kurtosis_compute(data):
    data_mean = np.mean(data)
    data_var = np.var(data)+0.1
    data_sc = np.mean((data - data_mean) ** 3)
    data_ku = np.mean((data - data_mean) ** 4) / pow(data_var, 2)  # 计算峰度
    period2_of_month(data_ku)
    return data_ku

# <-------------------------------------------------------------->构建每一个时间分布上的特征
def time_agg(train,test_df,vars_to_agg,vars_be_agg):  # 构建时间特征与峰度
    for var in vars_to_agg:
        print(var)
        print('var')
        agg = train.groupby(var)[vars_be_agg].agg(["sum", "mean", "std", "skew", "median", "min", "max","count",
                                                   kurtosis_compute])
        print('agg')
        print(agg)
        if isinstance(var, list):
            agg.columns = pd.Index(["fare_by_" + "_".join(var) + "_" + str(e) for e in agg.columns.tolist()])
        else:
            agg.columns = pd.Index(["fare_by_" + var + "_" + str(e) for e in agg.columns.tolist()])
        train = pd.merge(train, agg.reset_index(), on=var, how = "left")
        test_df = pd.merge(test_df, agg.reset_index(), on=var, how = "left")
    return train, test_df



#<---------------------------------------------------------->添加所要所要预测的距离该sku第一次售卖的时间的长度,权重
def add_time_diff(train_data,test_data):
    min_date = train_data["Account_date"].min()
    date_train_diff = train_data["Account_date"]-min_date
    date_test_diff = test_data["Account_date"]-min_date
    train_data["date_diff"] = date_train_diff.apply(lambda x:x.days)
    train_data = train_data[train_data["date_diff"]>0]
    train_data["date_diff"] = train_data["date_diff"].apply(lambda x : np.exp(1/x))
    test_data["date_diff"] = date_test_diff.apply(lambda x:x.days)
    test_data = test_data[test_data["date_diff"] > 0]
    test_data["date_diff"] = test_data["date_diff"].apply(lambda x :np.exp(1/x))
    return train_data,test_data


#--------------------------------------------------------------------------->构建随机森林模型
def construct_randomforest_model(train_feature,train_targe,test):
    # rf = RandomForestRegressor(n_estimators=50, max_features=30, max_depth=8, oob_score=True)#<---------参数有待调整
    rf = RandomForestRegressor(n_estimators=300, max_features=50, max_depth=20, oob_score=True)  # <---------参数有待调整
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
    prediction_date = test["Account_date"]
    DC_code = [np.unique(train["Dc_code"])[0]]*35
    Munit = [np.unique(train["Munit"])[0]] * 35
    Dc_name = [np.unique(train["Dc_name"])[0]]*35
    Wrh = [np.unique(train["Wrh"])[0]] * 35
    Warehouse_name = [np.unique(train["Warehouse_name"])[0]]*35
    Sku_name = [np.unique(train["Sku_name"])[0]] * 35
    sku_id = [np.unique(train["Sku_id"])[0]]*35
    prediction_date_algorithm = [date_parameter]*35
    prediction_df = pd.DataFrame({"Sku_id":sku_id,
                                 "Account_date":prediction_date,
                                  "Forecast_qty":prediction_sales,
                                  "Belonged_date":prediction_date_algorithm,
                                  "Dc_code":DC_code,
                                  "Munit":Munit,
                                  "Dc_name": Dc_name,
                                  "Wrh": Wrh,
                                  "Warehouse_name": Warehouse_name,
                                  "Sku_name": Sku_name
                                  })
    return prediction_df

#------------------------------------------------------------------------------->得到特征的输出结果
def prediction_feature_targe(sku_i,end):
    print('sku_i_Data')
    print(sku_i)
    train = time_subset(sku_i)
    test = creat_test(end)
    vars_be_agg = "Sales_qty"
    vars_to_agg = ["dayofweek", "weekofyear", "month", "day", "year", "period_of_month", "period2_of_month",
                   "week_of_month", "quarter",["month", "dayofweek"], ["quarter", "month"],
                   'Weekday','Chinese_festival','Solar_festival','Term_festival','Lunar_festival']
    data = time_agg(train, test, vars_to_agg, vars_be_agg)
    train_feature_data = data[0].fillna(0)
    test_feature_data = data[1].fillna(0)
    train_test_data = add_time_diff(train_feature_data,test_feature_data)
    train_feature_data_result = train_test_data[0]
    test_feature_data_result = train_test_data[1]
    print(train_feature_data_result)
    train_feature_data = train_feature_data_result.drop(["Account_date","Sku_id","Sales_qty",
                                                         'Price',"Gross_profit_rate",'Dc_code',
                                                         'Dc_name','Munit','Wrh','Warehouse_name',
                                                         'Sku_name'],axis=1)
    prediction_feature = list(train_feature_data.columns)
    prediction_target = ["Sales_qty"]
    return train,test,train_feature_data_result,test_feature_data_result,prediction_feature,prediction_target


#<------------------------------------------------------------------------>构建的随机森林模型得到最终的结果
def prediction_sku_sales_RandomForestRegressor_model(sku_i,date_parameter,end):
    prediction_feature_targe_data = prediction_feature_targe(sku_i,end)
    train = prediction_feature_targe_data[0]
    print('train')
    print(train)
    test = prediction_feature_targe_data[1]
    print('test')
    print(test)
    train_feature_data_result = prediction_feature_targe_data[2]
    print('train_feature_data_result')
    print(train_feature_data_result)
    test_feature_data_result = prediction_feature_targe_data[3]
    print('test_feature_data_result')
    print(test_feature_data_result)
    prediction_feature = prediction_feature_targe_data[4]
    print('prediction_feature')
    print(prediction_feature)
    prediction_target = prediction_feature_targe_data[5]
    print('prediction_target')
    print(prediction_target)
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
    RF_knn_merge_result["Forecast_qty"] = 0.5*(RF_result["Forecast_qty"]+knn_result["Forecast_qty"])
    return RF_knn_merge_result


#<--------------------------------------------------------------------------->定义一个空的数据框
def empty_dataframe():
    data = pd.DataFrame(columns = ["account_date",'DC_code'
                                   "forecast_qty","sku_id","belonged_date"])
    return data

#<------------------------------------------------------------------>最后预测
def prediction_result(end,data):
    date_parameter = date_parameter_read(end)
    final = pd.DataFrame(columns=['Belonged_date',])
    result_forecast = [RF_KNN_model_merge(x,date_parameter,end) for x in data]
    result_forecast = pd.concat(result_forecast).reset_index(drop=True)
    print("鲜丰水果预测成功")
    return result_forecast

#<----------------------------------------------------------------------->规整化数据
def Consolidation_data(data):
    data_result = pd.DataFrame({"Belonged_date":data["Belonged_date"],
                                "Account_date":data["Account_date"],
                                "Forecast_qty":data["Forecast_qty"],
                                "Sku_id":data["Sku_id"],
                                "Dc_name": data["Dc_name"],
                                "Dc_code": data["Dc_code"],
                                "Munit": data["Munit"],
                                "Warehouse_name": data["Warehouse_name"],
                                "Sku_name": data["Sku_name"],
                                "Wrh": data["Wrh"]
                                })
    data_result["Belonged_date"] = data_result["Belonged_date"].apply(lambda x: x.strftime("%Y-%m-%d"))
    data_result["Account_date"] = data_result["Account_date"].apply(lambda x: x.strftime("%Y-%m-%d"))
    return data_result


#------------------------------------------------------------------------>设置总函数
def main_function(start_date,end_date,n):
    start_time = time.time()
    DC_detail_list = diff_DC(n)
    print('总共参与预测的配送中心有：',DC_detail_list)
    result_data = pd.DataFrame()
    for i in tqdm(DC_detail_list):
        print('正在进行该配送中心的计算',i)
        sku_id = read_oracle_data(start_date, end_date,i)
        good_id = list(set(sku_id))
        print('总共预测的sku有：', list(good_id))
        data = all_pre_data(good_id,i,start_date,end_date)
        print(str(int(i))+'该配送中心的数据读取完成')
        #------------------------------如果有的配送中心的SKU在某段时间没有销售做一个逻辑判断
        if len(data):
            final_data = prediction_result(end_date, data)
            final_data = Consolidation_data(final_data)
            result_data = result_data.append(final_data)
        else:
            pass
    end_time =time.time()
    total_time = end_time-start_time
    print('程序运行结束，总耗时'+ str(total_time) + '秒')
    return result_data

start_date = '20180101'
end_date = '20190615'
final = main_function(start_date,end_date,5)
final.to_csv('D:/jimmy-ye/AI_supply_chain/data/forecast_holiday/final_holiday_new_parameters.csv',encoding='utf_8_sig')

