# -*- coding: utf-8 -*-
# @Time    : 2019/9/13 11:16
# @Author  : Ye Jinyu__jimmy
# @File    : XGBOOST_foreacst

import pandas as pd
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
from sklearn import preprocessing
import numpy as np
import time
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,train_test_split
import features_engineering
import cx_Oracle
import datetime
import pymysql
import tqdm
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print(
            "----生成新的文件目录----")
    else:
        print(
            "当前文件夹已经存在")


def print_in_log(string):
    print(string)
    file = open('./' + 'log_old.txt', 'a')
    file.write(str(string) + '\n')



#========================================================================
#------------------------------------------------------------------>根据SKU 的id来获取每个SKU的具体的销售明细数据
def get_detail_sales_data(sku_id,start_date,end_date,DC_CODE):
    print_in_log('连接到mysql服务器...，正在读取销售数据')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123", port=3306, charset='utf8')
    # 查看出货详细单的数据
    stkout_detail_sql = """SELECT * FROM sales_his sh  WHERE sh.Ocrdate >= DATE('%s') 
AND sh.Ocrdate <= DATE('%s') AND sh.GDGID = %s AND sh.SENDER= %s""" % \
                        (start_date,end_date,sku_id,DC_CODE)
    db.cursor()
    read_orignal_forecast = pd.read_sql(stkout_detail_sql, db)
    read_orignal_forecast.columns = ['SENDER','DC_NAME','WRH','WAREHOUSE_NAME','NUM','GDGID','SKU_NAME',
                                     'OCRDATE','CRTOTAL','MUNIT','QTY','QTYSTR','TOTAL','PRICE','QPC','RTOTAL']
    db.close()
    print_in_log(str(sku_id)+'销售数据读取完成')
    return read_orignal_forecast

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


#--------------------------------------------------------------------------------->对日期进行转化返回string前一天的日期
def date_convert(end):
    datetime_forma= datetime.datetime.strptime(end, "%Y%m%d")
    yesterday = datetime_forma - datetime.timedelta(days=1)
    yesterday = yesterday.strftime("%Y%m%d")
    return yesterday



#---------------------------------设置函数用于对异常点的处理
def process_abnormal(data):
    mid_data = data
    Q1 = mid_data['Sales_qty'].quantile(q=0.25)
    Q3 = mid_data['Sales_qty'].quantile(q=0.75)
    IQR = Q3 - Q1
    mid_data["Sales_qty"].iloc[np.where(mid_data["Sales_qty"] > Q3 + 1.5 * IQR)] =  np.median(mid_data['Sales_qty'])
    mid_data["Sales_qty"].iloc[np.where(mid_data["Sales_qty"] < Q1 - 1.5 * IQR)] = np.median(mid_data['Sales_qty'])
    return mid_data

'''采用三次指数平滑的方式进行数据噪声处理'''
def sigle_holt_winters(data):
    #先修正那些明显的错误的数据
    # data["Sales_qty"].iloc[np.where(data["Sales_qty"] < 0) ] = 0

    sales = data.drop(data[data.Sales_qty <= 0].index)
    # sales.to_csv('D:/AI/xianfengsg/online/V_1.2/data/sale.csv', encoding='utf_8_sig')
    y = pd.Series(sales['Sales_qty'].values)
    date = pd.Series(sales['Account_date'].values)
    seaonal = round(len(sales) / 4)

    ets3 = ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=seaonal)
    r3 = ets3.fit()
    anomaly_data = pd.DataFrame({
        'Account_date': date,
        'fitted': r3.fittedvalues,
    })
    merge_data = pd.merge(data,anomaly_data,on='Account_date',how='inner')
    # merge_data.to_csv('D:/AI/xianfengsg/online/V_1.2/data/merge_data.csv', encoding='utf_8_sig')
    merge_data.drop('Sales_qty',axis=1, inplace=True)
    merge_data = merge_data.rename(columns={'fitted':'Sales_qty'})
    return merge_data

# def holt_winters(data):



#----------------------------------------------------->对日期没有销量和价格等信息进行补齐操作,并做异常值的处理
def date_fill(start_date, end, data):
    yesterday = date_convert(end)
    date_range_sku = pd.date_range(start=start_date, end=yesterday)
    data_sku = pd.DataFrame({'Account_date': date_range_sku})

    #采用三阶指数平滑的方式处理
    '''此处需要设置判断的采用两种噪声处理方式'''
    # if len(data) >= 90:
    #     print('使用三阶指数平滑处理')
    #     process_data = sigle_holt_winters(data)
    # else:
    print('基于统计学处理')
    process_data = process_abnormal(data)
    process_data['Sales_qty'] = process_data['Sales_qty'].astype(int)
    result = pd.merge(process_data, data_sku, on=['Account_date'], how='right')
    # 如果在某一天没有销量的话，采取补零的操作
    result["Sales_qty"].iloc[np.where(np.isnan(result["Sales_qty"]))] = 0
    result["Sales_qty"].iloc[np.where(result["Sales_qty"] < 0)] = 0
    result = result.fillna(method='ffill')
    result = result.sort_values(["Account_date"], ascending=1)
    return result


# 获取所有的SKU的的gid
def get_all_sku(data):
    data.to_csv('./sku_convert_rate.csv',encoding='utf_8_sig')
    sku_columns = set(data["Sku_id"])
    sku_list = list(sku_columns)
    return sku_list

def get_features_target(data):
    data_array = pd.np.array(data)  # 传入dataframe，为了遍历，先转为array
    features_list = []
    target_list = []
    columns = [column for column in data]
    print_in_log('该资源的长度是:'+str(len(columns)))
    if 'Sales_qty' in columns:
        for line in data_array:
            temp_list = []
            for i in range(1, int(data.shape[1])):  # 一共有384个特征
                if i == 2:  # index=2对应的当前的目标值，也就是当下的销售量
                    target_temp = int(line[i])
                    target_list.append(target_temp)
                else:
                    temp_list.append(int(line[i]))
            features_list.append(temp_list)
    else:
        for line in data_array:
            temp_list = []
            target_list =[]
            for i in range(1, int(data.shape[1])):  # 一共有107个特征
                temp_list.append(int(line[i]))
            features_list.append(temp_list)
    return pd.DataFrame(features_list), pd.DataFrame(target_list)

def get_sku_number_dict(data):
    data_array = pd.np.array(data)
    max_dict = {}
    min_dict = {}
    ave_dict = {}
    sum_dict = {}
    count_dict = {}
    all_sku_list = []
    for line in data_array:
        all_sku_list.append(line[1])
    all_sku_id_set = set(all_sku_list)
    for sku in all_sku_id_set:
        max_dict[sku] = 0
        min_dict[sku] = 0
        ave_dict[sku] = 0
        sum_dict[sku] = 0
        count_dict[sku] = 0
    for line in data_array:
        sales_qty = line[2]
        sku = line[1]
        sum_dict[sku] += sales_qty
        count_dict[sku] += 1
        #获取最大最小的销量
        if max_dict[sku] < sales_qty:
            max_dict[sku] = sales_qty
        if min_dict[sku] > sales_qty:
            min_dict[sku] = sales_qty
    for sku in all_sku_id_set:
        ave_dict[sku] = sum_dict[sku] / count_dict[sku]
    return max_dict, min_dict, ave_dict


# 得到评价指标rmspe_xg训练模型
def rmspe_xg(yhat, y):
    # y DMatrix对象
    y = y.get_label()
    # y.get_label 二维数组
    y = np.exp(y)  # 二维数组
    yhat = np.exp(yhat)  # 一维数组
    rmspe = np.sqrt(np.mean((y - yhat) ** 2))
    return "rmspe", rmspe


# 该评价指标用来评价模型好坏
def rmspe(zip_list):
    sum_value = 0.0
    count = len(list(zip_list))
    for real, predict in zip_list:
        v1 = (real - predict) ** 2
        sum_value += v1
    v2 = sum_value / count
    v3 = np.sqrt(v2)
    return v3


def predict_with_XGBoosting(data,test_data):
    data_process = data
    data_process.ix[data_process['Sales_qty'] == 0, 'Sales_qty'] = 1
    train_and_valid, test = train_test_split(data_process, test_size=0.2, random_state=10)
    train, valid = train_test_split(train_and_valid, test_size=0.1, random_state=10)
    train_feature, train_target = get_features_target(train)
    test_feature, test_target = get_features_target(test)
    valid_feature, valid_target = get_features_target(valid)


    #--------------------------------------------------
    dtrain = xgb.DMatrix(train_feature, np.log(train_target))  # 取log是为了数据更稳定
    dvalid = xgb.DMatrix(valid_feature, np.log(valid_target))
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]

    # 设置参数
    num_trees = 450
    params = {"objective": "reg:linear",
              "eta": 0.15,
              "max_depth": 8,
              "subsample": 0.8,
              "colsample_bytree": 0.7,
              "silent": 1
              }

    # 训练模型
    gbm = xgb.train(params, dtrain, num_trees, evals=watchlist,
                    early_stopping_rounds=60, feval=rmspe_xg, verbose_eval=True)

    # 获取sku_id_list
    sku_id_list = []
    test_feature_array = pd.np.array(test_feature)
    for line in test_feature_array:
        sku_id_list.append(line[0])
    #---------------------------------------------------
    #用于最后的输出结果进行切分
    print_in_log('test_data:length'+str(len(test_data)))
    predict_feature,target_empty = get_features_target(test_data)
    # 将测试集代入模型进行预测
    print("Make predictions on the future set",'predict_feature')
    predict_probs = gbm.predict(xgb.DMatrix(predict_feature))
    predict_qty = list(np.exp(predict_probs))
    list_zip_id_real_predict = zip(sku_id_list, predict_qty)

    # 对预测结果进行矫正
    max_dict, min_dict, ave_dict = get_sku_number_dict(data_process)
    predict_qty_improve = []
    for sku_id, predict in list_zip_id_real_predict:
        # print shop_id,real,predict,max_dict[shop_id],min_dict[shop_id],ave_dict[shop_id]
        sku_id = str(sku_id)
        print_in_log('predict'+str(predict))
        print_in_log('sku_id'+str(sku_id))
        print_in_log('max_dict'+str(max_dict))
        print_in_log('min_dict' + str(min_dict))
        if predict > max_dict[sku_id]:
            predict = ave_dict[sku_id]
        if predict < min_dict[sku_id]:
            predict = ave_dict[sku_id]
        predict_qty_improve.append(predict)
    # 计算误差
    # list_zip_real_predict_improve = zip(test_target_list, predict_qty_improve)
    # error = rmspe(list_zip_real_predict_improve)
    # print('error', error)
    return predict_qty_improve



#定义一个函数用来切构建用于训练的数据集
def separate_data(sales_data,feature_data):
    # 以下这部操作是配送中心和sku进行区分，同时将特征信息加入进行合并
    sales_data = sales_data[['Account_date', 'Sku_id', 'Sales_qty']]
    merge_data = sales_data.merge(feature_data, on='Account_date', how='inner')
    merge_data = merge_data.reset_index(drop=True)
    return merge_data


#构建一个用来预测的数据集
def create_future(sku_id,features_data,end_date):
    sku_feature_data = features_data[features_data['Account_date'] >= end_date]
    # sku_feature_data = sku_feature_data.drop(['index'],axis=1)
    # sku_feature_data = sku_feature_data[sku_feature_data['Account_date'] < '2019-06-22']
    sku_feature_data.insert(1,'Sku_id',pd.np.array(sku_id))
    return sku_feature_data


#结果计算后再定义一个函数用来将信息不全
def fill_data(predict_data,sales_data,sku_id):
    mid_sales_data = sales_data[sales_data['Sku_id'] == sku_id]
    mid_predict = predict_data
    mid_predict["Price"] = pd.np.array(mid_sales_data["Price"].iloc[0])
    mid_predict["Gross_profit_rate"] = pd.np.array(mid_sales_data["Gross_profit_rate"].iloc[0])
    mid_predict["Dc_name"] = pd.np.array(mid_sales_data["Dc_name"].iloc[0])
    mid_predict["Dc_code"] = pd.np.array(mid_sales_data["Dc_code"].iloc[0])
    mid_predict["Munit"] = pd.np.array(mid_sales_data["Munit"].iloc[0])
    mid_predict["Wrh"] = pd.np.array(mid_sales_data["Wrh"].iloc[0])
    mid_predict["Warehouse_name"] = pd.np.array(mid_sales_data["Warehouse_name"].iloc[0])
    mid_predict["Sku_name"] = pd.np.array(mid_sales_data["Sku_name"].iloc[0])
    return mid_predict


#--------------------------------------------------------------将五位码转成7位码
def get_7th_code(i):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    #读取的数据包括销量的时间序列，天气和活动信息
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", parameters)
    #查看详细的出库数据，进行了日期的筛选，查看销量签50名的SKU
    goods_sql = """SELECT * FROM GOODSH g WHERE g.CODE IN %s""" %(i,)
    goods = pd.read_sql(goods_sql, conn)
    conn.close
    sku_id = goods['GID'].to_list()
    return sku_id



#设置预测的主函数

#这是从库存表中进行商品的选择,选择需要预测的sku的GID
#定义函数从Oracle数据库里面选择每日的库存数据
def get_stock(end_date):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    #读取的数据包括销量的时间序列，天气和活动信息
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", parameters)
    #查看详细的出库数据，进行了日期的筛选，查看销量签50名的SKU
    goods_sql = """SELECT PRODUCT_POSITIONING, GOODS_CODE, GOODS_NAME, DIFFERENCE_QTY, 
    FILDATE, WAREHOUSE, UP_ID, UP_TIME, INVENTORY FROM DC_hangzhou_inv 
    WHERE FILDATE =  to_date('%s','yyyy-mm-dd')""" %(end_date)
    goods = pd.read_sql(goods_sql, conn)
    goods.dropna(axis=0, how='any', inplace=True)
    print_in_log('goods:len'+str(len(goods)))
    conn.close
    return goods

#-------------最新的逻辑是从叫货目录进行选择
def get_order_code():
    print_in_log('正在读取叫货目录的数据')
    dbconn = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com", database="purchare_sys",
                             user="purchare_sys", password="purchare_sys@123", port=3306,
                             charset='utf8')
    get_orders = """SELECT pcr.goods_code GOODS_CODE FROM p_call_record pcr WHERE pcr.warehouse_id ='1000255'"""
    orders = pd.read_sql(get_orders, dbconn)
    orders['GOODS_CODE'] = orders['GOODS_CODE'].astype(str)
    print_in_log('叫货目录读取完成')
    dbconn.close()
    return orders

def get_his_sales(df,start_date,end_date):
    sku_code = df[['GOODS_CODE']]
    '''只用于测试使用'''
    # sku_code = sku_code.loc[0:2]
    sku_code = sku_code.dropna(axis=0,how='any')
    sku_id_list = sku_code['GOODS_CODE'].to_list()
    gid_tuple = tuple(sku_id_list)
    print_in_log('gid_tuple:'+str(gid_tuple))
    sku_id = get_7th_code(gid_tuple)
    #data为获取的所有需要预测的sku的历史销售数据
    data = pd.DataFrame()
    for i in sku_id:
        stkout_detail = get_detail_sales_data(i,start_date,end_date,1000255)
        print_in_log('sku的id:'+str(i))
        if len(stkout_detail) < 30:
            print_in_log('sku的id：'+str(i)+'未获取到销售数据')
            pass
        else:
            stkout_detail.to_csv('./stkout_detail' + str(i) + '.csv', encoding='utf_8_sig')
            result_mid = data_group(stkout_detail)
            result_mid.to_csv('./result_mid'+str(i)+'.csv',encoding='utf_8_sig')
            result = date_fill(start_date,end_date,result_mid)
            result.to_csv('./result' + str(i) + '.csv', encoding='utf_8_sig')
            data = data.append(result)
    return data


def main_forecast(start_date,end_date):
    df = get_order_code()
    sales_data = get_his_sales(df,start_date,end_date)
    '''这里需要对读取的销售信息再进行一次数据清洗，'''
    features_data = features_engineering.made_feature()
    train_data = separate_data(sales_data,features_data)
    sku_list =get_all_sku(sales_data)
    result = pd.DataFrame()
    #分别对每个sku进行学习预测
    print_in_log('分别对每个sku进行学习预测:'+str(set(sku_list)))
    for sku_id in sku_list:
        print_in_log('sku_id:'+str(sku_id))
    # sku_id = 3000174
        train_data_mid = train_data[train_data['Sku_id']==sku_id]
        test_data = create_future(sku_id,features_data,end_date)
        train_data_mid = train_data_mid.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)
        predict_qty_improve = predict_with_XGBoosting(train_data_mid,test_data)
        test_data['Forecast_qty'] = pd.Series(predict_qty_improve)
        predict_data = test_data[['Account_date','Sku_id','Forecast_qty']]
        result_data = fill_data(predict_data,sales_data,sku_id)
        print('predict_qty_improve',predict_qty_improve)
        result = result.append(result_data)
    result['Update_time'] = datetime.date.today().strftime('%Y-%m-%d')
    result['Account_date']= pd.to_datetime(result['Account_date'], unit='s').dt.strftime('%Y-%m-%d')
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.fillna(0)
    return result


def connectdb():
    print_in_log('连接到mysql服务器...')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123",port=3306, charset='utf8')
    print_in_log('连接成功')
    return db


#《-------------------------------------------------------------------------------------删除重复日期数据
def drop_data(db):
    cursor = db.cursor()
    date_parameter = datetime.date.today().strftime('%Y-%m-%d')
    sql = """delete from dc_forecast where Update_time = DATE ('%s')"""%(date_parameter)
    cursor.execute(sql)

#<======================================================================================================================
def insertdb(db,data):
    cursor = db.cursor()
    # param = list(map(tuple, np.array(data).tolist()))
    data_list = data.values.tolist()
    print_in_log('data_list:len'+str(len(data_list)))
    sql = """INSERT INTO dc_forecast (Account_date,Sku_id,Forecast_qty,
    Price,Gross_profit_rate,Dc_name,Dc_code,Munit,Wrh,Warehouse_name,
    Sku_name,Update_time)
     VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    try:
        cursor.executemany(sql, data_list)
        print_in_log("所有品牌的sku数据插入数据库成功")
        db.commit()
    except OSError as reason:
        print_in_log('出错原因是%s' % str(reason))
        db.rollback()
#<=============================================================================
def closedb(db):
    db.close()


#<=============================================================================
def main(today_test):
    date = datetime.datetime.strptime(today_test, "%Y-%m-%d")
    end_date = date.strftime('%Y%m%d')
    mkdir(end_date)
    start_date = (datetime.date.today() - datetime.timedelta(800)).strftime('%Y%m%d')
    result_forecast = main_forecast(start_date,end_date)
    print_in_log('result_forecast:len'+str(len(result_forecast)))
    result_forecast.to_csv('D:/AI/xianfengsg/online/V_1.2/'+str(end_date)+'.csv',encoding='utf_8_sig')
    return result_forecast
    # db = connectdb()
    # drop_data(db)
    # if result_forecast.empty:
    #     print_in_log("The data frame is empty")
    #     print_in_log("result:1")
    #     closedb(db)
    # else:
    #     insertdb(db,result_forecast)
    #     closedb(db)
    #     print_in_log("result:1")

#《============================================================================主函数入口
if __name__ == '__main__':
    try:
        today_test = '2019-11-15'
        result_forecast = main(today_test)
    except OSError as reason:
        print_in_log('出错原因是%s'%str(reason))
        print_in_log ("result:0")

