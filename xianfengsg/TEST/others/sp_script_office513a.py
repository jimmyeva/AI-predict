# 读取得到数据
import numpy as np
import pandas as pd
from pyhessian.client import HessianProxy
from pyhessian import protocol
from sklearn.ensemble import RandomForestRegressor
from tqdm import *
import itertools
import datetime,time
import os, sys
import pymysql
from sklearn.neighbors import KNeighborsRegressor
import math
import calendar
import warnings
import matplotlib.pyplot as plt
# from chinese_calendar import is_workday, is_holiday
# import chinese_calendar as calendar  #
import multiprocessing
import copy
from tqdm import tqdm

warnings.filterwarnings("ignore")

### 基本参数
timestrtest = '20190423'

activity_mode_list = ['特价','买赠','加购','立减','满送','其他','特价+买赠','乐高', '满减']
activity_mode_eng = ['tejia','maizeng','jiagou','lijian', 'mansong', 'others', 'mai+te','legao','manjian']
predperiods = 35
threshold = 10  ## 样本量阈值

isobserv = True  ## 是否打开对预测的评估指标记录
issvsult = False ## 是否保存预测计算结果
issvefun = False ## 是否保存预测结果的评估指标
isoffice = False ## 是否进入正式环境
iswrite = True  ## 是否写数据到数据库

d = 1
d30y = 30
demand_dura = {'999': d, 'CVS': d, 'DCeC': d, 'HM':d, 'IMDH': d,'LBB': d,'LBT': d,'LMM': d,'LP': d,'LSM': d,'LT': d,'MT': d,'OTHERS': d,'PLBB': d,'PSM': d,'SBB': d,'SBT': d,'SM': d,'SMM': d,'SP': d,'SSM': d,'ST': d,
               'eBTB': d30y, 'HSW':d30y, 'LUH': d30y, 'PSW': d30y,'PPeC': d30y,'ARMSUBD': d30y, 'BO': d30y, 'DS': d30y}

### part0 基本定义和基本函数
class group_data():
    def __init__(self):
        #基本信息
        self.shop_qtysum = None  ## 各门店销售量，字典型
        self.saledata = None     ## 上游节点, dataframe
        self.msg = None          ## 原始数据

        #统计特性

def string2tstamp():
    stpoint = "20170101"
    ststamp = pd.to_datetime(datetime.datetime.strptime(stpoint[0:8], '%Y%m%d'))

    if isoffice:
        mdpoint = sys.argv[1]  ##############################################################   参数修改
    else:
        mdpoint = timestrtest     ## 非正式版时间修改
    mdstamp = pd.to_datetime(datetime.datetime.strptime(mdpoint[0:8], '%Y%m%d'))

    # edpoint = "20190420000000"
    # edstamp = pd.to_datetime(datetime.datetime.strptime(edpoint[0:8], '%Y%m%d'))
    edstamp = mdstamp + datetime.timedelta(days=predperiods)
    return ststamp, mdstamp, edstamp

def print_msg(string):
    print(string)
    # file = open('log.txt', 'a')
    # file.write(string + '\n')

### part 1 数据读取与整理
##########新版本读取数据
class HessianUtil(object):
    def __init__(self, api_url=None):
        self._api_url = api_url
        self._client = None

    def set_api_url(self, api_url):
        self._api_url = api_url

    def _init_hessian_client(self):
        try:
            self._client = HessianProxy(self._api_url, timeout=60, overload=False)
        except Exception as e:
            print(e)
            raise e

    def request(self, method, *param):
        return_result = dict()
        try:
            self._init_hessian_client()
            start_response_time = datetime.datetime.now()
            response = getattr(self._client, method)(*param)
            after_response_time = datetime.datetime.now()
            return_result["time"] = self._time_delta_seconds(start_response_time, after_response_time)
            return_result["code"] = 200
            response = self._deal_response(response)
            if type(response) == str:
                return_result["data"] = response
            else:
                return_result["data"] = response
        except Exception as e:
            after_response_time = datetime.datetime.now()
            return_result["time"] = self._time_delta_seconds(start_response_time, after_response_time)
            print(e)
            return_result["code"] = 500
            return_result["data"] = str(e)
        self._client = None
        return return_result

    def _deal_response(self, response):
        return_result = None
        if self._is_binary(response):
            return_result = response.value.decode("utf-8")
        elif self._is_tuple(response):
            tuple_list = []
            for item in response:
                tuple_list.append(self._deal_response(item))
            return_result = tuple_list
        elif self._is_hessian_object(response):
            object_dict = dict()
            for k, v in response.__dict__.items():
                object_dict[k] = self._deal_response(v)
            return_result = object_dict
        elif self._is_dict(response):
            object_dict = dict()
            for k, v in response.items():
                object_dict[k] = self._deal_response(v)
            return_result = object_dict
        else:
            return_result = response
        return return_result

    def _is_tuple(self, result):
        if type(result) == tuple:
            return True
        else:
            return False

    def _is_dict(self, result):
        if type(result) == dict:
            return True
        else:
            return False

    def _is_binary(self, result):
        if type(result) == type(protocol.Binary("")):
            return True
        else:
            return False

    def _is_hessian_object(self, result):
        try:
            res = type(result)._hessian_factory_args
            return True
        except Exception as e:
            return False

    def _time_delta_seconds(self, start_time, end_time):
        interval = end_time - start_time
        seconds = interval.total_seconds()
        return seconds

def read_sale_dataA(string, s,w):
    hessian_util = HessianUtil("http://192.168.6.165:9106/com.wwwarehouse.xdw.supplychainBi.service.BiService")
    data_sale = hessian_util.request("getBiDataParam",
                                       "{"
                                       "\"selects\": [\"cnt_at\", \"stran_nid\", \"dtran_nid\", \"scirl_nid\", \"dcirl_nid\", \"resource_id\", \"type\", \"out_qty\", ],"
                                       "\"dataSet\": \"sell_data\","
                                       "\"filters\":["
                                       "{\"column\":\"resource_id\",\"type\":\"between\",\"values\":[\"%s\", \"%s\"]},"
                                       "{\"column\":\"type\",\"type\":\"in\",\"values\":[\"999\",\"CVS\",\"DCeC\",\"HM\",\"IMDH\",\"LBB\",\"LBT\",\"LMM\",\"LP\",\"LSM\",\"LT\",\"MT\",\"OTHERS\",\"PLBB\",\"PSM\",\"SBB\", \"SBT\",\"SM\",\"SMM\",\"SP\",\"SSM\",\"ST\"]},"
                                       "{\"column\":\"isallow\",\"type\":\"in\",\"values\":[\"Y\"]},"
                                       "]}" %(s, w))
    origin_data = data_sale["data"]["data"]["dataList"]
    if len(origin_data) == 0:
        print_msg(string + 'A类销售数据为空')
        return None
    else:
        origin_data = pd.DataFrame(origin_data)
        origin_data['cnt_at'] = origin_data.apply(lambda row: row['cnt_at']['value'] + datetime.timedelta(hours=8), axis=1)
        print_msg(string + 'A类销售数据读取成功，数据量为' + str(len(origin_data)) + '条')
        return origin_data

def read_sale_dataB(string, s,w):
    hessian_util = HessianUtil("http://192.168.6.165:9106/com.wwwarehouse.xdw.supplychainBi.service.BiService")
    data_sale = hessian_util.request("getBiDataParam",
                                       "{"
                                       "\"selects\": [\"cnt_at\", \"stran_nid\", \"dtran_nid\", \"scirl_nid\", \"dcirl_nid\", \"resource_id\", \"type\", \"out_qty\", ],"
                                       "\"dataSet\": \"sell_data\","
                                       "\"filters\":["
                                       "{\"column\":\"resource_id\",\"type\":\"between\",\"values\":[\"%s\", \"%s\"]},"
                                       "{\"column\":\"type\",\"type\":\"in\",\"values\":[\"HSW\",\"LUH\",\"PSW\",\"PPeC\",\"ARMSUBD\",\"BO\",\"DS\",\"eBTB\"]},"
                                       "{\"column\":\"isallow\",\"type\":\"in\",\"values\":[\"Y\"]},"
                                       "]}" %(s, w))
    origin_data = data_sale["data"]["data"]["dataList"]
    if len(origin_data) == 0:
        print_msg(string + 'B类销售数据为空')
        return None
    else:
        origin_data = pd.DataFrame(origin_data)
        origin_data['cnt_at'] = origin_data.apply(lambda row: row['cnt_at']['value'] + datetime.timedelta(hours=8), axis=1)
        print_msg(string + 'B类销售数据读取成功，数据量为' + str(len(origin_data)) + '条')
        return origin_data

def read_activity_data(string, s, w):
    def discount_compute(row):
        if type(row['price']) == type({}) and type(row['sprice_tax']) == type({}):
            a = float(row['sprice_tax']['value'])
            b = float(row['price']['value'])
            if b > 0.01:  ## 解决除以0的问题
                discount = a / b
            else:
                discount = 1
            return discount
        else:
            return 1

    hessian_util = HessianUtil("http://192.168.6.165:9106/com.wwwarehouse.xdw.supplychainBi.service.BiService")
    data_activity = hessian_util.request("getBiDataParam",
                                       "{"
                                       "\"selects\": [\"resource_id\", \"stran_nid\", \"price_chmod\", \"price\", \"sprice_tax\", \"effective_st\", \"effective_et\",],"
                                       "\"dataSet\": \"activity_price\","
                                       "\"filters\":["
                                       "{\"column\":\"resource_id\",\"type\":\"between\",\"values\":[\"%s\",\"%s\"]},"
                                       "]"
                                       "}" %(s, w))
    origin_data = data_activity["data"]["data"]["dataList"]

    if len(origin_data) == 0:
        print_msg(string + '活动数据读取成功，数据量为空')
        return None
    else:
        origin_data = pd.DataFrame(origin_data)
        if 'sprice_tax' not in origin_data.keys():
            origin_data['discount'] = 1
        else:
            origin_data['discount'] = origin_data.apply(lambda row: discount_compute(row), axis=1)
        origin_data['effective_st'] = origin_data.apply(lambda row: row['effective_st']['value'] + datetime.timedelta(hours=8), axis=1)
        origin_data['effective_et'] = origin_data.apply(lambda row: row['effective_et']['value'] + datetime.timedelta(hours=8), axis=1)
        origin_data = origin_data[['resource_id', 'stran_nid', 'price_chmod', 'discount', 'effective_st', 'effective_et']]
        print_msg(string + '活动数据读取成功，数据量为' + str(len(origin_data)) + '条')
        return origin_data

def read_sale_relationship():
    hessian_util = HessianUtil("http://192.168.6.165:9106/com.wwwarehouse.xdw.supplychainBi.service.BiService")
    data_activity = hessian_util.request("getBiDataParam",
                                       "{"
                                       "\"selects\": [\"parent_node\", \"node_id\",],"
                                       "\"dataSet\": \"node_description\","
                                       "}")
    origin_data = data_activity["data"]["data"]["dataList"]
    origin_data = pd.DataFrame(origin_data)
    print_msg('父子节点关系表的数据量为' + str(len(origin_data)) + '条')
    return origin_data

def read_alllinks(string, s, w):
    if isoffice:
        mdpoint = sys.argv[1]  ##############################################################   参数修改
    else:
        mdpoint = timestrtest     ## 非正式版时间修改

    date0 = datetime.datetime.date(datetime.datetime.strptime(mdpoint, '%Y%m%d'))
    day = str(date0)

    # Today_date = datetime.date(datetime.strptime(date_parameter, '%Y%m%d'))
    hessian_util = HessianUtil("http://192.168.6.165:9106/com.wwwarehouse.xdw.supplychainBi.service.BiService")
    data_links = hessian_util.request("getBiDataParam",
                                       "{"
                                       "\"selects\": [\"resource_id\", \"sup_tran_node_id\", \"dem_tran_node_id\", \"sup_cirl_node_id\", \"dem_cirl_node_id\",],"
                                       "\"dataSet\": \"resource_link\","
                                       "\"filters\":["
                                       "{\"column\":\"resource_id\",\"type\":\"between\",\"values\":[\"%s\",\"%s\"]},"
                                       "{\"column\":\"cnt_at\",\"type\":\"in\",\"values\":[\"%s\",]},"
                                       "]"
                                       "}" %(s, w, day))
    origin_data = data_links["data"]["data"]["dataList"]
    origin_data = pd.DataFrame(origin_data)
    print_msg(string + '资源id~交供id~交需id，链路' + str(len(origin_data)) + '条')
    return origin_data

def data_arrange(data_activity = None, relation_df = None):
    condX = data_activity is not None
    condY = relation_df is not None
    if condX and condY:
        ## data_activity的'stran_nid'是门店子节点， relation_df的'node_id'是门店子节点，
        df = pd.merge(data_activity, relation_df, left_on=['stran_nid'], right_on=['node_id'], how = 'left')
        data_activity = df.drop(['stran_nid', 'node_id'], axis = 1)
        data_activity.rename(columns={'parent_node': 'dtran_nid'}, inplace=True)
        # 此处的 'parent_node'已经转换成销售数据里的'dtran_nid'了
    return data_activity

def data_devide3(data_norm, keys1, keys2):
    if data_norm is None:
        return None

    all_data = {}  ## 所有数据组的字典
    data_classfy = data_norm.groupby(keys1)
    for gclass_name, gclass_data in data_classfy:
        datacls = group_data()
        ### 交供节点下的总数据
        gclass_data = gclass_data.drop(keys1, axis=1)
        ### 交供节点下各门店销售数据求和
        eachshop = {}
        msg = {}
        data_classfy2 = gclass_data.groupby(keys2[0])
        for gclass_name2, gclass_data2 in data_classfy2:
            eachshop[gclass_name2] = gclass_data2['out_qty'].sum()
            type = gclass_data2[keys2[1]].iloc[0]
            scirl_nid = gclass_data2[keys2[2]].iloc[0]
            dcirl_nid = gclass_data2[keys2[3]].iloc[0]
            msg[gclass_name2] = [type, scirl_nid, dcirl_nid]
        datacls.shop_qtysum = eachshop
        datacls.msg = msg
        gclass_data = gclass_data.drop(keys2, axis=1)
        datacls.saledata = gclass_data
        all_data[gclass_name] = datacls
    return all_data

def data_devide2(data_norm, keys):
    if data_norm is None:
        return None
    all_data = {}  ## 所有数据组的字典
    data_norm['count'] = 1
    data_classfy = data_norm.groupby(keys)
    for gclass_name, gclass_data in data_classfy:
        gclass_data = gclass_data.drop(keys, axis=1)
        # gclass_data.drop_duplicates(subset=['price_chmod','discount', 'effective_st', 'effective_et'],keep='first',inplace=True)
        data_classfy2 = gclass_data.groupby(['price_chmod','discount', 'effective_st', 'effective_et'])
        all_data[gclass_name] ={}
        for gclass_name2, gclass_data2 in data_classfy2:
            all_data[gclass_name][gclass_name2] = gclass_data2['count'].sum()
    return all_data

def data_devide0(data_norm, keys):
    all_data = {}  ## 所有数据组的字典
    data_classfy = data_norm.groupby(keys)
    for gclass_name, gclass_data in data_classfy:
        gclass_data = gclass_data.drop(keys, axis=1)
        all_data[gclass_name] = gclass_data
    return all_data

def ttttnew(origin_data, tpoints):
    input = origin_data
    # output = origin_data[['cnt_at', 'out_qty']]

    train_raw = input[input['cnt_at'] < tpoints[1]]
    test_raw = input[input['cnt_at'] >= tpoints[1]]
    train_raw = train_raw.fillna(0)
    test_raw = test_raw.fillna(0)
    trainx, trainy, testx, testy = feature_project(train_raw, test_raw)
    trainx = trainx.fillna(0)
    testx = testx.fillna(0)
    trainy = trainy.fillna(0)
    testy = testy.fillna(0)

    return trainx, trainy, testx, testy

def observe2(dataA, dataB, dataC, dataD, dataE, dayx):
    plt.subplot(211)
    day = dataA.index
    l1 = len(dataA)
    max = dataB['out_qty'].max()
    plt.plot(dataB['out_qty'])
    labels = []
    for column in range(len(activity_mode_list)):
        if dataA[activity_mode_list[column]].sum() == 0 and activity_mode_list[column] not in ['满减', '满送', '立减','乐高', '加购']:
            continue
        else:
            labels.append(activity_mode_eng[column])
            plt.bar(day, dataA[activity_mode_list[column]] * max, width=1, alpha = 0.3)
    plt.legend(['out_qty'] + labels)
    plt.ylim([-5, max*1.2])
    # plt.xticks(np.arange(0,l1,7))
    plt.subplot(212)
    day2 = [i for i in range(len(dataC))]
    y = dataE
    y1= np.array(dataD['out_qty'])
    plt.plot(y1)
    plt.plot(y)

    max2 = np.max(y1)
    labels = []
    for column in range(len(activity_mode_list)):
        if dataC[activity_mode_list[column]].sum() == 0 and activity_mode_list[column] not in ['满减', '满送', '立减','乐高', '加购']:
            continue
        else:
            labels.append(activity_mode_eng[column])
            plt.bar(day2, dataC[activity_mode_list[column]] * max2, width=1, alpha = 0.3)
    plt.legend(['out_qty', 'pred'] + labels)
    plt.ylim([-5, max2*1.2+ 100])
    plt.xlabel('day')
    plt.show()

def pred_model(trainx, trainy, testx, testy):
    # xgb = XGBRegressor()
    # xgb.fit(trainx, trainy)
    # xgb_pred = xgb.predict(testx)

    rf = RandomForestRegressor(n_estimators=50, max_features=30, max_depth=8, oob_score=True)
    rf.fit(trainx, trainy)
    rf_pred = rf.predict(testx)

    knn = KNeighborsRegressor(n_neighbors=3, leaf_size=5, n_jobs=-1)
    knn.fit(trainx, trainy)  # <---------------------------------------------------参数有待调整
    knn_pred = knn.predict(testx)
    knn_pred = np.reshape(knn_pred, (1, len(testy)))

    result = 0.5 * rf_pred + 0.5 * knn_pred[0]
    return result

def data_tackle(tpoints, statistics, data_sale = None, data_activity = None, type = None):
    if data_sale is None:
        print_msg(type + '空')
        return None

    data_merge = {}  ## 合并后的数据字典
    tindex = pd.date_range(start=tpoints[0], end=tpoints[2] - datetime.timedelta(days=1))
    bottle = pd.DataFrame({'cnt_at': tindex})  ## 数据空瓶，用来装数据
    for mode in activity_mode_list:
        bottle[mode] = 0
    bottle['discount'] = 1

    if data_activity is not None:  # 有活动信息
        for key in tqdm(data_sale):
            keynew = (key[0], key[1], type)
            datacls = data_sale[key]
            temp = datacls.saledata.groupby('cnt_at').sum()   ######## 对同日期的qty求和合并
            samples_num = len(temp)
            statistics[keynew] = [samples_num, 0, 0, 0, 0, 0, 0, 0, 0]  ### 统计样本量
            if samples_num > threshold:  # 样本量充足
                df_bottle = bottle.copy()
                dtran_nids = datacls.shop_qtysum.keys()
                for dtran_nid in dtran_nids:
                    keyact = (key[0], dtran_nid)
                    if keyact in data_activity:
                        for actmsg in data_activity[keyact]:
                            (mode, discount, st, et) = actmsg
                            indexs = df_bottle[(df_bottle['cnt_at'] >= st) & (df_bottle['cnt_at'] <= et)].index         ########## 请注意
                            df_bottle[mode][indexs] += data_activity[keyact][actmsg] *10
                            df_bottle['discount'][indexs] = discount
                temp = pd.merge(df_bottle, temp, how='left', on=['cnt_at'], sort = False)  #### 对上面的temp与数据空瓶合并，此时已经包含了空的活动信息与qty数据
                data_merge[keynew] = temp.fillna(0)
            else:  ######### 样本量不足
                data_merge[keynew] = temp
            pass
    else:
        for key in tqdm(data_sale):
            keynew = (key[0], key[1], type)
            datacls = data_sale[key]
            temp = datacls.saledata.groupby('cnt_at').sum()
            samples_num = len(temp)
            statistics[keynew] = [samples_num, 0, 0, 0, 0, 0, 0, 0, 0]  ### 统计样本量
            if samples_num > threshold:  # 样本量充足
                df_bottle = bottle.copy()
                temp = pd.merge(df_bottle, temp, how='left', on=['cnt_at'], sort = False)
                data_merge[keynew] = temp.fillna(0)
            else:  ######### 样本量不足
                data_merge[keynew] = temp

    print_msg(type + '处理完成')
    return data_merge


def data_preprocs_small(dataone, tpoints, key, statistics):
    days = (tpoints[1] - tpoints[0]).days
    data = dataone[dataone.index < tpoints[1]]
    qtysum = data['out_qty'].sum()
    qtymean = qtysum / days
    dayspred = (tpoints[2] - tpoints[1]).days
    result = np.array([qtymean] * dayspred)
    if isobserv:
        statistics[key][1] = 0  ## 平均绝对误差
        statistics[key][2] = 0  ## 标准差
        statistics[key][3] = 0  ## 误差最大值
        statistics[key][4] = 0  ## 评价时间粒度的准确性指标
        statistics[key][5] = 0  ## 评价时间粒度的准确性指标
        statistics[key][6] = 0  ## 评价时间粒度的准确性指标
        statistics[key][7] = 0  ## 评价时间粒度的准确性指标
        statistics[key][8] = 1  ## 状态
    return result

def data_preprocs_new(dataone, tpoints, key, statistics):
    # dataone = dataone.drop(activity_mode_list, axis = 1)
    trainx, trainy, testx, testy = ttttnew(dataone, tpoints)

    result1 = pred_model(trainx, trainy, testx, testy)
    # resultx = pred_model2(trainx, trainy, testx, testy)

    # print(key)
    # observe2(trainx, trainy, testx, testy, result1, dataone['cnt_at'])

    if isobserv:
        mae, std, emax, tpeak, tpeakfact, factmean, predmean = result_statistics(result1, testy)
        statistics[key][1] = mae  ## 平均绝对误差
        statistics[key][2] = std  ## 标准差
        statistics[key][3] = emax  ## 误差最大值
        statistics[key][4] = tpeak  ## 评价时间粒度的准确性指标
        statistics[key][5] = tpeakfact  ## 状态
        statistics[key][6] = factmean  ## 状态
        statistics[key][7] = predmean  ## 状态
        statistics[key][8] = 1  ## 状态

    return result1

### part 2 特征工程

### part 3 算法模块
def datatackle(trainx, trainy, testx, testy):
    trnxtemp = np.array(trainx)
    trnytemp = np.array(trainy)
    row, col = np.shape(trnxtemp)
    trnx = np.zeros((row, col, 1))
    trnx[:,:,0] = trnxtemp
    trny = trnytemp

    tstxtemp = np.array(testx)
    tstytemp = np.array(testy)
    row, col = np.shape(tstxtemp)
    tstx = np.zeros((row, col, 1))
    tstx[:, :, 0] = tstxtemp
    tsty = tstytemp
    return trnx, trny, tstx, tsty

### part 4 数据写入
def result_statistics(result, testy):
    pred = result[0]
    fact = np.array(testy['out_qty'])
    error = pred - fact
    x = abs(error)
    mad = np.mean(x)  # 平均绝对偏差
    std = np.std(error)  # 标准差
    emax = max(x)    # 最大绝对误差
    tpeak = len(x[x > 0.01])## 体现时间粒度的准确性
    tpeakfact = len(fact[fact > 0.01])
    factmean = np.mean(fact)
    predmean = np.mean(pred)
    return mad, std, emax, tpeak, tpeakfact, factmean, predmean

def data_normalize(result, datacls, key, final_wdata_temp, string):
    values = datacls.shop_qtysum.values()
    sumx = np.sum(list(values))

    date0 = pd.to_datetime(datetime.datetime.strptime(timestrtest, '%Y%m%d'))
    date = str(date0)

    datetemp = pd.to_datetime(datetime.datetime.strptime(timestrtest, '%Y%m%d')) + datetime.timedelta(hours=12)
    datetemp = str(datetemp)

    for va in datacls.shop_qtysum:
        normal_result = []
        msg = datacls.msg[va]
        if sumx == 0:
            part = 0
        else:
            part = datacls.shop_qtysum[va] / sumx
        partresult = part * result ## 分量

        monthsum = sum(result[0:30])
        # quartersum = sum(result[0, 0:90])
        # halfsum = sum(result[0, 0:180])

        demand_date_1 = date
        sup_tran_node_id_2 = int(key[1])
        dem_tran_node_id_3 = int(va)
        sup_cirl_node_id_4 = int(msg[1])
        dem_cirl_node_id_5 = int(msg[2])
        cnt_at_6 = datetemp
        resource_id_7 = int(key[0])
        demand_qty_8 = -1
        demand_duration_10 = demand_dura[msg[0]]
        demand_qty_month_11 = int(monthsum)
        demand_qty_quarter_12 = int(monthsum * 3)
        demand_qty_halfyear_13 = int(monthsum * 6)
        docking_business_id_14 = 1000001

        for i in range(predperiods):
            demand_date_1 = str(date0 +  datetime.timedelta(days=i + 1))
            demand_qty_8 = int(partresult[i])
            line = (demand_date_1, sup_tran_node_id_2, dem_tran_node_id_3, sup_cirl_node_id_4, dem_cirl_node_id_5, cnt_at_6, resource_id_7,
                    demand_qty_8, demand_duration_10, demand_qty_month_11, demand_qty_quarter_12, demand_qty_halfyear_13, docking_business_id_14)
            normal_result.append(line)

        key_norm = (int(key[0]), int(key[1]), int(va))
        try:
            final_wdata_temp[key_norm] = [(int(msg[1]), int(msg[2])), normal_result]
        except:
            print_msg(string + '超出字典范围' + str((key[0], key[1], va, msg[1], msg[2])))

def data_normalize1(rid, nods):
    normal_result = []
    date_str = timestrtest
    date0 = pd.to_datetime(datetime.datetime.strptime(date_str, '%Y%m%d'))
    date = str(date0)

    datetemp = pd.to_datetime(datetime.datetime.strptime(timestrtest, '%Y%m%d')) + datetime.timedelta(hours=12)
    datetemp = str(datetemp)

    demand_date_1 = date
    sup_tran_node_id_2 = int(nods[0])
    dem_tran_node_id_3 = int(nods[1])
    sup_cirl_node_id_4 = int(nods[2])
    dem_cirl_node_id_5 = int(nods[3])
    cnt_at_6 = datetemp
    resource_id_7 = rid
    demand_qty_8 = -1
    demand_duration_10 = -1
    demand_qty_month_11 = -1
    demand_qty_quarter_12 = -1
    demand_qty_halfyear_13 = -1
    docking_business_id_14 = 1000001

    for i in range(predperiods):
        demand_date_1 = str(date0 + datetime.timedelta(days=i + 1))
        line = (demand_date_1, sup_tran_node_id_2, dem_tran_node_id_3, sup_cirl_node_id_4, dem_cirl_node_id_5, cnt_at_6,
            resource_id_7, demand_qty_8, demand_duration_10, demand_qty_month_11, demand_qty_quarter_12,
            demand_qty_halfyear_13, docking_business_id_14)

        normal_result.append(line)

    return normal_result

def final_wdata_tackle(data_alllink, final_wdata_temp, string):
    final_wdata_part = []
    key_refer = final_wdata_temp.keys()
    for key in data_alllink:
        if key not in key_refer:
            scirl_nid = data_alllink[key]['sup_cirl_node_id'].iloc[0]
            dcirl_nid = data_alllink[key]['dem_cirl_node_id'].iloc[0]
            fournods = [key[1], key[2], scirl_nid, dcirl_nid]
            normdata = data_normalize1(key[0], fournods)
            # final_wdata_temp[key] = [(scirl_nid, dcirl_nid), normdata]
            final_wdata_part += normdata
        else:
            final_wdata_part += final_wdata_temp[key][1]

    samples = len(data_alllink.keys()) * predperiods
    samplescheck = len(final_wdata_part)
    if int(samples) != int(samplescheck):
        print_msg(string + '数据长度校验有误，请关注' + str((samples,samplescheck)))
    else:
        print_msg(string + '数据正常')
    return final_wdata_part

def deleta_data():
    ##########连接服务器##############
    print_msg('将执行数据写入程序')
    # db = pymysql.connect(host="192.168.6.122",database="data_cjgm",user="root",password="Rpt@123456",port=3306,charset='utf8')
    db = pymysql.connect(host="172.16.4.7", database="supply_chain", user="bi_user", password="RL9FCS4@QTrmOsRk",
                         port=3306, charset='utf8')
    print_msg('服务器已连接')
    ##########删除重复日期数据##############
    print_msg('执行删除重复日期数据')
    cursor = db.cursor()
    # dt = sys.argv[1]
    dt = timestrtest
    dtnew = dt[0:4] + '-' + dt[4:6] + '-' + dt[6:8] + ' 00:00:00'
    print_msg('时间参数：' + dtnew)
    sql = """delete from supply_chain.core_out_sf_node_demand
     where demand_date = str_to_date(\'%s\','%%Y-%%m-%%d %%H:%%i:%%s')"""%(dtnew)
    cursor.execute(sql)

def write_data(data):
    ##########连接服务器##############
    print_msg('将执行数据写入程序')
    # db = pymysql.connect(host="192.168.6.122",database="data_cjgm",user="root",password="Rpt@123456",port=3306,charset='utf8')
    db = pymysql.connect(host="172.16.4.7", database="supply_chain", user="bi_user", password="RL9FCS4@QTrmOsRk",
                         port=3306, charset='utf8')
    print_msg('服务器已连接')

    # deleta_data() ################################  删除数据

    # ## 删除重复数据
    ##########向数据库中写入数据##############
    if len(data) == 0:  ### 数据为空
        print_msg("数据为空，不需要写")
    else:  ### 数据非空
        cursor = db.cursor()
        sql = """INSERT INTO supply_chain.core_out_sf_node_demand
            (demand_date,
            sup_tran_node_id,
            dem_tran_node_id,
            sup_cirl_node_id,
            dem_cirl_node_id,
            cnt_at,
            resource_id,
            demand_qty,
            demand_duration,
            demand_qty_month,
            demand_qty_quarter,
            demand_qty_halfyear,
            docking_business_id)
            VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        try:
            datalen = len(data)
            batch = np.ceil(datalen /20000)
            print_msg('批次量' + str(batch))
            l = int(batch - 1)
            for i in range(l):
                print_msg('正在写入第' + str(i + 1) + '批次')
                bias = 20000 * i
                datax = data[0 + bias  : 20000 + bias]
                cursor.executemany(sql, datax)

            print_msg('正在写入第' + str(batch) + '批次')
            datax = data[20000 * l: ]
            cursor.executemany(sql, datax)

            # cursor.execute(sql)
            print_msg("所有数据写入成功")
            db.commit()
        except OSError as reason:
            print_msg('出错原因是%s' % str(reason))
            db.rollback()
    ##########数据接口关闭##############
    print_msg("关闭数据接口")
    db.close()

###############################################################################
###############################################################################
## 林源老程序
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

def time_subset(x):
    x["dayofweek"] = x['cnt_at'].apply(lambda x: x.dayofweek)
    x["weekofyear"] = x["cnt_at"].apply(lambda x: x.weekofyear)
    x['year'] = x['cnt_at'].apply(lambda x: x.year)
    x['month'] = x['cnt_at'].apply(lambda x: x.month)
    x['day'] = x['cnt_at'].apply(lambda x: x.day)
    x['period_of_month'] = x['day'].apply(lambda x: period_of_month(x))
    x['period2_of_month'] = x['day'].apply(lambda x: period2_of_month(x))
    x['week_of_month'] = x['day'].apply(lambda x: week_of_month(x))
    x['quarter'] = x['month'].apply(lambda x: quarter(x))

def creat_test(tpoints):
    date_prediction = pd.date_range(start = tpoints[1], periods = predperiods)
    date_dataframe = pd.DataFrame({"cnt_at": date_prediction})
    test = time_subset(date_dataframe)
    return test

# <----------------------------------------------------------------------------------------------构建每一个时间分布上的特征
def feature_project(train_data, test_data):
    time_subset(train_data)
    time_subset(test_data)
    vars_be_agg = "out_qty"
    vars_to_agg = ["dayofweek", "weekofyear", "month", "day", "year", "period_of_month", "period2_of_month",
                   "week_of_month", "quarter", ["month", "dayofweek"], ["quarter", "month"]]
    # vars_to_agg = ["dayofweek", "weekofyear", "period2_of_month", "week_of_month", "quarter", ["month", "dayofweek"], ["quarter", "month"]]

    for var in vars_to_agg:
        agg = train_data.groupby(var)[vars_be_agg].agg(["sum", "mean", "std", "skew", "median", "min", "max", "count", pd.Series.kurt])
        if isinstance(var, list):
            agg.columns = pd.Index(["fare_by_" + "_".join(var) + "_" + str(e) for e in agg.columns.tolist()])
        else:
            agg.columns = pd.Index(["fare_by_" + var + "_" + str(e) for e in agg.columns.tolist()])
        train_data = pd.merge(train_data, agg.reset_index(), on=var, how="left")
        test_data = pd.merge(test_data, agg.reset_index(), on=var, how="left")

    trainy = train_data[["out_qty"]]
    trainx = train_data.drop(["cnt_at", "out_qty"], axis=1)

    testy = test_data[["out_qty"]]
    testx = test_data.drop(["cnt_at", "out_qty"], axis=1)
    return trainx, trainy, testx, testy


def procedure(tpoints, relation_df, skul, skur, process):
    ## relation_dict为交易需方父节点下的门店列表
    statistics = {}  ## 统计结果
    string = '进程' + str(process) + ':'
    print_msg(string + '负责资源id段为' + str(skul) + '~' + str(skur))

    t1 = time.time()   ##### 计时点
    ##### 读取活动信息数据
    data_alllink = read_alllinks(string, skul, skur)
    data_activity = read_activity_data(string, skul, skur)
    ##### 读取销售信息数据
    data_saleA = read_sale_dataA(string, skul, skur)
    data_saleB = read_sale_dataB(string, skul, skur)
    t2 = time.time()    ##### 计时点
    print_msg(string + '读数据耗时:' + str(t2 - t1) + '秒')
    ########################################################
    data_activity = data_arrange(data_activity, relation_df) ## 交易需方父节点下的活动信息，即所有同类门店活动信息的总和

    data_alllink = data_devide0(data_alllink, ['resource_id', 'sup_tran_node_id', 'dem_tran_node_id'])
    data_saleA = data_devide3(data_saleA, ['resource_id', 'stran_nid'], ['dtran_nid', 'type', 'scirl_nid', 'dcirl_nid'])
    data_saleB = data_devide3(data_saleB, ['resource_id', 'stran_nid'], ['dtran_nid', 'type', 'scirl_nid', 'dcirl_nid'])
    data_activity = data_devide2(data_activity, ['resource_id', 'dtran_nid'])

    all_datasA = data_tackle(tpoints, statistics, data_saleA, data_activity, 'A')  ## 函数中统计了样本量
    all_datasB = data_tackle(tpoints, statistics, data_saleB, data_activity, 'B')  ## 函数中统计了样本量
    t3 = time.time()    ##### 计时点
    print_msg(string + '数据预处理:' + str(t3 - t2) + '秒')
    # #####################################################

    final_wdata_temp = {}
    if all_datasA is not None:
        for key in tqdm(all_datasA):
            dataone = all_datasA[key]
            if statistics[key][0] > threshold:
                resultA = data_preprocs_new(dataone, tpoints, key, statistics)  ## 函数中统计了预测评估指标
            else:
                resultA = data_preprocs_small(dataone, tpoints, key, statistics)  ## 函数中统计了预测评估指标
            data_normalize(resultA, data_saleA[(key[0], key[1])], key, final_wdata_temp, string)

    if all_datasB is not None:
        for key in tqdm(all_datasB):
            dataone = all_datasB[key]
            if statistics[key][0] > threshold:
                resultB = data_preprocs_new(dataone, tpoints, key, statistics)  ## 函数中统计了预测评估指标
            else:
                resultB = data_preprocs_small(dataone, tpoints, key, statistics)  ## 函数中统计了预测评估指标

            data_normalize(resultB, data_saleB[(key[0], key[1])], key, final_wdata_temp, string)  ###########################待修改

    final_wdata_part = final_wdata_tackle(data_alllink, final_wdata_temp, string)

    t4 = time.time()    ##### 计时点
    print_msg(string + '随机森林预测耗时:' + str(t4 - t3) + '秒')

    # write_data(all_normal_data)
    if len(statistics) == 0:
        statis = None
    else:
        statis = pd.DataFrame(statistics).T
    print_msg(string + '预测完成，进程结束')
    return statis, final_wdata_part

def multi_read():
    statis_datas = [] # 统计结果
    pred_datas = []  # 预测结果
    ridlest = [] # 残余新品ID
    fournods = [] # 4个节点
    results = []
    pool = multiprocessing.Pool(processes=4)  # 创建4个进程
    # cutpoint = [0, 2400, 3100, 3900, 4900, 6000, 7800, 15000]
    cutpoint = [0, 3200, 3600, 4000, 4400, 4800, 5200, 5800, 15000]
    # cutpoint = [2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300]
    # cutpoint = [3300, 3305, 3310, 3315, 3320, 3325, 3330, 3335, 3340]
    # cutpoint = [3300, 3305, 3310, 3315, 3320]
    for i in range(8):
        results.append(pool.apply_async(procedure, args=(tpoints, relation_df, cutpoint[i], cutpoint[i + 1] - 1, i)))
    pool.close()  # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
    pool.join()  # 等待进程池中的所有进程执行完毕
    print('各进程已全部运算结束')
    for res in results:
        if res.get()[0] is not None:
            statis_datas.append(res.get()[0])
        pred_datas += res.get()[1]

    if issvsult:  ## 是否保存预测结果
        will_write_data = pd.DataFrame(pred_datas)  ############ 需要写入的数据 保存下来以便不实之需
        will_write_data.to_csv('will_write_data.csv', header=None)
        print_msg('写入数据保存完成')

    if iswrite:
        tr0 = time.time()
        write_data(pred_datas)  ############ 写数据到数据库
        tr1 = time.time()
        print_msg('写数据耗时:' + str(tr1 - tr0))

    if issvefun:  ## 是否保存评估指标
        pred_result = pd.concat(statis_datas)  ############ 预测统计的结果
        pred_result.to_csv('pred_result.csv', header=['samplenum', 'mae', 'std', 'emax', 'terrorpeak', 'tfactpeak', 'factmean', 'predmean', 'status'])


if __name__ == '__main__':
    print_msg('start')
    starttime = time.time()
    tpoints = string2tstamp()
    # data_alllink = read_alllinks('ll', 3341, 3341)

    relation_df = read_sale_relationship()   ###### 读取父子节点关系表
    ############################################################
    if isobserv == False:
        multi_read()
    else:
        multi_read()
        # dataf = procedure(tpoints, relation_df, 3120, 3121, 1)
        # dataf[0].to_csv('statis.csv', header= None)
        pass
    #############################################################
    endtime = time.time()
    print_msg('算法整体耗时：' + str(endtime - starttime))
    print_msg('end')