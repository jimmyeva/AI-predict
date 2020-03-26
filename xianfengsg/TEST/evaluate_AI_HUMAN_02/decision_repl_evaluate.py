# -*- coding: utf-8 -*-
# @Time    : 2019/9/25 9:44
# @Author  : Ye Jinyu__jimmy
# @File    : decision_repl


import sys
print(sys.version)
import pandas as pd
import cx_Oracle
import os
import numpy as np

'''2019-11-29日与产品中心沟通，在安全库存处进行优化，增大安全库存的设置'''
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
# 注：设置环境编码方式，可解决读取数据库乱码问题
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
from matplotlib import pyplot as plt
import re
plt.switch_backend('agg')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import datetime
import warnings
import time
import tqdm
import pymysql



def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print(
            "----生成新的文件目录----")
    else:
        print("当前文件夹已经存在")

def print_in_log(string):
    print(string)
    file = open('./data/log.txt', 'a')
    file.write(str(string) + '\n')


# import chinese_calendar as calendar  #
warnings.filterwarnings("ignore")

#--------------------------------------------------------------获取商品的五位码
def read_oracle_data(i):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    #读取的数据包括销量的时间序列，天气和活动信息
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", parameters)
    #查看详细的出库数据，进行了日期的筛选，查看销量签50名的SKU
    goods_sql = """SELECT g.GID,g.CODE FROM GOODSH g WHERE g.GID IN %s""" %(i,)
    goods = pd.read_sql(goods_sql, conn)
    goods = goods.rename(index=str, columns={'GID': 'Sku_id','CODE':'Code'})
    #将SKU的的iD转成list，并保存前80个，再返回值
    conn.close
    return goods

#----------------------------------------------------------------对数据预测数据获取，并进行处理
def get_forecast(data):
    #获取五位的商品码
    print('data_length:'+str(len(data)))
    data_sku_id = data[['Sku_id']]
    data_sku_id = data_sku_id.drop_duplicates()
    data_sku_id = data_sku_id.reset_index(drop=True)
    sku_id_list = data_sku_id['Sku_id'].to_list()
    gid_tuple = tuple(sku_id_list)
    print_in_log('gid_tuple_length:'+str(len(gid_tuple)))
    sku_code= read_oracle_data(gid_tuple)
    print_in_log('sku_code_length:'+str(len(sku_code)))
    data_sku_id = pd.merge(data_sku_id,sku_code,on='Sku_id',how='inner')
    final = pd.merge(data_sku_id,data,on='Sku_id',how='right')
    return final

#取出所有sku对应的装箱比例
def resource_convert(i):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    #读取的数据包括销量的时间序列，天气和活动信息
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", parameters)
    #查看详细的出库数据，进行了日期的筛选，查看销量签50名的SKU
    resource_sql = """SELECT s.QPC,COUNT(s.QPC) 总次数 FROM STKOUTDTL s WHERE s.GDGID= %s AND rownum < 1000
AND s.CLS='统配出' GROUP BY s.QPC ORDER BY count(s.QPC)DESC""" %(i)
    resource = pd.read_sql(resource_sql, conn)
    #将SKU的的iD转成list，并保存前80个，再返回值
    conn.close
    resource = resource.sort_values(by=['总次数'],ascending=False)
    if resource.empty ==True:
        resource_conver_rate = 1
    else:
        resource_conver_rate = resource['QPC'].iloc[0]
    return resource_conver_rate

#-------------------------------叫货目录获取装箱规格
def box_gauge(sku):
    print_in_log('连接到mysql服务器...')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123", port=3306, charset='utf8')
    print_in_log('连接成功,开始读取预测数据')
    resource_sql = """  SELECT * FROM p_call_record pcr WHERE pcr.warehouse_id = '1000255' 
    AND pcr.goods_id='%s'""" % (sku)
    db.cursor()
    resource = pd.read_sql(resource_sql, db)
    print_in_log('连接成功,预测数据读取完成')
    db.close()
    if resource.empty ==True:
        resource_conver_rate = 1
    elif resource['box_gauge'].iloc[0] == '0':
        resource_conver_rate = 1
    else:
        resource_conver_rate = resource['box_gauge'].iloc[0]
    resource_conver_rate = float(resource_conver_rate)
    resource_conver_rate = round(resource_conver_rate)
    resource_conver_rate = int(resource_conver_rate)
    return resource_conver_rate


# resource_conver_rate = resource_convert(3006430)
# print(resource_conver_rate)
#如下是要拿到所有的sku对应的转换比例是多少


# 获取五位的商品码
def get_convert_rate(data):
    data_sku_id = data[['Sku_id']]
    data_sku_id = data_sku_id.drop_duplicates()
    data_sku_id = data_sku_id.reset_index(drop=True)
    sku_id_list = data_sku_id['Sku_id'].to_list()
    sku_convert_rate = pd.DataFrame(columns={'Sku_id','rate'})
    for sku in sku_id_list:
        print_in_log('正在获取sku的装箱规格数据:' + str(sku))
        convert_rate = box_gauge(sku)
        # sku_convert_rate = sku_convert_rate.append({'Sku_id': sku}, ignore_index=True)
        sku_convert_rate = sku_convert_rate.append({'rate': convert_rate,'Sku_id': sku}, ignore_index=True)
    return sku_convert_rate


#将资源转换因子和产品相匹配
def get_final_forecast(data,sku_convert):
    forecast = get_forecast(data)
    final = pd.merge(forecast,sku_convert,on='Sku_id',how='left')
    # final.to_csv('D:/AI/xianfengsg/online/V_1.2/final-11-21.csv',encoding='utf_8_sig')
    final['Forecast_box'] = final['Forecast_qty']/final['rate']
    final['Forecast_box'] = final['Forecast_box'].apply(lambda x: round(x))
    return final


# 获取昨天，今天，明天和后天共四天的日期str
def get_all_date(today):
    today_datetime = datetime.datetime.strptime(today, '%Y%m%d')
    yesterday = (today_datetime - datetime.timedelta(1)).strftime('%Y%m%d')
    tomorrow_date = (today_datetime + datetime.timedelta(1)).strftime('%Y%m%d')
    TDAT_date = (today_datetime + datetime.timedelta(2)).strftime('%Y%m%d')

    return yesterday,today,tomorrow_date,TDAT_date

#----------------获取预测数据
def get_original_forecast(today_date):
    print_in_log('连接到mysql服务器...')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123", port=3306, charset='utf8')
    print_in_log('连接成功,开始读取预测数据')
    weather_sql = """SELECT * FROM dc_forecast where Update_time = DATE('%s')"""%(today_date)
    db.cursor()
    read_orignal_forecast = pd.read_sql(weather_sql, db)
    print_in_log('连接成功,预测数据读取完成')
    db.close()
    return read_orignal_forecast

#定义函数从Oracle数据库里面选择每日的库存数据
def get_stock(today_date):
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
    WHERE FILDATE =  to_date('%s','yyyy-mm-dd')""" %(today_date)
    goods = pd.read_sql(goods_sql, conn)
    goods.dropna(axis=0, how='any', inplace=True)
    goods['GOODS_CODE'] = goods['GOODS_CODE'].astype(str)
    conn.close
    return goods


'''该设置的参数是在试水阶段采用特定的SKU进行补货的方式和方法'''
#--------------------------------------------------------先设置需要的SKU，ssd=selected_sku_dataframe
def test_sku(data):
    ssd = pd.DataFrame({'GOODS_CODE':['65550','07540','07310','11620','11600','16010','16040','07350','13160','06390',
                                      '05200','01310','08890','05020','07640','11120','06850','65770','07600','12190',
                                      '01270','07340','07300','11650','07950','11710','12130','01020','07220','12240'
                                      ]})
    result_data = pd.merge(data,ssd,on='GOODS_CODE',how='inner')
    return result_data

#设置主函数用于读取数据库的库存和预测数据
def get_db(yes_date,today_date):
    stock_data = get_stock(today_date)
    stock_data = test_sku(stock_data)
    original_forecast = get_original_forecast(today_date)
    return stock_data,original_forecast


#-----------------------------------------------这里是进行编码转换，基本的数据
def cleaning_data(original_forecast, stock_data):
    sku_convert_rate = get_convert_rate(original_forecast)
    # sku_convert_rate.to_csv('D:/jimmy-ye/AI/AI_supply_chain/data/decision/sku_convert_rate.csv',encoding='utf_8_sig')
    final = get_final_forecast(original_forecast,sku_convert_rate)
    #以下需要对每日的盘点后的库存进行解析，并保存
    # df = data_xls.parse(sheet_name='采购更新模块', header=1, converters={u'订货编码': str})
    data_stock = stock_data[['GOODS_CODE','GOODS_NAME','DIFFERENCE_QTY','INVENTORY']]
    data_stock = data_stock.dropna(axis=0,how='any')
    data_stock.columns = ['Code','Sku_name','Stock','Inventory']
    print_in_log('数据清洗完成')
    return data_stock,final


'''分仓采购场景中默认提前期是一天，对于协同的压力将会在统采的嘉兴仓中产生'''
#=========================================================获取在途库存的数据
#-----------------------------------------------先确定决策的日
def get_orders_time(data_xls):
    df = data_xls.parse(sheet_name='采购更新模块')
    columns = list(df.columns.values)
    date = re.sub("\D", "", columns[4])
    date_time = datetime.datetime.strptime(date, '%Y%m%d')
    date_delta = date_time - datetime.timedelta(1)
    date_str_today = datetime.datetime.strftime(date_time, "%Y-%m-%d")
    date_str_yesterday = datetime.datetime.strftime(date_delta, "%Y-%m-%d")
    date_TDAT = date_time + datetime.timedelta(2)
    date_TDAT = datetime.datetime.strftime(date_TDAT, "%Y-%m-%d")
    return date_str_yesterday,date_str_today,date_TDAT


def algorithm_SS(final, tomorrow_date, TDAT_date):
    #=================================================以下SS的计算,K取值2
    Code = list(set(final['Code'].tolist()))
    data_SS = pd.DataFrame()
    for i in Code:
        data_mid = final[final['Code'] == i]
        forecast = data_mid['Forecast_box'].values
        #因为设置较大的安全库存的关系，在原有的基础上增加SS
        std = 2 * (np.sqrt(((forecast - np.mean(forecast)) ** 2).sum() / (forecast.size)))
        data_SS = data_SS.append({'Code': i,'SS': std,}, ignore_index=True)
    # print(data_SS)
    data_merge = pd.merge(final,data_SS,on='Code',how='left')
    data_merge.fillna(method='ffill')
    data_final = data_merge[data_merge['Account_date'] == TDAT_date ]
    #如下是获得第二天的预测补货值，对二配行为进行部分规避
    forecast_tomorrow = final[final['Account_date'] == tomorrow_date]
    forecast_tomorrow = forecast_tomorrow[['Code','Forecast_box']]
    forecast_tomorrow= forecast_tomorrow.rename(index=str, columns={'Forecast_box': 'Forecast_box_tomorrow'})
    data_final = pd.merge(data_final,forecast_tomorrow,on='Code',how='inner')
    def compare(x):
        if x['Code'] == '16040':
            return 0
        elif x['Code'] == '16010':
            return 0
        else:
            if x['SS'] < x['Forecast_box']:
                return x['Forecast_box']
            else:
                return x['SS']
    print_in_log('data_final_length:' + str(len(data_final)))
    data_final['SS'] = data_final.apply(lambda x: compare(x), axis=1)
    data_final = data_final[['Sku_id','Code','Dc_name','Dc_code','Munit','Wrh',
                             'Warehouse_name','Sku_name','rate','Forecast_box','SS','Forecast_box_tomorrow']]
    return data_final


def main_function(data_stock,data_final,today_date):
    data_stock['Code'] = data_stock['Code'].astype(str)
    data_stock = data_stock[['Code','Stock','Inventory']]
    '''选择外连接合并，是因为存在仓库的表格并没有数据，但是实际情况是需要订货的，因此采取外连接，并进行补零操作'''
    '''最终选择采用内连接的方式，在于产品进行测试的阶段，先对一些指定的SKU进行计算'''
    data_merge = pd.merge(data_final,data_stock,on='Code',how='inner')
    data_merge= data_merge.fillna(0)
    print_in_log('data_merge_length:'+str(len(data_merge)))
    #对
    def compare_predict(x):
        predict = x['Inventory'] - x['Stock']
        if predict <= x['Forecast_box_tomorrow']:
            return x['Inventory'] - x['Forecast_box_tomorrow']
        else:
            return x['Stock']
    data_merge['Stock_mid'] = data_merge.apply \
        (lambda x: compare_predict(x), axis=1)

    #在加一个逻辑是基于昨日销量的一个对预测值的修正情况，如果预测的数量小于实际的销售数量的话，将会在预测的时候进行未来那天的修正
    def predict_revised(x):
        predict = x['Inventory'] - x['Stock']
        if predict <= x['Forecast_box_tomorrow']:
            return x['Forecast_box']
        else:
            return x['Forecast_box'] +(predict -x['Forecast_box_tomorrow'])

    data_merge['Forecast_box'] = data_merge.apply \
        (lambda x: predict_revised(x), axis=1)


    def calculate_final(x):
        demand = x['Forecast_box'] + x['SS']
        if demand <= x['Stock_mid']:
            return 0
        else:
            return round(demand - x['Stock_mid'])
    data_merge['Suggestion_qty'] = data_merge.apply \
        (lambda x: calculate_final(x), axis=1)
    data_merge['Update_time'] = pd.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data_merge['Account_date'] = today_date
    data_merge = data_merge.drop(['Forecast_box_tomorrow','Inventory','Stock_mid'], axis=1)
    return data_merge

#定义主计算调度逻辑进行函数的汇总
def main(today):
    time_start = datetime.datetime.now()
    yes_date, today_date, tomorrow_date, TDAT_date = get_all_date(today)
    #创建文件夹
    mkdir(today)
    stock_data,original_forecast = get_db(yes_date, today_date)
    data_stock, final = cleaning_data(original_forecast, stock_data)
    data_final = algorithm_SS(final, tomorrow_date, TDAT_date)
    AI_suggestion = main_function(data_stock, data_final,today)
    print_in_log('AI_suggestion')
    db = connectdb()
    drop_data(db,today_date)
    time_end = datetime.datetime.now()
    print(AI_suggestion)
    AI_suggestion.to_csv('./AI_suggestion'+str(today)+'.csv',encoding='utf_8_sig')
    if AI_suggestion.empty:
        print_in_log("The data frame is empty")
        print_in_log("result:1")
        print_in_log("总耗时："+str(time_end-time_start))
        closedb(db)
    else:
        insertdb(db,AI_suggestion)
        closedb(db)
        print_in_log("result:1")
        print_in_log("总耗时："+str(time_end-time_start))


def connectdb():
    print_in_log('连接到mysql服务器...')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123",port=3306, charset='utf8')
    print_in_log('连接成功')
    return db

#《---------------------------------------------------------------------------------------------------------------------删除重复日期数据
def drop_data(db,today_date):
    cursor = db.cursor()
    # date_parameter = datetime.date.today().strftime('%Y-%m-%d')
    # sql = """delete from dc_replenishment"""
    sql = """delete from dc_replenishment where Account_date = DATE('%s')"""%(today_date)
    print_in_log('已经删除重复数据')
    print_in_log(str(sql))
    cursor.execute(sql)

#<======================================================================================================================
def insertdb(db,data):
    cursor = db.cursor()
    # param = list(map(tuple, np.array(data).tolist()))
    data_list = data.values.tolist()
    print_in_log('data_list'+str(data_list))
    sql = """INSERT INTO dc_replenishment (Sku_id,Code,Dc_name,
    Dc_code,Munit,Wrh,Warehouse_name,Sku_name,rate,Forecast_box,
    SS,Stock,Suggestion_qty,update_time,Account_date)
     VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    try:
        cursor.executemany(sql, data_list)
        print_in_log("所有品牌的sku数据插入数据库成功")
        db.commit()
    except OSError as reason:
        print_in_log('出错原因是%s' % str(reason))
        db.rollback()
#<================================================================关闭连接函数
def closedb(db):
    db.close()


#《============================================================================主函数入口
if __name__ == '__main__':
    try:
        today = '20191120'
        main(today)
    except OSError as reason:
        print_in_log('出错原因是%s'%str(reason))
        print_in_log ("result:0")
