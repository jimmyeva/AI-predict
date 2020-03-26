# -*- coding: utf-8 -*-
# @Time    : 2019/12/26 14:35
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
import redis



'''这是第二天及以后采取的补货策略'''
# client = redis.Redis(host="192.168.1.180",port=6379, decode_responses=True,socket_connect_timeout=6000)


# def mkdir(path):
#     folder = os.path.exists('/root/ai/wh_repl/program/prediction/log/'+path)
#     if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
#         os.makedirs('/root/ai/wh_repl/program/prediction/log/'+path)  # makedirs 创建文件时如果路径不存在会创建这个路径
#         print(
#             "----生成新的文件目录----")
#     else:
#         print("当前文件夹已经存在")

# def print_in_log(string):
#     print(string)
#     date_1 = datetime.datetime.now()
#     str_10 = datetime.datetime.strftime(date_1, '%Y%m%d')
#     file = open('/root/ai/wh_repl/program/log/' + 'log_decision' + str(str_10) + '.txt', 'a')
#     file.write(str(string) + '\n')
#
#
def print_in_log(string):
    print(string)
    date_1 = datetime.datetime.now()
    str_10 = datetime.datetime.strftime(date_1, '%Y%m%d')
    file = open('./' + 'log_decision' + str(str_10) + '.txt', 'a')
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


#取出所有sku对应的装箱比例
def resource_convert(i):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", parameters)
    #根据历史的装箱规则进行获取
    '''数据库中是用gid作为主键，所以再用code进行查询的时候，速度过慢，需要再进行一次单独的转换'''
    gid_sql = """SELECT g.gid FROM GOODS g WHERE g.code='%s'""" % (i)
    gid_data = pd.read_sql(gid_sql, conn)
    gid = gid_data['GID'].iloc[0]

    resource_sql = """SELECT s.QPC,COUNT(s.QPC) 总次数 FROM STKOUTDTL s WHERE s.GDGID= %s AND rownum < 200
AND s.CLS='统配出' GROUP BY s.QPC ORDER BY count(s.QPC)DESC""" %(gid)
    resource = pd.read_sql(resource_sql, conn)
    # print('根据历史的装箱规则进行获取'+resource_sql)
    #将SKU的的iD转成list，并保存前80个，再返回值
    conn.close
    resource = resource.sort_values(by=['总次数'],ascending=False)
    if resource.empty ==True:
        resource_conver_rate = 1
    else:
        resource_conver_rate = resource['QPC'].iloc[0]
    return resource_conver_rate

#-------------------------------叫货目录获取装箱规格
def box_gauge(wh_code,sku):
    print_in_log('连接到mysql服务器...')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123", port=3306, charset='utf8')
    print_in_log('连接成功,开始读取预测数据')
    resource_sql = """  SELECT * FROM p_call_record pcr WHERE pcr.warehouse_code LIKE '%s%%' 
    AND pcr.goods_code='%s'""" % (wh_code,sku)
    db.cursor()
    # print_in_log(resource_sql)
    resource = pd.read_sql(resource_sql, db)
    db.close()
    if resource.empty ==True:
        print_in_log('该资源规格box_gauge为空,选择历史的装箱规格')
        resource_conver_rate = resource_convert(sku)
        Munit = '箱'
    elif resource['box_gauge'].iloc[0] == '0':
        print_in_log('该资源规格box_gauge为0，,选择历史的装箱规格')
        resource_conver_rate = resource_convert(sku)
        Munit = resource['unit'].iloc[0]
    elif resource['box_gauge'].iloc[0] == '1' :
        print_in_log('该资源规格box_gauge为1,选择历史的装箱规格')
        resource_conver_rate = resource_convert(sku)
        Munit = resource['unit'].iloc[0]
    else:
        resource_conver_rate = resource['box_gauge'].iloc[0]
        Munit = resource['unit'].iloc[0]
    resource_conver_rate = float(resource_conver_rate)
    resource_conver_rate = round(resource_conver_rate)
    resource_conver_rate = int(resource_conver_rate)
    return resource_conver_rate,Munit



# 获取五位的商品码
def get_convert_rate(Id,wh_code,data):
    # print(data)
    data_sku_id = data[['Sku_code']]
    data_sku_id = data_sku_id.drop_duplicates()
    data_sku_id = data_sku_id.reset_index(drop=True)
    sku_id_list = data_sku_id['Sku_code'].to_list()
    sku_convert_rate = pd.DataFrame(columns={'Sku_code','rate'})
    # client.hincrby(Id, '20', 1)
    print_in_log('redis+1')
    for sku in sku_id_list:
        print_in_log('正在获取sku的装箱规格数据:' + str(sku))
        convert_rate,Munit = box_gauge(wh_code,sku)
        print_in_log('convert_rate'+ str(convert_rate))
        # sku_convert_rate = sku_convert_rate.append({'Sku_id': sku}, ignore_index=True)
        sku_convert_rate = sku_convert_rate.append({'rate': convert_rate,'Sku_code': sku,'Munit':Munit}, ignore_index=True)
    return sku_convert_rate

#将资源转换因子和产品相匹配
def get_final_forecast(data,sku_convert):
    # forecast = get_forecast(data)
    final = pd.merge(data,sku_convert,on='Sku_code',how='left')
    # final.to_csv('D:/AI/xianfengsg/online/V_1.2/final-11-21.csv',encoding='utf_8_sig')
    final['Forecast_box'] = final['Forecast_qty']/final['rate']
    final['Forecast_box'] = final['Forecast_box'].apply(lambda x: round(x))
    return final

#获取昨天，今天，明天和后天共四天的日期str
def get_all_date(Id,today_date):
    today_date_datetime = datetime.datetime.strptime(today_date, '%Y-%m-%d')
    yes_date = (today_date_datetime - datetime.timedelta(1)).strftime('%Y-%m-%d')
    tomorrow_date = (today_date_datetime + datetime.timedelta(1)).strftime('%Y-%m-%d')
    TDAT_date = (today_date_datetime + datetime.timedelta(2)).strftime('%Y-%m-%d')

    return yes_date,today_date,tomorrow_date,TDAT_date

#---------------------------------------------------获取预测数据
def get_original_forecast(wh_code,today_date):
    print_in_log('连接到mysql服务器...')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123", port=3306, charset='utf8')
    print_in_log('连接成功,开始读取预测数据')
    forecast_sql = """SELECT * FROM dc_forecast df WHERE df.Dc_code='%s' AND df.Update_time = DATE('%s')"""%(wh_code,today_date)
    db.cursor()
    read_original_forecast = pd.read_sql(forecast_sql, db)
    print_in_log('连接成功,预测数据读取完成')
    db.close()
    return read_original_forecast


'''仿真环境中，主要的区别在于每日库存的变化和缺货的统计等'''
def get_stock(wh_code,today_date):
    print('stock_new-----日期'+today_date)
    stock_new = pd.read_csv('./stock_new' + str(today_date) + '.csv',encoding='utf_8_sig',
                            converters = {u'Sku_code':str},index_col=False)
    stock_new =stock_new[['Sku_code','Sku_name','Stock_new']]
    stock_new = stock_new.rename(index=str, columns={'Stock_new': 'Stock'})
    order_shop = get_shop_order(today_date)
    order_shop = order_shop.rename(index=str, columns={'SKU_CODE': 'Sku_code'})
    order_shop['Sku_code'] = order_shop['Sku_code'].astype(str)
    order_shop['Sku_code'] = order_shop['Sku_code'].astype(str)
    stock_total = pd.merge(stock_new,order_shop, on='Sku_code',how='left')
    stock_total.fillna(0)
    stock_total['Inventory'] = stock_total['Stock'] - stock_total['ORDER_SUM']
    record = stock_total
    record['stockout'] = record['Inventory']
    record['stockout'] = np.where(record['stockout']>0, 0, record['stockout'])
    record.to_csv('./record'+str(today_date)+'.csv',encoding='utf_8_sig')

    return stock_total


'''记录当天的门店的总订货数量'''
def get_shop_order(today_date):
    host = "192.168.1.11"  # 数据库ip
    port = "1521"  # 端口
    sid = "hdapp"  # 数据库名称
    parameters = cx_Oracle.makedsn(host, port, sid)
    # hd40是数据用户名，xfsg0515pos是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", parameters)
    #查看详细的出库数据，进行了日期的筛选，查看销量签50名的SKU
    order_shop_sql = """SELECT oep.GOODSCODE Sku_code,oep.NEWQTY order_sum,oep.QPCSTR FROM OA_EXAMINE_PREALCPOOL_BAK oep INNER JOIN
  (SELECT s.CODE FROM STORE s WHERE s.AREA LIKE '3301%%' AND s.STAT ='0')b ON
  b.code = oep.STOREACODE WHERE oep.DT = to_date('%s','yyyy-mm-dd')
  ORDER BY oep.QPCSTR DESC"""%(today_date)
    order_shop = pd.read_sql(order_shop_sql, conn)
    order_shop = order_shop.rename(index=str, columns={'SKU_CODE': 'Sku_code'})
    #将SKU的的iD转成list，并保存前80个，再返回值
    conn.close
    '''再读取门店订货的时候，直接进行规格的转换'''
    order_shop_new = order_shop['QPCSTR'].str.split(pat=r'*', n=2, expand=True)
    order_shop_new.columns = ['max', 'rate']
    order_shop_new['rate'] = order_shop_new['rate'].astype(float)
    final = pd.concat([order_shop,order_shop_new],join="outer",axis=1)
    final['order_sum_box'] = final['ORDER_SUM'] / final['rate']
    final['order_sum_box'] = final['order_sum_box'].apply(lambda x: round(x))
    final = final[['Sku_code','order_sum_box']]
    final = final.groupby(['Sku_code'], as_index=False).agg(sum)
    final = final.rename(index=str, columns={'Sku_code': 'SKU_CODE','order_sum_box':'ORDER_SUM'})








    #
    # order_shop = order_shop.rename(index=str, columns={'SKU_CODE': 'Sku_code'})
    # Id = '001'
    # wh_code = '001'
    # sku_convert = get_convert_rate(Id,wh_code,order_shop)
    # final = pd.merge(order_shop,sku_convert,on='Sku_code',how='left')
    # final.to_csv('./final'+(today_date)+'.csv',encoding='utf_8_sig')
    # final['order_sum_box'] = final['ORDER_SUM'] / final['rate']
    # final['order_sum_box'] = final['order_sum_box'].apply(lambda x: round(x))
    # final = final[['Sku_code','order_sum_box']]
    # final = final.rename(index=str, columns={'Sku_code': 'SKU_CODE','order_sum_box':'ORDER_SUM'})
    return final


'''该设置的参数是在试水阶段采用特定的SKU进行补货的方式和方法'''
#--------------------------------------------------------先设置需要的SKU，ssd=selected_sku_dataframe
def test_sku(data):
    ssd = pd.DataFrame({'goods_code':['65550','07540','07310','11620','11600','16010','16040','07350','13160','06390',
                                      '05200','01310','08890','05020','07640','11120','06850','65770','07600','12190',
                                      '01270','07340','07300','11650','07950','11710','12130','01020','07220','12240'
                                      ]})
    result_data = pd.merge(data,ssd,on='Sku_code',how='inner')
    return result_data

#设置主函数用于读取数据库的库存和预测数据
def get_db(Id,wh_code,today_date):

    stock_data = get_stock(wh_code,today_date)
    # stock_data = pd.read_excel('./test_stock.xlsx',index_col=False,converters={u'goods_code': str})
    #先采取全部sku进行使用的方式，如业务部门不进行
    # stock_data = test_sku(stock_data)
    original_forecast = get_original_forecast(wh_code,today_date)

    return stock_data,original_forecast


# def get_final_forecast(data,sku_convert):
#     # forecast = get_forecast(data)
#     final = pd.merge(data,sku_convert,on='Sku_code',how='left')
#     # final.to_csv('D:/AI/xianfengsg/online/V_1.2/final-11-21.csv',encoding='utf_8_sig')
#     final['Forecast_box'] = final['Forecast_qty']/final['rate']
#     final['Forecast_box'] = final['Forecast_box'].apply(lambda x: round(x))
#     return final

#-----------------------------------------------这里是进行编码转换，基本的数据
def cleaning_data(Id,wh_code,original_forecast,stock_data):
    sku_convert_rate = get_convert_rate(Id,wh_code,original_forecast)
    forecast_data_final = get_final_forecast(original_forecast,sku_convert_rate)
    print_in_log('数据清洗完成')
    return stock_data,forecast_data_final


'''分仓采购场景中默认提前期是一天，对于协同的压力将会在统采的嘉兴仓中产生'''
#=========================================================获取在途库存的数据
def algorithm_SS(Id,final, tomorrow_date, TDAT_date):
    #=================================================以下SS的计算,K取值2
    Code = list(set(final['Sku_code'].tolist()))
    data_SS = pd.DataFrame()
    for i in Code:
        data_mid = final[final['Sku_code'] == i]
        forecast = data_mid['Forecast_box'].values
        #因为设置较大的安全库存的关系，在原有的基础上增加SS
        std = 2.56 * (np.sqrt(((forecast - np.mean(forecast)) ** 2).sum() / (forecast.size)))
        data_SS = data_SS.append({'Sku_code': i,'SS': std,}, ignore_index=True)

    # print(data_SS)
    data_merge = pd.merge(final,data_SS,on='Sku_code',how='left')
    data_merge.fillna(method='ffill')
    data_final = data_merge[data_merge['Account_date'] == TDAT_date ]

    #如下是获得第二天的预测补货值，对二配行为进行部分规避
    forecast_tomorrow = final[final['Account_date'] == tomorrow_date]
    forecast_tomorrow = forecast_tomorrow[['Sku_code','Forecast_box']]
    forecast_tomorrow= forecast_tomorrow.rename(index=str, columns={'Forecast_box': 'Forecast_box_tomorrow'})
    data_final = pd.merge(data_final,forecast_tomorrow,on='Sku_code',how='inner')
    def compare(x):
        if x['Sku_code'] == '16040':
            return 0
        elif x['Sku_code'] == '16010':
            return 0
        else:
            #防止因为计算原因导致安全库存过大，如果安全库存设置过大，那么最多只保留一天的量
            if x['SS'] > x['Forecast_box']:
                qty = x['Forecast_box']
                return qty
            else:
                return x['SS']
    print_in_log('data_final_length:' + str(len(data_final)))
    data_final['SS'] = data_final.apply(lambda x: compare(x), axis=1)
    data_final = data_final[['Sku_id','Sku_code','Dc_name','Dc_code','Wrh','Munit',
                             'Warehouse_name','Sku_name','rate','Forecast_box','SS','Forecast_box_tomorrow']]
    return data_final

def main_function(Id,data_stock,data_final,today_date_str):
    data_stock['Sku_code'] = data_stock['Sku_code'].astype(str)
    data_stock = data_stock[['Sku_code','Stock','Inventory']]
    '''选择外连接合并，是因为存在仓库的表格并没有数据，但是实际情况是需要订货的，因此采取外连接，并进行补零操作'''
    '''最终选择采用内连接的方式，在于产品进行测试的阶段，先对一些指定的SKU进行计算'''
    data_merge = pd.merge(data_final,data_stock,on='Sku_code',how='inner')
    data_merge= data_merge.fillna(0)
    data_merge.to_csv('./data_merge_old.csv', encoding='utf_8_sig')
    #在加一个逻辑是基于昨日销量的一个对预测值的修正情况，如果预测的数量小于实际的销售数量的话，将会在预测的时候进行未来那天的修正
    def predict_revised(x):
        real_sales = x['Stock'] - x['Inventory']
        if real_sales < x['Forecast_box_tomorrow']:
            return x['Forecast_box']
        else:
            return x['Forecast_box'] + (real_sales - x['Forecast_box_tomorrow'] )
    data_merge['Forecast_box'] = data_merge.apply \
        (lambda x: predict_revised(x), axis=1)
    def predictt_tomorrow_revised(x):
        real_sales = x['Stock'] - x['Inventory']
        if real_sales <= x['Forecast_box']:
            return x['Forecast_box']
        else:
            return x['Forecast_box'] + (real_sales - x['Forecast_box'] )
    data_merge['Forecast_box'] = data_merge.apply \
        (lambda x: predictt_tomorrow_revised(x), axis=1)
    '''目前预测算法中还未进行活动的特征提取，因此先基于规则制定针对大型活动的策略'''
    if today_date_str == '2019-12-08' or today_date_str == '2019-12-09':
        print('需要为双十二单独备货')
        data_merge['Forecast_box'] = data_merge['Forecast_box'] * 1.8
    else:
        pass
    data_merge.to_csv('./data_merge.csv',encoding='utf_8_sig')
    #判断实际需要的的数量，
    def calculate_final(x):
        foreacst_demand = x['Forecast_box'] + x['SS']
        if foreacst_demand <= x['Inventory']:
            return 0
        else:
            return round(foreacst_demand - x['Inventory'])
    data_merge['Suggestion_qty'] = data_merge.apply \
        (lambda x: calculate_final(x), axis=1)


    # data_merge['Suggestion_qty'].iloc[np.where(data_merge["Sku_code"] == '16040')] = \
    #     np.array(data_merge['Suggestion_qty'].iloc[np.where(data_merge["Sku_code"] == '16010')]/3)
    data_merge['Update_time'] = pd.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data_merge['Account_date'] = today_date_str
    data_merge = data_merge.drop(['Forecast_box_tomorrow','Stock'], axis=1)
    return data_merge




def connectdb():
    print_in_log('连接到mysql服务器...')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123",port=3306, charset='utf8')
    print_in_log('连接成功')
    return db

#测试的地址
# def connectdb():
#     print_in_log('连接到mysql服务器...')
#     db = pymysql.connect(host="rm-bp1d0oo43ew2r861feo.mysql.rds.aliyuncs.com",
#                          database="purchare_test", user="purchare_test",
#                          password="purchare_test@123",port=3306, charset='utf8')
#     print_in_log('连接成功')
#     return db


#《---------------------------------------------------------------------------------------------------------------------删除重复日期数据
def drop_data(Id,db,wh_code,today_date):
    cursor = db.cursor()
    # date_parameter = datetime.date.today().strftime('%Y-%m-%d')
    # sql = """delete from dc_replenishment"""
    sql = """delete from dc_replenishment where Dc_code = '%s' and Account_date = DATE('%s')  """%(wh_code,today_date)
    print_in_log('已经删除重复数据')
    print_in_log(str(sql))

    cursor.execute(sql)

#<======================================================================================================================
def insertdb(Id,db,data):
    cursor = db.cursor()
    # param = list(map(tuple, np.array(data).tolist()))
    data_list = data.values.tolist()
    print_in_log('data_list'+str(data_list))
    sql = """INSERT INTO dc_replenishment (Sku_id,Sku_code,Dc_name,Dc_code,Wrh,Munit,Warehouse_name,Sku_name,rate,Forecast_box,
    	SS,Stock,Suggestion_qty,Update_time,Account_date)
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

#定义主计算调度逻辑进行函数的汇总
def main(redis_key,wh_code,today_date_str):
    # client.hset(redis_key,'20',0)
    time_start = datetime.datetime.now()
    yes_date, today_date, tomorrow_date, TDAT_date = get_all_date(redis_key,today_date_str)
    stock_data,original_forecast = get_db(redis_key,wh_code,today_date)
    data_stock, final = cleaning_data(redis_key,wh_code,original_forecast, stock_data)
    data_final = algorithm_SS(redis_key,final, tomorrow_date, TDAT_date)
    AI_suggestion = main_function(redis_key,data_stock, data_final,today_date_str)
    print_in_log('AI_suggestion')
    db = connectdb()
    drop_data(redis_key,db,wh_code,today_date)
    time_end = datetime.datetime.now()
    AI_suggestion.to_csv('./AI_suggestion' + str(today_date_str) + '.csv', encoding='utf_8_sig')
    AI_suggestion_merge = AI_suggestion[['Suggestion_qty', 'Sku_code']]
    # AI_suggestion_merge = AI_suggestion_merge.rename(index=str, columns={'Sku_code': 'goods_code'})
    stock_new = pd.merge(data_stock, AI_suggestion_merge, on='Sku_code', how='inner')
    stock_new['Stock_new'] = stock_new['Inventory'] + stock_new['Suggestion_qty']
    stock_new.to_csv('./stock_new' + str(tomorrow_date) + '.csv', encoding='utf_8_sig')
#
def get_parameter():
    redis_key = sys.argv[1]
    wh_code = sys.argv[2]
    today_date_str = sys.argv[3]
    return redis_key,wh_code,today_date_str

#《============================================================================主函数入口
if __name__ == '__main__':
    try:
        # redis_key,wh_code,today_date_str = get_parameter()
        redis_key = 'test'
        wh_code = '001'
        today_date_str = '2019-12-03'
        main(redis_key,wh_code,today_date_str)
    except OSError as reason:
        print_in_log('出错原因是%s'%str(reason))
        print_in_log ("result:0")




