# -*- coding: utf-8 -*-
# @Time    : 2020/2/25 16:37
# @Author  : Ye Jinyu__jimmy
# @File    : evlautate_forecast


import pymysql
import pandas as pd
import numpy as np
import psycopg2
import datetime,time
from tqdm import tqdm
from pylab import *
plt.switch_backend('agg')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
#如下是支持中文数字
mpl.rcParams['font.sans-serif'] = ['SimHei']

'''该程序是用和查看预测的仓库销量与实际的销量的之间的差值'''



#——————————————————————————先根据仓库维度时间维度获取预测数据——————————————
def get_original_forecast(sku_code,wh_code,today_date):
    print('连接到mysql服务器...')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123", port=3306, charset='utf8')
    print('连接成功,开始读取预测数据')
    forecast_sql = """SELECT df.Sku_code,df.Forecast_qty,df.Account_date,df.Sku_name FROM dc_forecast df WHERE df.Dc_code='%s' 
            AND df.Update_time = DATE('%s') 
                    and df.Sku_code = '%s'"""%(wh_code,today_date,sku_code)
    db.cursor()
    read_original_forecast = pd.read_sql(forecast_sql, db)
    if read_original_forecast.empty == True:
        new_days = today_date.strftime('%Y%m%d')
        read_original_forecast = pd.DataFrame({'Account_date':[new_days],'Sku_code':[sku_code],'Forecast_qty':[0]})
    else:
        pass

    print('连接成功,预测数据读取完成')
    db.close()
    return read_original_forecast


#————————————————————————根据时间获取真实的销售数据————————————————————
def get_detail_sales_data(wh_code,sku_code,date):
    conn = psycopg2.connect(database="dc_rpt", user="ads", password="ads@xfsg2019", host="192.168.1.205", port="3433")
    print("Opened database successfully,connected with PG DB")
    ads_rpt_ai_wh_d_sql = """SELECT * FROM ads_aig_supply_chain.ads_rpt_ai_wh_d WHERE wh_code ='%s' AND sty_code = '%s' 
                         AND stat_date = '%s' """ % \
                        (wh_code,sku_code,date)
    try:
        wh_sales = pd.read_sql(ads_rpt_ai_wh_d_sql,conn)
    except:
        print("load data from postgres failure !")
        wh_sales = pd.DataFrame()
        exit()
    conn.close()
    if wh_sales.empty == True:
        wh_sales_new = pd.DataFrame({'Sku_code':[sku_code],'Real_sales':[0]})
    else:
        wh_sales_new = wh_sales[['sty_code', 'sal_qty_1d']]
        wh_sales_new.columns = ['Sku_code', 'Real_sales']

    print(str(sku_code)+'销售数据读取完成')
    return wh_sales_new

#这是从库存表中进行商品的选择,选择需要预测的sku的code
#-------------最新的逻辑是从叫货目录进行选择
def get_order_code(wh_code):
    print('正在读取叫货目录的数据')
    dbconn = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com", database="purchare_sys",
                             user="purchare_sys", password="purchare_sys@123", port=3306,
                             charset='utf8')
    get_orders = """  SELECT pcr.goods_code GOODS_CODE,pcr.goods_name GOODS_NAME FROM p_call_record pcr WHERE pcr.warehouse_code 
     LIKE '%s%%' GROUP BY pcr.goods_code"""%\
                 (wh_code)
    orders = pd.read_sql(get_orders, dbconn)
    orders['GOODS_CODE'] = orders['GOODS_CODE'].astype(str)
    print('叫货目录读取完成')
    dbconn.close()
    return orders

#————————————————————设置函数查看固定天数的销量MAE————————————————————
def compare_qty(day,wh_code,satrt_date,end_date):
    orders = get_order_code(wh_code)
    sku_list = orders['GOODS_CODE'].to_list()
    name_list = orders['GOODS_NAME'].to_list()
    for i in tqdm(range(len(orders))):
        code = sku_list[i]
        sku_name = name_list[i]
        print(code)
        data = pd.DataFrame()
        for days in pd.date_range(satrt_date, end_date):
            #得到对应SKU和日期维度的预测的销量
            forecast_data = get_original_forecast(code,wh_code,days)
            compare_date = days + datetime.timedelta(day)
            compare_data_forecast = forecast_data[forecast_data['Account_date'] == compare_date]
            #得到对应的真实的销量
            new_days = days.strftime('%Y%m%d')
            wh_sales = get_detail_sales_data(wh_code,code,new_days)
            single_compare = pd.merge(compare_data_forecast,wh_sales,on='Sku_code')
            data = data.append(single_compare)
        data['MAE'] = (data['Forecast_qty'] -data['Real_sales'])/(data['Real_sales'] + 0.1)
        data.to_csv('./'+str(code) +'_'+str(day)+'.csv',encoding='utf_8_sig')
        plot_function(data,code,sku_name,day)
        del data



#针对当前的数据形态设置一下画图并保存
def plot_function(data,code,sku_name,day):
    date = data['Account_date']
    real_qty = data['Real_sales']
    Forecast = data['Forecast_qty']
    fig = plt.figure(figsize=(20, 10), facecolor='white')
    ax1 = fig.add_subplot(111)
    # 左轴
    ax1.bar(date, real_qty, width=0.5, align='center', label='real_qty', color="black")
    ax1.plot(date, Forecast, color='red', marker = 'o', linestyle = 'dashed', label='forecast_qty',
             markersize=1.5)
    ax1.set_xlabel('date')
    ax1.set_ylabel('real_qty')

    plt.savefig("./" +
                str(code) +
                '_' + str(sku_name) +'_'+str(day)+ '.jpg', dpi=600,
                bbox_inches='tight')
    plt.close()


compare_qty(3 ,'001','2019-12-01','2020-02-01')