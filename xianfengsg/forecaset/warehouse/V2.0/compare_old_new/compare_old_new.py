# -*- coding: utf-8 -*-
# @Time    : 2020/3/22 11:56
# @Author  : Ye Jinyu__jimmy
# @File    : compare_old_new

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib as mpl
mpl.use('Agg')
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
from matplotlib.pylab import rcParams
import psycopg2
import pymysql
import time
import datetime
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

'''这里是将新版本的预测算法与旧版本的预测算法，并结合真实的历史销售做对比；
从数据库读取预测销售数据'''


import data_process             #导入数据处理
import feature_engineing        #导入特征合并
import price_feature            #导入价格特征
import time_feature             #导入时间特征
import weather_feature          #导入天气特征
import get_holiday              #导入节假日爬取函数
import sales_feature            #导入销售特征处理函数
import forecast_model           #导入预测函数


#------------------------------------------------------>先读取基本的仓库的销售数据<-------------------------------------------
def get_real_sales_data(wh_code, sku_code, start_date):
    conn = psycopg2.connect(database="dc_rpt", user="ads", password="ads@xfsg2019", host="192.168.1.205",
                            port="3433")
    print("Opened database successfully,connected with PG DB,read sales data")
    ads_rpt_ai_wh_d_sql = """SELECT stat_date,sal_qty_1d FROM ads_aig_supply_chain.ads_rpt_ai_wh_d 
    WHERE wh_code ='%s' AND sty_code = '%s' 
                         AND stat_date >'%s'""" % \
                          (wh_code, sku_code, start_date)
    try:
        wh_sales            = pd.read_sql(ads_rpt_ai_wh_d_sql, conn)
    except:
        print("load data from postgres failure !")
        wh_sales            = pd.DataFrame()
        exit()
    conn.close()
    wh_sales['stat_date']   = pd.to_datetime(wh_sales['stat_date'])
    wh_sales                = wh_sales.rename(index=str, columns={'stat_date': 'account_date'})
    #===============================================方便后面程序的字段匹配，这里先进行字段匹配
    wh_sales.columns                = ['Account_date','sales_qty']
    print(str(sku_code) + '销售数据读取完成,数据长度为%d'%(len(wh_sales)))
    return wh_sales



'''该脚本函数是用于将分别对每个城市公司的销售情况进行总预测是主调度逻辑'''
#----------------------------------------------------->设置函数获取<--------------------------------------------------------
def get_wh_list():
    conn = psycopg2.connect(database="dc_rpt", user="ads", password="ads@xfsg2019", host="192.168.1.205",
                            port="3433")
    print("Opened database successfully,connected with PG DB")
    wh_code_sql = """SELECT wh_code,wh_name FROM ads_aig_supply_chain.ads_rpt_ai_wh_d GROUP BY wh_code,wh_name """
    try:
        wh_df               = pd.read_sql(wh_code_sql, conn)
    except:
        print("load data from postgres failure !")
        wh_df               = pd.DataFrame()
        exit()
    conn.close()
    return wh_df

#---------------------------------------------------->最新的逻辑是从叫货目录进行选择<---------------------------
def get_order_code(wh_code,wh_name):
    print('正在读取叫货目录的数据')
    dbconn = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com", database="purchare_sys",
                             user="purchare_sys", password="purchare_sys@123", port=3306,
                             charset='utf8')
    get_orders = """SELECT pcr.goods_code GOODS_CODE,pcr.goods_name FROM 
    p_call_record pcr WHERE pcr.warehouse_code  LIKE '%s%%' group by pcr.goods_code """%\
                 (wh_code)
    orders = pd.read_sql(get_orders, dbconn)
    dbconn.close()
    orders['GOODS_CODE']  =      orders['GOODS_CODE'].astype(str)
    code_list             =      orders['GOODS_CODE'].to_list()
    name_list             =      orders['goods_name'].to_list()
    print(str(wh_name)+',叫货目录读取完成,共有商品%s'%(len(code_list)))
    return code_list,name_list

#==============================================================单个SKU计算的主逻辑================================================
def each_wh_main(wh_code,wh_name,start_date, end_date,city_name):
    print('wh_code,wh_name,start_date, end_date,city_name',wh_code,wh_name,start_date, end_date,city_name)
    code_list,name_list = get_order_code(wh_code,wh_name)
    #——————————————————暂时从本地取数据————————
    # holiday_data = get_holiday.get_holiday_function()
    holiday_data = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/data_date.csv', encoding='utf_8_sig')
    #获取销售函数的类
    sales_class = data_process.sales_cleaning()
    #时间函数的类
    time_class = time_feature.time_process()
    #价格函数的类
    price_class = price_feature.price_process()
    # ---------------------------------
    time_df = time_class.main(holiday_data)
    # ---------------------------------
    weather_df = weather_feature.get_weather_features('杭州')
    total_predict = pd.DataFrame()
    compare_total = pd.DataFrame()
    for i in range(85,len(code_list)):
    # for i in range(100,200):
    # i  = 0
        code = code_list[i]
        name = name_list[i]
        print('正在进行预测计算的是：' + str(name))
        #得到历史的销售数据
        df_sales                = get_real_sales_data('001',code,start_date)
        print('销售数据的总长度是：%d'%len(df_sales))
        if len(df_sales) < 5  or len(df_sales[ ~ df_sales['sales_qty'].isin([0])]) <= 5 :
            pass
        else:
            sales               = sales_class.data_cleaning(df_sales)
            sales_df            = sales_class.fill_date(sales,start_date,end_date)
            # ---------------------------------
            price_df            = price_class.price_main(end_date,sales_df,code,city_name)
            if price_df.empty == True:
                print('价格序列为空')
                pass
            else:
                #-------------------------------------------------
                sales_fea_df            = sales_feature.feature(sales)
                merge                   = feature_engineing.data_prepare(weather_df,sales_fea_df,price_df,time_df)
                #--------------------------------------------开始预测----------------------------------------------------
                # end_date                = (datetime.datetime.now()-datetime.timedelta(8)).strftime('%Y%m%d')
                predict_df              = forecast_model.forecast_merge(merge, end_date)
                predict_df['wh_name']   = wh_name
                predict_df['wh_code']   = wh_code
                predict_df['sku_code']  = code
                predict_df['sku_name']  = name
                print('end_date',end_date)

                #--------------------------------------------读取老版本的预测值-------------------------------------------
                read_original_forecast  = get_old_predict(wh_code, end_date,code)

                read_original_forecast.rename(columns={'Sku_code':'sku_code', 'Forecast_qty':'forecast_old'}, inplace = True)
                predict_new   = pd.merge(predict_df,read_original_forecast,on=['Account_date'],how='left')
                predict_new   = predict_new.fillna(0)
                #-------------------将每次合并的SKU进行添加保存---------------------------
                total_predict = total_predict.append(predict_new)
                compare_df    = compare_real(wh_code,code,name,end_date,predict_new)
                #------------------

                compare_df.to_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/compare_df'+str(name)+'.csv',encoding='utf_8_sig',index=False)
                compare_total = compare_total.append(compare_df)
                index_value = compare_total.groupby(['sku_name', 'wh_name', 'wh_code', 'sku_code']).mean()
                index_value.to_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/index_value.csv', encoding='utf_8_sig',
                               index=False)
        compare_total.to_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/compare_total.csv',encoding='utf_8_sig', index=False)
        total_predict.to_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/total_predict.csv',encoding='utf_8_sig',index=False)
    return total_predict


#---------------------------------------------------->设置函数读取老版本的预测值<--------------------------------------------
def get_old_predict(wh_code, today_date,code):
    print('连接到mysql服务器...')
    db = pymysql.connect(host="rm-bp1jfj82u002onh2t.mysql.rds.aliyuncs.com",
                         database="purchare_sys", user="purchare_sys",
                         password="purchare_sys@123", port=3306, charset='utf8')
    print('连接成功,开始读取预测数据')
    forecast_sql = """SELECT df.Account_date,df.Forecast_qty FROM 
    dc_forecast df WHERE df.Dc_code='%s' AND df.Update_time = DATE('%s') and df.Sku_code = '%s'
    """ % (wh_code, today_date,code)
    db.cursor()
    read_original_forecast = pd.read_sql(forecast_sql, db)
    print('连接成功,预测数据读取完成')
    db.close()
    return read_original_forecast



#---------------------------------------------------->读取真实的销售数据进行合并<------------------------------------------
def compare_real(wh_code,code,name,new_start,predict_df):
    conn = psycopg2.connect(database="dc_rpt", user="ads", password="ads@xfsg2019", host="192.168.1.205",
                            port="3433")
    print("Opened database successfully,connected with PG DB,read sales data")
    ads_rpt_ai_wh_d_sql = """SELECT stat_date,sal_qty_1d FROM ads_aig_supply_chain.ads_rpt_ai_wh_d 
                            WHERE wh_code ='%s' AND sty_code = '%s' AND stat_date >'%s'""" % \
                          (wh_code, code, new_start)
    try:
        wh_sales            = pd.read_sql(ads_rpt_ai_wh_d_sql, conn)
    except:
        print("load data from postgres failure !")
        wh_sales = pd.DataFrame()
        exit()
    conn.close()
    if wh_sales.empty == True:
        end_date = ( datetime.datetime.strptime(new_start, '%Y%m%d') + datetime.timedelta(8)).strftime('%Y%m%d')
        wh_sales            = pd.DataFrame({'Account_date': pd.date_range(new_start,end_date,  freq='D'),
                             'sales_qty': 0})
    else:
        pass
    print('wh_sales',name,code,new_start,len(wh_sales))

    wh_sales                        = wh_sales.rename(index=str, columns={'stat_date': 'Account_date'})
    wh_sales['Account_date']        = pd.to_datetime(wh_sales['Account_date']).dt.normalize()
    #方便后面程序的字段匹配，这里先进行字段匹配
    wh_sales.columns                = ['Account_date','sales_qty']
    predict_df_= predict_df.fillna(0)
    total_compare                   = pd.merge(predict_df_,wh_sales,on='Account_date',how='left')
    total_compare                   = total_compare[~(total_compare['forecast'].isnull())]
    total_compare['Account_date']   = pd.to_datetime(total_compare['Account_date']).dt.normalize()
    compare_df                      = description_error(total_compare)
    #---------------------------调用函数画图---------------------
    compare_plot(compare_df,code,name)

    return compare_df


#---------------------------------------------以下为画图的逻辑----------------------
def compare_plot(compare_df,code,name):
    print('compare_df',compare_df)

    df = compare_df[['Account_date', 'forecast_old', 'sales_qty', 'forecast']]

    df = df.fillna(0)
    df['Account_date'] = df['Account_date'].astype(str)
    df['forecast_old'] = df['forecast_old'].astype(int)
    df['forecast']     = df['forecast'].astype(int)
    df['sales_qty']    = df['sales_qty'].astype(int)

    print('df', df)
    print(type(df['Account_date'][2]))
    print(type(df['forecast_old'][2]))
    print(type(df['forecast'][2]))
    print(type(df['sales_qty'][3]))
    df.sort_values(by='Account_date', axis=0, ascending=True)
    plt.figure()
    font_size = 10  # 字体大小
    fig_size = (8, 6)  # 图表大小

    subjects = df.index.values  # 科目
    date = df['Account_date']
    predict_old = df['forecast_old']
    real_qty = df['sales_qty']
    predict = df['forecast']

    # 更新字体大小
    mpl.rcParams['font.size'] = font_size
    # 更新图表大小
    mpl.rcParams['figure.figsize'] = fig_size
    bar_width = 0.3

    index = np.arange(len(subjects))
    # 绘制「new」的成绩
    rects1 = plt.bar(index, predict, bar_width, color='#0072BC', label=u'New_forecast')
    # 绘制「old」的成绩
    rects2 = plt.bar(index + bar_width, predict_old, bar_width, color='#ED1C24', label=u'Old_forecast')
    # 绘制「real」的成绩
    plt.plot(date, real_qty, color='cyan', marker='o', linestyle='-.', label=u'Real',
             markerfacecolor='magenta', markersize=5)
    # X轴标题
    plt.xticks(index+bar_width, df['Account_date'], rotation=45)

    # 图表标题
    plt.title(u'forecast VS Real')
    # 图例显示在图表下方
    plt.legend(loc='upper left', fontsize=10,framealpha=0.5, fancybox=True, ncol=5)

    # 添加数据标签
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height, height,color='magenta', ha='center', va='center')
            # 柱形图边缘用白色填充，纯粹为了美观
            rect.set_edgecolor('white')

    # 设置数字标签
    for a, b in zip(date, real_qty):
        plt.text(a, b,b,ha='center',color='black', va='bottom',weight=20, fontsize=10)

    add_labels(rects1)
    add_labels(rects2)

    # 图表输出到本地
    plt.savefig("D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/" +
                str(code) +
                '_' + str(name) +
                '.jpg',
                dpi=300,
                bbox_inches='tight')
    plt.close()
    print('保存完成')



#----------------------------------------->查看和比较真实和预测数据的真实误差，并查看误差的统计学指标，保存到csv
def description_error(data):
    y               = data['sales_qty']
    yhat            = data['forecast']
    yhat_old        = data['forecast_old']
    n = len(y)
    data['MAE']     = np.abs(y-yhat)
    data['MAPE']    = np.fabs((y - yhat) / y)  / n
    data['RMSPE']   = (((y - yhat) ** 2)/ n)**0.5
    data['MAPE_old']= np.fabs((y - yhat_old) / y)  / n
    return data



#设置函数用于循环进行每个配送中心的单独运算
def main_function(start_date,end_date):
    wh_list = get_wh_list()
    wh_code_list = wh_list['wh_code'].to_list()
    wh_name_list = wh_list['wh_name'].to_list()
    num = len(wh_code_list)
    print('一共有%d个城市公司进行计算' % num)
    for i in range(num):
        wh_code = wh_code_list[i]
        wh_name = wh_name_list[i]
        print(start_date, end_date, wh_code)
        print('正在进行的城市是和代码是：', wh_name,wh_code)
        city_name = wh_name[0:2]
        data_sku = each_wh_main(wh_code,wh_name,start_date, end_date,city_name)
        print( wh_list['wh_name'].iloc[i],'计算完成')
        print('所有城市计算完成')



#——————————————————————————————主要的计单个维度的SKU的销量预测————————————
if __name__ == '__main__':
    wh_code= '001'
    wh_name ='杭州配送中心'
    start_date = '20180101'
    end_date = '20200318'
    city_name = '杭州'
    each_wh_main(wh_code, wh_name, start_date, end_date,city_name)