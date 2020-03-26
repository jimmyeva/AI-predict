# -*- coding: utf-8 -*-
# @Time    : 2020/1/14 11:34
# @Author  : Ye Jinyu__jimmy
# @File    : simulation.py



import sys
print(sys.version)
import pandas as pd
import cx_Oracle
import os
import numpy as np

'''设置仿真，2019-12-01----2020-01-01'''
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
import decision_repl_2nd
import decision_repl


'''仿真算法的主逻辑'''


#设置时间序列用于仿真的起始日期进行计算，先跑了每日的决策数据
def main_run_simulation():
    start = '2019-11-30'
    end = '2020-01-01'
    id = '001'
    wh_code = '001'
    for days in pd.date_range(start, end):
        day = days.strftime('%Y-%m-%d')
        print(day)
        if day == start:
            print('第一天')
            decision_repl.main(id,wh_code,day)
        else:
            decision_repl_2nd.main(id, wh_code, day)
            print('不是第一天')
        print(day)

# main_run_simulation()

'''如下是评估具体的指标，库存周转率，和缺货率这两个指标，程序需要进行两步处理，先是计算缺货率，然后修正销售量与期末库存，在进行
库存周转率，周转天数的指标衡量'''

#先获取sku的价格，与产品中心人确认按照Excel的价格走，叫货目录里面价格是由人工维护，叫货目录里面的是不准的
def get_goods_price():
    df = pd.read_excel('D:/jimmy-ye/AI/AI_supply_chain/product_design/DATA_SOURCE'
                       '/product_center_rel/浙北仓库存模板12.24.xlsx',header=1)  #,converters={u'订货编码': str}
    data_new = df.dropna(subset=['订货编码'],inplace=False)
    data_new = data_new[['订货编码','箱价']]
    return data_new

data_new = get_goods_price()

def kpi_AI():
    start = '2019-12-02'
    end = '2020-01-01'
    final_data = pd.DataFrame()
    for days in pd.date_range(start, end):
        day = days.strftime('%Y-%m-%d')
        print(day)
        stock_new = pd.read_csv('./record' + str(day) + '.csv', encoding='utf_8_sig',index_col=0)
        print(stock_new)
        #先删除空值的数据
        stock_new.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
        '''删除特殊的春蕉和进口的商品'''

        stock_new = stock_new[~stock_new['Sku_code'].isin(['16010','16040'])]
        print('stock_new','\n',stock_new)


        #先对价格进行合并
        data_price = data_new.rename(index=str, columns = {'订货编码': 'Sku_code'})
        stock_new = pd.merge(stock_new,data_price,on='Sku_code',how='inner')
        #先计算缺货率
        def shortage_stock(x):
            if x['stockout'] < 0:
                shortage_stock_rate = (x['stockout'] / x['ORDER_SUM']) * -1
            else:
                shortage_stock_rate = 0
            return shortage_stock_rate
        stock_new['缺货率'] = stock_new.apply \
            (lambda x: shortage_stock(x), axis=1)


        #计算有的sku存在的缺货，需要对真实销售进行修正
        def real_sales(x):
            if x['stockout'] < 0:
                real_sales = x['Stock']

            else:
                real_sales = x['ORDER_SUM']
            return real_sales
        stock_new['ORDER_SUM'] = stock_new.apply \
            (lambda x: real_sales(x), axis=1)

        #对如果有发生缺货的sku进行修正计算处理
        def inventory_revised(x):
            if x['Inventory'] < 0:
                return 0
            else:
                return x['Inventory']
        stock_new['Inventory'] = stock_new.apply \
            (lambda x: inventory_revised(x), axis=1)
        '''资金占用成本'''
        def Rocc(x):
            rocc = x['Inventory'] * x['箱价'] * 1.006
            return rocc
        stock_new['资金占用成本'] = stock_new.apply \
            (lambda x: Rocc(x), axis=1)
        '''库存管理成本'''
        def stock_cost(x):
            mean_stock = (x['Stock'] + x['Inventory']) / 2
            cost = mean_stock * x['箱价'] * 0.001
            return cost
        stock_new['库存管理成本'] = stock_new.apply \
            (lambda x: stock_cost(x), axis=1)
        '''商品折旧成本'''
        def depreciation_cost(x):
            depreciation_cost = x['Inventory'] * x['箱价'] * 0.05
            return depreciation_cost
        stock_new['商品价值损耗成本'] = stock_new.apply \
            (lambda x: depreciation_cost(x), axis=1)
        '''计算缺货成本'''
        def shortage_stock_cost(x):
            if x['stockout'] < 0:
                shortage_stock_cost_price = x['stockout'] * x['箱价'] * 1.07 * -1
            else:
                shortage_stock_cost_price = 0
            return shortage_stock_cost_price
        stock_new['缺货成本'] = stock_new.apply \
            (lambda x: shortage_stock_cost(x), axis=1)
        '''计算销售额'''
        def sales_price(x):
            sales = x['ORDER_SUM'] * x['箱价']
            return sales
        stock_new['销售额'] = stock_new.apply \
            (lambda x: sales_price(x), axis=1)



        #计算库存周转率和周转天数
        def ito(x):
            ito = x['ORDER_SUM']/(x['Stock'] + x['Inventory'])
            return ito
        stock_new['ito'] = stock_new.apply \
            (lambda x: ito(x), axis=1)

        '''计算周转天数'''
        def ito_day(x):
            if x['ito'] == 0:
                ito_day = 0
            else:
                ito_day =  1 / x['ito']
            return ito_day
        stock_new['ito_day'] = stock_new.apply \
            (lambda x: ito_day(x), axis = 1)
        stock_new['time'] = day
        stock_new.to_csv('./stock_kpi'+str(day)+'.csv',encoding='utf_8_sig')
        final_data = final_data.append(stock_new)
    final_data= final_data.fillna(0)
    final_data.to_csv('./final_data.csv',encoding='utf_8_sig')
    finance_data = finance_index(final_data)
    finance_data.to_csv('./finance_data_AI.csv', encoding='utf_8_sig')




#定义函数用来聚合计算财务指标
def finance_index(data):
    data_finance = pd.DataFrame(columns = ['缺货率','资金占用成本','库存管理成本','商品价值损耗成本',
                                           '缺货成本','销售额','库存周转率','库存周转天数'])
    data_finance["缺货率"] = data.groupby(["time"]).mean()["缺货率"]
    data_finance["资金占用成本"] = data.groupby(["time"]).sum()["资金占用成本"]
    data_finance["库存管理成本"] = data.groupby(["time"]).sum()["库存管理成本"]
    data_finance["商品价值损耗成本"] = data.groupby(["time"]).sum()["商品价值损耗成本"]
    data_finance["缺货成本"] = data.groupby(["time"]).sum()["缺货成本"]
    data_finance["销售额"] = data.groupby(["time"]).sum()["销售额"]
    data_finance["库存周转率"] = data.groupby(["time"]).mean()["ito"]
    data_finance["库存周转天数"] = data.groupby(["time"]).mean()["ito_day"]
    data_finance['总成本'] = data_finance['资金占用成本'] + data_finance['库存管理成本'] +\
                          data_finance['商品价值损耗成本'] + data_finance['缺货成本']
    return data_finance

    # data_daterbind["account_date"]=data.groupby(["account_date"]).sum().index
    # data_daterbind["piece_bar_code"] = [data["piece_bar_code"].iloc[0]]*len(data_daterbind["delivery_qty"])





#先是计算AI补货的KPI指标
kpi_AI()

#以下是对人工补货进行计算
def kpi_human():
    start = '2019-12-01'
    end = '2019-12-31'

    final_data = pd.DataFrame()
    for days in pd.date_range(start, end):
        day = days.strftime('%Y-%m-%d')
        print(day)
    # day = '2019-12-01'
        df = pd.read_excel('D:/jimmy-ye/AI/AI_supply_chain/product_design/DATA_SOURCE/product_center_rel/test'
                           +str(day)+'.xlsx',converters = {u'订货编码':str})
        ssd = pd.DataFrame(
            {'goods_code': ['65550', '07540', '07310', '11620', '11600',  '07350', '13160', '06390',
                            '05200', '01310', '08890', '05020', '07640', '11120', '06850', '65770', '07600', '12190',
                            '01270', '07340', '07300', '11650', '07950', '11710', '12130', '01020', '07220', '12240'
                            ]}) #  '16010', '16040',

        df = df.rename(index=str, columns={'订货编码': 'goods_code','库存': 'Stock','差数': 'Inventory'})
        result = pd.merge(df,ssd,on='goods_code',how='inner')

        data_price = data_new.rename(index=str, columns={'订货编码': 'goods_code'})
        print(result)
        print(data_price)
        result = pd.merge(result,data_price,on='goods_code',how='inner')
        print(result)

        result['ORDER_SUM'] = result['Stock'] - result['Inventory']
        result['stockout'] = result['Inventory']
        result['stockout'] = np.where(result['stockout'] > 0, 0, result['stockout'])
        def shortage_stock(x):
            if x['stockout'] < 0:
                shortage_stock_rate = (x['stockout'] / x['ORDER_SUM']) * -1
            else:
                shortage_stock_rate = 0
            return shortage_stock_rate
        result['缺货率'] = result.apply \
            (lambda x: shortage_stock(x), axis=1)

        #计算有的sku存在的缺货，需要对真实销售进行修正
        def real_sales(x):
            if x['stockout'] < 0:
                real_sales = x['Stock']

            else:
                real_sales = x['ORDER_SUM']
            return real_sales
        result['ORDER_SUM'] = result.apply \
            (lambda x: real_sales(x), axis=1)

        # 对如果有发生缺货的sku进行修正计算处理
        def inventort_revised(x):
            if x['Inventory'] < 0:
                return 0
            else:
                return x['Inventory']

        result['Inventory'] = result.apply \
            (lambda x: inventort_revised(x), axis=1)

        print(result)
        # 计算库存周转率和周转天数
        def ito(x):
            if x['Stock'] + x['Inventory'] == 0:
                ito = 0
            else:
                ito = x['ORDER_SUM'] / (x['Stock'] + x['Inventory'])
            return ito
        result['ito'] = result.apply \
            (lambda x: ito(x), axis=1)

        # 计算周转天数
        def ito_day(x):
            if x['ito'] == 0:
                ito_day = 0
            else:
                ito_day = 1 / x['ito']
            return ito_day
        result['ito_day'] = result.apply \
            (lambda x: ito_day(x), axis=1)


        #对如果有发生缺货的sku进行修正计算处理
        def inventory_revised(x):
            if x['Inventory'] < 0:
                return 0
            else:
                return x['Inventory']
        result['Inventory'] = result.apply \
            (lambda x: inventory_revised(x), axis=1)
        '''资金占用成本'''
        def Rocc(x):
            rocc = x['Inventory'] * x['箱价'] * 1.006
            return rocc
        result['资金占用成本'] = result.apply \
            (lambda x: Rocc(x), axis=1)
        '''库存管理成本'''
        def stock_cost(x):
            mean_stock = (x['Stock'] + x['Inventory']) / 2
            cost = mean_stock * x['箱价'] * 0.001
            return cost
        result['库存管理成本'] = result.apply \
            (lambda x: stock_cost(x), axis=1)
        '''商品折旧成本'''
        def depreciation_cost(x):
            depreciation_cost = x['Inventory'] * x['箱价'] * 0.05
            return depreciation_cost
        result['商品价值损耗成本'] = result.apply \
            (lambda x: depreciation_cost(x), axis=1)
        '''计算缺货成本'''
        def shortage_stock_cost(x):
            if x['stockout'] < 0:
                shortage_stock_cost_price = x['stockout'] * x['箱价'] * 1.07 * -1
            else:
                shortage_stock_cost_price = 0
            return shortage_stock_cost_price
        result['缺货成本'] = result.apply \
            (lambda x: shortage_stock_cost(x), axis=1)
        '''计算销售额'''
        def sales_price(x):
            sales_qty = x['Stock'] - x['Inventory']
            sales = sales_qty * x['箱价']
            return sales
        result['销售额'] = result.apply \
            (lambda x: sales_price(x), axis=1)

        result['time'] = day
        result.to_csv('./stock_kpi_人工' + str(day) + '.csv', encoding='utf_8_sig')
        final_data = final_data.append(result)
    final_data.to_csv('./final_data_人工.csv',encoding='utf_8_sig')
    finance_data = finance_index(final_data)
    finance_data.to_csv('./finance_data_人工.csv', encoding='utf_8_sig')

#运行人工的kpi计算
# kpi_human()

