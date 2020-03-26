# -*- coding: utf-8 -*-
# @Time    : 2019/6/23 14:34
# @Author  : Ye Jinyu__jimmy
# @File    : stkoutdtl_download

import pandas as pd
import cx_Oracle
import os
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
# 注：设置环境编码方式，可解决读取数据库乱码问题
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
import math
import time
import multiprocessing
import datetime
from datetime import datetime,timedelta




#设置一个读取数据的函数
def query(table,table_01):
    host = "192.168.1.11"   #数据库ip
    port = "1521"        #端口
    sid = "hdapp"         #数据库名称
    dsn = cx_Oracle.makedsn(host, port, sid)

    # scott是数据用户名，tiger是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", dsn)

    # 这条sql是用来读取出库的数据
    stkout_sql = """SELECT s.NUM,s.OCRDATE,s.SENDER,s.CLIENT FROM STKOUT s WHERE s.CLS='统配出' AND
                s.OCRDATE >= to_date('%s','yyyy-mm-dd')
                AND s.OCRDATE < to_date('%s','yyyy-mm-dd')""" %(table,table_01)
    #以下sql是读取store的数据，sender是仓库的类型
    warehouse_sql = """SELECT s.gid,s.code,s.NAME FROM STORE s 
    where bitand(s.property,32)=32"""
    #以下sql是查询client是门店的gid
    shop_sql = """SELECT s.gid,s.code,s.NAME FROM STORE s WHERE
     bitand(s.property,32)<>32 AND substr(s.AREA,2,3)<'8000'"""
    # 使用pandas 的read_sql函数，可以直接将数据存放在dataframe中
    stkout = pd.read_sql(stkout_sql, conn)
    warehouse = pd.read_sql(warehouse_sql, conn)
    shop = pd.read_sql(shop_sql, conn)
    # stkout_detail = pd.read_sql(stkout_detail_sql, conn)
    conn.close
    return stkout,warehouse,shop

def query_stkout_detail(NUM):
    host = "192.168.1.11"   #数据库ip
    port = "1521"        #端口
    sid = "hdapp"         #数据库名称
    dsn = cx_Oracle.makedsn(host, port, sid)

    # scott是数据用户名，tiger是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", dsn)
   #查看出货详细单的数据
    stkout_detail_sql ="""SELECT * FROM STKOUTDTL s WHERE s.CLS='统配出' AND s.NUM in %s """%(NUM,)
    stkout_detail = pd.read_sql(stkout_detail_sql, conn)
    conn.close
    return stkout_detail

def main_function(start,end):
    stkout, warehouse, shop = query(start, end)
    # 定义函数用来对stkout数据进行筛选，从来判断是仓库向线下门店的具体的销售数据
    warehouse = warehouse.rename(index=str, columns={"GID": "SENDER", 'NAME': 'WAREHOUSE_NAME'})
    stkout_mid = pd.merge(stkout, warehouse, on=['SENDER'], how='inner')
    shop = shop.rename(index=str, columns={"GID": "CLIENT", 'NAME': 'SHOP_NAME'})
    sales = pd.merge(stkout_mid, shop, on=['CLIENT'], how='inner')
    # print(sales)
    num_list = sales['NUM'].to_list()
    num_tuple = tuple(num_list)
    # 新建一个空的裂变用于接受返回值

    length = len(num_list)
    step = 1000
    n = math.ceil(length / (step - 1))
    print(n)
    # 这里的N表示一共可以分成多少个1000份的num
    times_cycle = 4
    times = math.ceil(n / (times_cycle - 1))
    print(times)
    data = pd.DataFrame()
    for k in range(0, times):
        print('这是第' + str(k) + '批读取的销售数据')

        data_mid = pd.DataFrame()
        lists = {}
        for i in range(times_cycle * k + 1, times_cycle * (1 + k)):
            lists[(i - 1) * step] = i * step
        for start_user, end_user in lists.items():
            print('进行这段资源id的供应计划的计算' + str(start_user) + ',' + str(end_user))
            num_list_mid = num_tuple[(start_user):(end_user)]
            # print(num_list_mid)
            if num_list_mid:
                results = query_stkout_detail(num_list_mid)
                data_mid = data_mid.append(results, ignore_index=True)
                print(data_mid.size)
            else:
                pass
        data = data.append(data_mid, ignore_index=True)
        result_final = pd.merge(sales,data,on=['NUM'],how='inner')
    return result_final


def multi_read(date_start,date_end):
    pool = multiprocessing.Pool(processes=2)  # 创建4个进程
    results = []
    data = pd.DataFrame()
    start_date = datetime.date(datetime.strptime(date_start, '%Y%m%d'))
    end_date = datetime.date(datetime.strptime(date_end, '%Y%m%d'))
    result_time1 = start_date.strftime('%Y-%m-%d')
    result_time2 = end_date.strftime('%Y-%m-%d')
    d1 = datetime.strptime(result_time1, "%Y-%m-%d")
    d2 = datetime.strptime(result_time2, "%Y-%m-%d")
    x = (d2 - d1).days
    print(x)
    for j in range(x):
        print('正在计算第'+str(j)+'天的详细订单')
        start_date = datetime.date(datetime.strptime(date_start, '%Y%m%d')) + timedelta(j)
        print('start_date:'+ str(start_date))
        end_date = datetime.date(datetime.strptime(date_end, '%Y%m%d')) + timedelta(j+1)+ timedelta(seconds = -1)
        print('end_date:'+ str(end_date))
        results.append(pool.apply_async(main_function,
                                        args=(start_date,end_date)))
    pool.close()  # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
    pool.join()  # 等待进程池中的所有进程执行完毕

    for i in results:
        if i.get().empty == True:
            pass
        else:
            a = i.get()
            data = data.append(a, ignore_index=True)
    return data

if __name__ == '__main__':
    print('start')
    start_time = time.time()
    start = '20190401'
    end = '20190410'

    ############################################################
    result = multi_read(start,end)
    result.to_csv('D:/jimmy-ye/智能供应链/result.csv',encoding="utf_8_sig")
    #############################################################
    end_time = time.time()
    print('测试算法整体耗时：' + str(end_time - start_time))
    print('end')


