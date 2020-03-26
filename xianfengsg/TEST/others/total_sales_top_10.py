# -*- coding: utf-8 -*-
# @Time    : 2019/6/18 16:29
# @Author  : Ye Jinyu__jimmy
# @File    : total_sales_top_10.py

#这个函数是用来查看那些仓库向门店的出库信息，并且是销售总额在前10名的数据，这个排名可以自己定义
#需要读取的数据是销售明细的数据

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

import time

#设置一个读取数据的函数
def query_stkout_detail(NUM):
    host = "192.168.1.11"   #数据库ip
    port = "1521"        #端口
    sid = "hdapp"         #数据库名称
    dsn = cx_Oracle.makedsn(host, port, sid)

    # scott是数据用户名，tiger是登录密码（默认用户名和密码）
    conn = cx_Oracle.connect("hd40", "xfsg0515pos", dsn)
   #查看出货详细单的数据
    stkout_detail_sql ="""SELECT s.GDGID,b.NUM,s.RTOTAL,b.OCRDATE,s.CRTOTAL,s.MUNIT,s.QTY,s.QTYSTR,s.TOTAL,s.PRICE,s.QPC FROM STKOUTDTL s INNER JOIN(
                        select *
                        from STKOUT s
                        INNER JOIN STORE s1  ON s.sender = s1.gid
                        INNER JOIN STORE s2 ON s.CLIENT = s2.gid 
                        WHERE bitand(s1.property,32)=32 
                        AND bitand(s2.property,32)<>32 
                        AND substr(s2.AREA,2,3)<'8000' 
                        AND s.CLS='统配出')b ON s.NUM = b.NUM AND s.CLS='统配出' 
                        and s.GDGID in %s """%(NUM,)
    stkout_detail = pd.read_sql(stkout_detail_sql, conn)
    # stkout_detail = pd.read_sql(stkout_detail_sql, conn)
    conn.close
    return stkout_detail

def main_function():
    gd_gid = pd.read_excel('C:/Users/dell/Desktop/sales_by_totali_1000.xlsx',nrows=100)
    good_id = gd_gid['GDGID'].to_list()
    good_id_tuple = tuple(good_id)
    stkout_detail = query_stkout_detail(good_id_tuple)
    stkout_detail.to_csv('D:\jimmy-ye\AI_supply_chain\data.csv',encoding="utf_8_sig")


if __name__ == '__main__':
    print('start')
    start_time = time.time()
    start= time.strftime("%Y-%m-%d %H:%M:%S")
    print(start)
    ############################################################
    main_function()
    #############################################################
    end_time = time.time()
    end= time.strftime("%Y-%m-%d %H:%M:%S")
    print(end)
    print('测试算法整体耗时：' + str(end_time - start_time))
    print('end')





