## for parameters settings
## paramters for data reading
import Data_Read
import pandas as pd

from datetime import datetime,date
from datetime import timedelta
import pymysql
import sys
## parameters for predict data
predict_txt = "F:/data_sales/result/chuanjie_prediction_9_18.txt"

# ## parameters for the time of today
# Today_date = date.today()-timedelta(4)
# # d = sys.argv[1]
# # splite_d = d[0:8]
# # Today_date = datetime.date(datetime.strptime(splite_d,'%Y%m%d'))-timedelta(1)
# Parameter_cnt = Today_date.strftime('%Y-%m-%d')
# # Parameter_cnt = datetime.date(datetime.strptime(Today_date,'%Y%m%d'))
# predict_time = "date'%s'"%Parameter_cnt



## parameters of Ess_data from database for mysql
global parame_mysql

parame_mysql ={'host':"192.168.6.122",
                 'port': 3306,
                 'user':"root",
                 'password':"Rpt@123456",
                 'database':"data_cjgm"}


class Original_Data():

    def __init__(self):
        return

    ## calculate the range of time ,use for select data,function use for create time range
    def rq_range(delf, start_time, end_time):
        rq_week = pd.date_range(start_time.strftime('%Y-%m-%d'), end_time.strftime('%Y-%m-%d'))
        return rq_week

    # read original data for product activities(like in out and save)
    def Mysql_Data(self, sql_name):
        conn = pymysql.connect(**parame_mysql)
        conn.cursor()
        datamysql = pd.read_sql(sql_name, conn)
        return datamysql


    # read original data for predict data
    def Pre_Data(self):
        DataRead0 = Data_Read.READ_DATA()
        predict_data0 = DataRead0.Read_txt(predict_txt)
        return predict_data0

# ## parameters for weeklist that caculate the error between predict num and real num
# end_time = Today_date
# start_time = end_time - timedelta(15*28-1)
#
# ## parameters for predict time
# end_time_pre = Today_date
# start_time_pre = end_time_pre - timedelta(30)
#
# ## parameters for calculating unsold goods(6 months)
# ## mean sales for sold goods
#
# srq_end = Today_date
# srq_start = srq_end - timedelta(91)
#
# ## date parameter for sale_sql
# sale_sql_date = str(Today_date-timedelta(400))
# start_sale_time = "date'%s'"%sale_sql_date
#
# ## date parameter for usable_inventory_sql
# usable_inventory_time = "date'%s'"%Today_date
# ## date parameter for order_sql
# order_date = str(Today_date - timedelta(Today_date.day))
# order_time = "date'%s'"%order_date
# ## date parameter for usable_promotion_quota_sql
# quota_time = "date'%s'"%order_date
#
# #t0 = len(rq_range(start_time_pre, end_time_pre))
# #t = 12  # for lead time
# #t2 = 16 # for all lead time
# t3 = 35 # for predictive cycle roll time
# sigma = 1.2 #for The uncertainty of prediction
# ## parameters for  sales data

#manufacturer_num,custom_business_num,custom_stock_num,custom_terminal_num,t2 = 1,2,3,4,5,
#manufacturer_num,custom_business_num,custom_stock_num,custom_terminal_num,account_date = 1,2,3,4,5












