import cx_Oracle as cx
import pandas as pd
import pymysql
# use for reading data from different files
class READ_DATA():

    def __init__(self):
        return

    # read data from database from oracle
    def Read_sql(self, sql_name, host, port, dbname, user_name, password, dev_name):
        dsn = cx.makedsn(host, port, dbname)
        connection=cx.connect(user_name,password,dsn)
        connection.cursor()
        dataf = pd.read_sql(sql_name, connection)
        return dataf

    # read data from txt
    def Read_txt(self, txt_name):
        data_fra = pd.read_table(txt_name ,encoding='gb2312',delim_whitespace=True,low_memory=False)
        return  data_fra

    # read data from csv
    def Read_csv(self, csv_name):
        data_fra = pd.read_csv(csv_name)
        return data_fra
