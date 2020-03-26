# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 20:02
# @Author  : Ye Jinyu__jimmy
# @File    : test.py

import redis
import time
import sys
import pandas as pd

# redis_key = sys.argv[1]
# wh_code = sys.argv[2]
# today_date_str = sys.argv[3]
#
#
#
# client = redis.Redis(host="192.168.1.180",port=6379, db=7,decode_responses=True,socket_connect_timeout=6000)
# client.hset(redis_key, "total", 50)
# for i in range(0,50):
#     client.hincrby(redis_key, 'total', 1)
#     print(client.hgetall(redis_key))


#————————————————————读取数据库中的特殊：这里选择出对应商品在当前节点的逻辑规则,当前只加入安全库存的设置————————————————————
def special_rule_data():
    data = pd.read_excel('./data_rule.xlsx',converters={u'sku_code': str})
    data = data[['sku_code','expert_dim','expert_rule']]
    return data

# data = special_rule_data()
# code = '01319'


#————————————————传入对应的五位码，返回对应的安全库存的设置量——————————————————
def get_special_rule(code,data):
    if code in data['sku_code'].values == True:
        print('存在特殊规则')
        expert_rule = data[data['sku_code'] == code]['expert_rule'].iloc[0]
    else:
        print('不存在特殊规则')
        expert_rule = 2
    return expert_rule



a = 13
print(len(str(a)))
x = int(-len(str(a)) + 1)
print(x)
print(round(a,x))


