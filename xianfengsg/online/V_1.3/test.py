# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 20:02
# @Author  : Ye Jinyu__jimmy
# @File    : test.py

import redis
import time
import sys

redis_key = sys.argv[1]
wh_code = sys.argv[2]
today_date_str = sys.argv[3]



client = redis.Redis(host="192.168.1.180",port=6379, db=7,decode_responses=True,socket_connect_timeout=6000)
client.hset(redis_key, "total", 50)
for i in range(0,50):
    client.hincrby(redis_key, 'total', 1)
    print(client.hgetall(redis_key))
