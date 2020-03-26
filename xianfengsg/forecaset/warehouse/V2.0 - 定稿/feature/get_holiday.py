# -*- coding: utf-8 -*-
# @Time    : 2020/3/2 10:17
# @Author  : Ye Jinyu__jimmy
# @File    : get_holiday

import json
import urllib.request
import socket
import time
import pandas as pd


timeout = 10
socket.setdefaulttimeout(timeout)

'''获取中国的法定节假日的信息,用来看是工作日，还是周末，还是假日'''


def time_series(days):
    # server_url = "http://api.goseek.cn/Tools/holiday?date="
    server_url = "http://www.easybots.cn/api/holiday.php?d="
    week_day_dict = {
        0: '星期一',
        1: '星期二',
        2: '星期三',
        3: '星期四',
        4: '星期五',
        5: '星期六',
        6: '星期天',
    }
    week_day = week_day_dict[days.weekday()]
    format = "%Y%m%d"
    date_str = days.strftime(format)
    headers = {'User-Agent': 'User-Agent:Mozilla/5.0'}

    vop_url_request = urllib.request.Request(server_url + date_str,headers=headers)

    vop_response = urllib.request.urlopen(vop_url_request)
    vop_data = json.loads(vop_response.read())

    if vop_data[date_str] == '0':
        holiday =  'weekday'
    elif vop_data[date_str] == '1' or '2':
        holiday =  'holiday'
    else:
        holiday = 'Error'
    return week_day,holiday



def holiday_1st(start_date,end_date):
    data_date = pd.DataFrame()
    for days in pd.date_range(start_date, end_date):
        # time.sleep(1)
        try:
            week_day,holiday = time_series(days)

            single_day = pd.DataFrame({'Account_date':[days],'week_day':[week_day],'holiday':[holiday]})
            data_date = data_date.append(single_day)
        except Exception as e:
            print(e)  # 这里开始报错，都是说超时，或者代理错误
            continue
    return data_date


#-----------------------差缺补漏————————————————————————
def fill_date(data_date,start_date = '2018-01-01',end_date='2021-01-01'):

    for days in pd.date_range(start_date, end_date):
        format = "%Y-%m-%d"
        date_str = days.strftime(format)

        if date_str in data_date['Account_date'].values:
            pass
        else:
            week_day,holiday = time_series(days)
            # single_day = pd.DataFrame({'Account_date':[days],'week_day':[week_day],'holiday':[holiday]})
            data_date = data_date.append({'Account_date':days,'week_day':week_day,'holiday':holiday},
                                         ignore_index=True)
    # data_date.to_csv('./data_date.csv', index=False, encoding='utf_8_sig')
    return data_date

#————————————————————————————————完整获取时间的节假日和工作日的函数————————————————
def get_holiday_function(start_date = '2018-01-01',end_date='2021-01-01'):
    data_date = holiday_1st(start_date,end_date)
    final_date = fill_date(data_date,start_date, end_date)
    return final_date

#——————————————————————滑窗特征————————————————————————
if __name__ == '__main__':
    start_date = '2018-01-01'
    end_date = '2021-01-01'
    data_date = get_holiday_function(start_date, end_date)

