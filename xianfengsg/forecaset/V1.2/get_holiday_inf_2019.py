# -*- coding: utf-8 -*-
# @Time    : 2019/7/17 10:11
# @Author  : Ye Jinyu__jimmy
# @File    : get_holiday_inf.py

import os, io, sys, re, time, datetime, base64
import pandas as pd

__version__ = "$Rev: 123 $"
__all__ = ['LunarDate']

solar_year = 1900
solar_month = 1
solar_day = 31
solar_weekday = 0

lunar_year = 0
lunar_month = 0
lunar_day = 0
lunar_isLeapMonth = False


class LunarDate(object):
    _startDate = datetime.date(1900, 1, 31)

    def __init__(self, year, month, day, isLeapMonth=False):
        global lunar_year
        global lunar_month
        global lunar_day
        global lunar_isLeapMonth

        lunar_year = int(year)
        lunar_month = int(month)
        lunar_day = int(day)
        lunar_isLeapMonth = bool(isLeapMonth)

        self.year = year
        self.month = month
        self.day = day
        self.isLeapMonth = bool(isLeapMonth)

    def __str__(self):
        return 'LunarDate(%d, %d, %d, %d)' % (self.year, self.month, self.day, self.isLeapMonth)

    __repr__ = __str__

    @staticmethod
    def fromSolarDate(year, month, day):
        solarDate = datetime.date(year, month, day)
        offset = (solarDate - LunarDate._startDate).days
        return LunarDate._fromOffset(offset)

    def toSolarDate(self):
        def _calcDays(yearInfo, month, day, isLeapMonth):
            isLeapMonth = int(isLeapMonth)
            res = 0
            ok = False
            for _month, _days, _isLeapMonth in self._enumMonth(yearInfo):
                if (_month, _isLeapMonth) == (month, isLeapMonth):
                    if 1 <= day <= _days:
                        res += day - 1
                        return res
                    else:
                        raise ValueError("day out of range")
                res += _days

            raise ValueError("month out of range")

        offset = 0
        if self.year < 1900 or self.year >= 2050:
            raise ValueError('year out of range [1900, 2050)')
        yearIdx = self.year - 1900
        for i in range(yearIdx):
            offset += yearDays[i]

        offset += _calcDays(yearInfos[yearIdx], self.month, self.day, self.isLeapMonth)
        return self._startDate + datetime.timedelta(days=offset)

    def __sub__(self, other):
        if isinstance(other, LunarDate):
            return self.toSolarDate() - other.toSolarDate()
        elif isinstance(other, datetime.date):
            return self.toSolarDate() - other
        elif isinstance(other, datetime.timedelta):
            res = self.toSolarDate() - other
            return LunarDate.fromSolarDate(res.year, res.month, res.day)
        raise TypeError

    def __rsub__(self, other):
        if isinstance(other, datetime.date):
            return other - self.toSolarDate()

    def __add__(self, other):
        if isinstance(other, datetime.timedelta):
            res = self.toSolarDate() + other
            return LunarDate.fromSolarDate(res.year, res.month, res.day)
        raise TypeError

    def __radd__(self, other):
        return self + other

    def __lt__(self, other):
        return self - other < datetime.timedelta(0)

    def __le__(self, other):
        return self - other <= datetime.timedelta(0)

    @classmethod
    def today(cls):
        res = datetime.date.today()
        return cls.fromSolarDate(res.year, res.month, res.day)

    @staticmethod
    def _enumMonth(yearInfo):
        months = [(i, 0) for i in range(1, 13)]
        leapMonth = yearInfo % 16
        if leapMonth == 0:
            pass
        elif leapMonth <= 12:
            months.insert(leapMonth, (leapMonth, 1))
        else:
            raise ValueError("yearInfo %r mod 16 should in [0, 12]" % yearInfo)

        for month, isLeapMonth in months:
            if isLeapMonth:
                days = (yearInfo >> 16) % 2 + 29
            else:
                days = (yearInfo >> (16 - month)) % 2 + 29
            yield month, days, isLeapMonth

    @classmethod
    def _fromOffset(cls, offset):
        def _calcMonthDay(yearInfo, offset):
            for month, days, isLeapMonth in cls._enumMonth(yearInfo):
                if offset < days:
                    break
                offset -= days
            return (month, offset + 1, isLeapMonth)

        offset = int(offset)

        for idx, yearDay in enumerate(Info.yearDays()):
            if offset < yearDay:
                break
            offset -= yearDay
        year = 1900 + idx

        yearInfo = Info.yearInfos[idx]
        month, day, isLeapMonth = _calcMonthDay(yearInfo, offset)
        return LunarDate(year, month, day, isLeapMonth)


class ChineseWord():
    def weekday_str(tm):
        a = '星期日 星期一 星期二 星期三 星期四 星期五 星期六'.split()
        return a[tm]

    def solarTerm(year, month, day):
        a = '小寒 大寒 立春 雨水 惊蛰 春分\
             清明 谷雨 立夏 小满 芒种 夏至\
             小暑 大暑 立秋 处暑 白露 秋分\
             寒露 霜降 立冬 小雪 大雪 冬至'.split()
        return

    def day_lunar(ld):
        a = '初一 初二 初三 初四 初五 初六 初七 初八 初九 初十\
             十一 十二 十三 十四 十五 十六 十七 十八 十九 廿十\
             廿一 廿二 廿三 廿四 廿五 廿六 廿七 廿八 廿九 三十'.split()
        return a[ld - 1]

    def month_lunar(le, lm):
        a = '正月 二月 三月 四月 五月 六月 七月 八月 九月 十月 十一月 十二月'.split()
        if le:
            return "闰" + a[lm - 1]
        else:
            return a[lm - 1]

    def year_lunar(ly):
        y = ly
        tg = '甲 乙 丙 丁 戊 己 庚 辛 壬 癸'.split()
        dz = '子 丑 寅 卯 辰 巳 午 未 申 酉 戌 亥'.split()
        sx = '鼠 牛 虎 兔 龙 蛇 马 羊 猴 鸡 狗 猪'.split()
        return tg[(y - 4) % 10] + dz[(y - 4) % 12] + '[' + sx[(y - 4) % 12] + ']' + '年'


class Festival():
    # 国历节日 *表示放假日
    def solar_Fstv(solar_month, solar_day):
        sFtv = [
            "0101元旦节",
            "0214情人节",
            "0501劳动节",
            "0504青年节",
            "0910中国教师节",
            "1001国庆节",
            "1002国庆节假日",
            "1003国庆节假日",
            "1224平安夜",
            "1225圣诞节",
        ]
        solar_month_str = str(solar_month) if solar_month > 9 else "0" + str(solar_month)
        solar_day_str = str(solar_day) if solar_day > 9 else "0" + str(solar_day)
        pattern = "(" + solar_month_str + solar_day_str + ")([\w+?\#???\d+\s?·?]*)"
        for solar_fstv_item in sFtv:
            result = re.search(pattern, solar_fstv_item)
            if result is not None:
                return result.group(2)

    def lunar_Fstv(lunar_month, lunar_day):
        # 农历节日 *表示放假日
        # 每年单独来算
        lFtv = [
            "0101春节",
            "0115元宵节",
            "0202春龙节",
             "0314清明节", #每年不一样，此为2018年，事实上为公历节日
            "0505端午节",
            "0707七夕情人节",
            "0715中元节",
            "0815中秋节",
            "0909重阳节",
            "1208腊八节",
            "1223小年",
            "1230除夕"   #每年不一样，此为2018年
        ]
        lunar_month_str = str(lunar_month) if lunar_month > 9 else "0" + str(lunar_month)
        lunar_day_str = str(lunar_day) if lunar_day > 9 else "0" + str(lunar_day)
        pattern = "(" + lunar_month_str + lunar_day_str + ")([\w+?\#?\s?]*)"
        for lunar_fstv_item in lFtv:
            result = re.search(pattern, lunar_fstv_item)
            if result is not None:
                return result.group(2)

    # 国历节日 *表示放假日
    def weekday_Fstv(solar_month, solar_day, solar_weekday):
        # 某月的第几个星期几
        wFtv = [
            "0150世界防治麻风病日",
            "0520国际母亲节",
            "0530全国助残日",
            "0630父亲节",
            "0730被奴役国家周",
            "0932国际和平日",
            "0940国际聋人节 世界儿童日",
            "0950世界海事日",
            "1011国际住房日",
            "1013国际减轻自然灾害日(减灾日)",
            "1128感恩节"]

        # 7，14等应该属于1, 2周，能整除的那天实际属于上一周，做个偏移
        offset = -1 if solar_day % 7 == 0 else 0
        # 计算当前日属于第几周，得出来从0开始计周，再向后偏移1
        weekday_ordinal = solar_day // 7 + offset + 1

        solar_month_str = str(solar_month) if solar_month > 9 else "0" + str(solar_month)
        solar_weekday_str = str(weekday_ordinal) + str(solar_weekday)

        pattern = "(" + solar_month_str + solar_weekday_str + ")([\w+?\#?\s?]*)"
        for weekday_fstv_item in wFtv:
            result = re.search(pattern, weekday_fstv_item)
            if result is not None:
                return result.group(2)

        # 如何计算某些最后一个星期几的情况，..........

    # 24节气
    def solar_Term(solar_month, solar_day):
        # 人为设定2019年的节气信息
        stFtv = [
            "0105小寒",
            "0120大寒",
            "0204立春",
            "0219雨水",
            "0306惊蛰",
            "0321春分",
            "0405清明",
            "0420谷雨",
            "0506立夏",
            "0521小满",
            "0606芒种",
            "0621夏至",
            "0707小暑",
            "0723大暑",
            "0808立秋",
            "0823处暑",
            "0908白露",
            "0923秋分",
            "1008寒露",
            "1024霜降",
            "1108立冬",
            "1122小雪",
            "1207大雪",
            "1222冬至",
        ]
        solar_month_str = str(solar_month) if solar_month > 9 else "0" + str(solar_month)
        solar_day_str = str(solar_day) if solar_day > 9 else "0" + str(solar_day)
        pattern = "(" + solar_month_str + solar_day_str + ")([\w+?\#?]*)"
        for solarTerm_fstv_item in stFtv:
            result = re.search(pattern, solarTerm_fstv_item)
            if result is not None:
                return result.group(2)


class Info():
    yearInfos = [
        #    /* encoding:
        #               b bbbbbbbbbbbb bbbb
        #       bit#    1 111111000000 0000
        #               6 543210987654 3210
        #               . ............ ....
        #       month#    000000000111
        #               M 123456789012   L
        #
        #    b_j = 1 for long month, b_j = 0 for short month
        #    L is the leap month of the year if 1<=L<=12; NO leap month if L = 0.
        #    The leap month (if exists) is long one iff M = 1.
        #    */
        0x04bd8,  # /* 1900 */
        0x04ae0, 0x0a570, 0x054d5, 0x0d260, 0x0d950,  # /* 1905 */
        0x16554, 0x056a0, 0x09ad0, 0x055d2, 0x04ae0,  # /* 1910 */
        0x0a5b6, 0x0a4d0, 0x0d250, 0x1d255, 0x0b540,  # /* 1915 */
        0x0d6a0, 0x0ada2, 0x095b0, 0x14977, 0x04970,  # /* 1920 */
        0x0a4b0, 0x0b4b5, 0x06a50, 0x06d40, 0x1ab54,  # /* 1925 */
        0x02b60, 0x09570, 0x052f2, 0x04970, 0x06566,  # /* 1930 */
        0x0d4a0, 0x0ea50, 0x06e95, 0x05ad0, 0x02b60,  # /* 1935 */
        0x186e3, 0x092e0, 0x1c8d7, 0x0c950, 0x0d4a0,  # /* 1940 */
        0x1d8a6, 0x0b550, 0x056a0, 0x1a5b4, 0x025d0,  # /* 1945 */
        0x092d0, 0x0d2b2, 0x0a950, 0x0b557, 0x06ca0,  # /* 1950 */
        0x0b550, 0x15355, 0x04da0, 0x0a5d0, 0x14573,  # /* 1955 */
        0x052d0, 0x0a9a8, 0x0e950, 0x06aa0, 0x0aea6,  # /* 1960 */
        0x0ab50, 0x04b60, 0x0aae4, 0x0a570, 0x05260,  # /* 1965 */
        0x0f263, 0x0d950, 0x05b57, 0x056a0, 0x096d0,  # /* 1970 */
        0x04dd5, 0x04ad0, 0x0a4d0, 0x0d4d4, 0x0d250,  # /* 1975 */
        0x0d558, 0x0b540, 0x0b5a0, 0x195a6, 0x095b0,  # /* 1980 */
        0x049b0, 0x0a974, 0x0a4b0, 0x0b27a, 0x06a50,  # /* 1985 */
        0x06d40, 0x0af46, 0x0ab60, 0x09570, 0x04af5,  # /* 1990 */
        0x04970, 0x064b0, 0x074a3, 0x0ea50, 0x06b58,  # /* 1995 */
        0x05ac0, 0x0ab60, 0x096d5, 0x092e0, 0x0c960,  # /* 2000 */
        0x0d954, 0x0d4a0, 0x0da50, 0x07552, 0x056a0,  # /* 2005 */
        0x0abb7, 0x025d0, 0x092d0, 0x0cab5, 0x0a950,  # /* 2010 */
        0x0b4a0, 0x0baa4, 0x0ad50, 0x055d9, 0x04ba0,  # /* 2015 */
        0x0a5b0, 0x15176, 0x052b0, 0x0a930, 0x07954,  # /* 2020 */
        0x06aa0, 0x0ad50, 0x05b52, 0x04b60, 0x0a6e6,  # /* 2025 */
        0x0a4e0, 0x0d260, 0x0ea65, 0x0d530, 0x05aa0,  # /* 2030 */
        0x076a3, 0x096d0, 0x04afb, 0x04ad0, 0x0a4d0,  # /* 2035 */
        0x1d0b6, 0x0d250, 0x0d520, 0x0dd45, 0x0b5a0,  # /* 2040 */
        0x056d0, 0x055b2, 0x049b0, 0x0a577, 0x0a4b0,  # /* 2045 */
        0x0aa50, 0x1b255, 0x06d20, 0x0ada0  # /* 2049 */
    ]

    def yearInfo2yearDay(yearInfo):
        yearInfo = int(yearInfo)

        res = 29 * 12

        leap = False
        if yearInfo % 16 != 0:
            leap = True
            res += 29

        yearInfo //= 16

        for i in range(12 + leap):
            if yearInfo % 2 == 1:
                res += 1
            yearInfo //= 2
        return res

    def yearDays():
        yearDays = [Info.yearInfo2yearDay(x) for x in Info.yearInfos]
        return yearDays

    def day2LunarDate(offset):
        offset = int(offset)
        res = LunarDate()

        for idx, yearDay in enumerate(yearDays()):
            if offset < yearDay:
                break
            offset -= yearDay
        res.year = 1900 + idx


class SolarDate():

    def __init__(self):
        global solar_year
        global solar_month
        global solar_day
        global solar_weekday

        solar_year = int(time.strftime("%Y", time.localtime()))
        solar_month = int(time.strftime("%m", time.localtime()))
        solar_day = int(time.strftime("%d", time.localtime()))
        solar_weekday = int(time.strftime("%w", time.localtime()))

        self.year = solar_year
        self.month = solar_month
        self.day = solar_day
        self.weekday = solar_weekday

    def __str__(self):
        return 'LunarDate(%d, %d, %d, %d)' % (self.year, self.month, self.day, self.isLeapMonth)


def getCalendar_today():
    solar = SolarDate()
    LunarDate.fromSolarDate(solar_year, solar_month, solar_day)

    festival = ""

    if Festival.solar_Term(solar_month, solar_day):
        festival = festival + " 今日节气：" + Festival.solar_Term(solar_month, solar_day)
    if Festival.solar_Fstv(solar_month, solar_day):
        festival = festival + " 公历节日：" + Festival.solar_Fstv(solar_month, solar_day)
    if Festival.weekday_Fstv(solar_month, solar_day, solar_weekday):
        if festival.find("公历节日") == -1:
            festival = festival + " 公历节日：" + Festival.weekday_Fstv(solar_month, solar_day, solar_weekday)
        else:
            festival = festival + " " + Festival.weekday_Fstv(solar_month, solar_day, solar_weekday)
    if Festival.lunar_Fstv(lunar_month, lunar_day):
        festival = festival + " 农历节日：" + Festival.lunar_Fstv(lunar_month, lunar_day)

    twitter = \
        "今天是" + str(solar_year) + "年" + str(solar_month) + "月" + str(solar_day) + "日" + " " \
        + ChineseWord.weekday_str(solar_weekday) + " 农历" + ChineseWord.year_lunar(lunar_year) \
        + ChineseWord.month_lunar(lunar_isLeapMonth, lunar_month) \
        + ChineseWord.day_lunar(lunar_day) + festival
    print(twitter)
    return twitter


def getCalendar_all_day():
    # solar = SolarDate()
    global solar_year
    global solar_month
    global solar_day
    global solar_weekday

    #    solar_year = 2012

    solar_year = 2019

    solar_month = 1
    weekday_offset = 0  # 1月1号星期几?
    index_day = 2
    chinese_holiday = pd.DataFrame()
    for solar_month in range(1, 13):
        if solar_month in [1, 3, 5, 7, 8, 10, 12]:
            solar_day_max = 31
        elif solar_month in [4, 6, 9, 11]:
            solar_day_max = 30
        elif solar_month == 2:
            if ((solar_year % 4 == 0) and (solar_year % 100 != 0)) or (solar_year % 400 == 0):
                solar_day_max = 29
            else:
                solar_day_max = 28
        else:
            None

        for solar_day in range(1, solar_day_max + 1):
            index_day += 1
            solar_weekday = (index_day) % 7 + - 1
            solar_weekday = 0 if solar_weekday == 7 else solar_weekday
            solar_weekday = 6 if solar_weekday == -1 else solar_weekday

            LunarDate.fromSolarDate(solar_year, solar_month, solar_day)
            festival = ""

            if Festival.solar_Term(solar_month, solar_day):
                festival = festival + " 节气：" + Festival.solar_Term(solar_month, solar_day)
            if Festival.solar_Fstv(solar_month, solar_day):
                festival = festival + " 节日：" + Festival.solar_Fstv(solar_month, solar_day)
            if Festival.weekday_Fstv(solar_month, solar_day, solar_weekday):
                if festival.find("节日") == -1:
                    festival = festival + " 节日：" + Festival.weekday_Fstv(solar_month, solar_day, solar_weekday)
                else:
                    festival = festival + " " + Festival.weekday_Fstv(solar_month, solar_day, solar_weekday)
            if Festival.lunar_Fstv(lunar_month, lunar_day):
                if festival.find("节日") == -1:
                    festival = festival + " 节日：" + Festival.lunar_Fstv(lunar_month, lunar_day)
                else:
                    festival = festival + " " + Festival.lunar_Fstv(lunar_month, lunar_day)

            index_yy = str(solar_year)
            if int(solar_month) < 10:
                index_mm = "0" + str(solar_month)
            else:
                index_mm = str(solar_month)
            if int(solar_day) < 10:
                index_dd = "0" + str(solar_day)
            else:
                index_dd = str(solar_day)

            index_yyddmm = index_yy + index_mm + index_dd

            twitter = ("message(" + str(index_yyddmm) + ') = "') + \
                      str(solar_year) + "年" + str(solar_month) + "月" + str(solar_day) + "日" + " " \
                      + ChineseWord.weekday_str(solar_weekday) + " 农历" + ChineseWord.year_lunar(lunar_year) \
                      + ChineseWord.month_lunar(lunar_isLeapMonth, lunar_month) \
                      + ChineseWord.day_lunar(lunar_day) + festival + '"'
            chinese_holiday_mid =pd.DataFrame()
            chinese_holiday_mid['Account_date'] = pd.Series(str(index_yyddmm))
            chinese_holiday_mid['Weekday'] = pd.Series(ChineseWord.weekday_str(solar_weekday))
            chinese_holiday_mid['Chinese_festival'] = pd.Series(Festival.solar_Fstv(solar_month, solar_day))
            chinese_holiday_mid['Solar_festival'] = pd.Series(Festival.weekday_Fstv(solar_month, solar_day, solar_weekday))
            chinese_holiday_mid['Term_festival'] = pd.Series(
                Festival.solar_Term(solar_month, solar_day))
            chinese_holiday_mid['Lunar_festival'] = pd.Series(
                Festival.lunar_Fstv(lunar_month, lunar_day))
            # chinese_holiday = chinese_holiday.append(chinese_holiday)
            # print(chinese_holiday_mid)
            chinese_holiday = chinese_holiday.append(chinese_holiday_mid)
            # print(twitter)
    return twitter,chinese_holiday


def main():
    "main function"
    print(base64.b64decode(b'Q29weXJpZ2h0IChjKSAyMDEyIERvdWN1YmUgSW5jLiBBbGwgcmlnaHRzIHJlc2VydmVkLg==').decode())
    # getCalendar_all_day()
    # getCalendar_today()
    reslut = getCalendar_all_day()
    # print(reslut[1])
    return reslut




if __name__ == '__main__':
    main()