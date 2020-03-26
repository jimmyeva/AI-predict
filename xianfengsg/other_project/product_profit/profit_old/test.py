# -*- coding: utf-8 -*-
# @Time    : 2019/8/5 15:28
# @Author  : Ye Jinyu__jimmy
# @File    : test_xgboost.py

#coding = utf-8
import sys
import math




max_value =26
def CompletePack_min(W, V, MW):  # 不完全背包
    print('开始计算最小值')
    # 存储最大价值的一维数组

    valuelist = [max_value] * (MW + 1)
    valuelist[0] = 0

    # 存储物品编号的字典
    codedict = {i: [] for i in range(0, MW + 1)}
    # 开始计算
    print('valuelist', valuelist)
    for i in range(len(W)):  # 从第一个物品
        print('正在计算',int(i),'个商品')
        copyvalue = valuelist.copy()
        copydict = codedict.copy()
        start_num = math.ceil(MW/W[i])
        print('start_num',start_num)
        for j in range(0,MW + 1):  # 从重量0
            print('j',j)
            if j >= W[i]:  # 如果重量大于物品重量
                cc = copyvalue[j]
                print('copyvalue',copyvalue[j])
                print('copyvalue_i-1',copyvalue[j - W[i]] + V[i])
                print('cc',cc)
                copyvalue[j] = min(copyvalue[j - W[i]] + V[i], copyvalue[j])  # 选中第i个物品和不选中，取最小
                print('copyvalue_after',copyvalue[j])
                # 输出被选中的物品编号
                # # copydict[j] = [i]
                # for hh in copydict[j - i]:
                #     copydict[j].append(hh)
                if copyvalue[j] <= cc:       #逐步迭代操作
                    copydict[j] = [i]
                    print(copydict[j])
                    for h in copydict[j - W[i]]:       #将最小值记录在copydict内
                        copydict[j].append(h)
                    print(copydict[j])
        codedict = copydict.copy()  # 更新
        valuelist = copyvalue.copy()  # 更新
    print('codedict',codedict)
    result = ''
    print(list(set(copydict[MW])))
    total_cost = 0
    for hcode in sorted(list(set(copydict[MW]))):
        result += '物品：%d :%d个,' % (hcode + 1, copydict[MW].count(hcode))
        weight_s = weight[hcode] * copydict[MW].count(hcode)
        total_cost += weight_s
    return '最小价值：', valuelist[-1], total_cost, result
weight =    [2.1,3.2, 3.3,4.4,5.2,3.2]
value =     [1,1,2,2,2,1]
maxweight = 20


#  也输出选择物品的编号以及个数
def CompletePack(W, V, MW):#每个商品可以选择多次
    #存储最大价值的一维数组
    valuelist = [0] * (MW + 1)
    #存储物品编号的字典
    codedict = {i: [] for i in range(0, MW + 1)}
    #开始计算
    for ii in range(len(W)):#从第一个物品
        copyvalue = valuelist.copy()
        copydict = codedict.copy()
        print('正在计算',int(ii),'个商品')
        for jj in range(MW + 1):#从重量0
            print('正在计算的重量是', int(jj))
            if jj >= W[ii]:#如果重量大于物品重量
                cc = copyvalue[jj]
                print('上一次最大的价值是：',cc)
                x = round(jj - W[ii])     #索引必须要整数，但是在实际中的金额可能会存在小数的情况
                print('x',x)
                print('在放入第',ii,'个物品前最大的价值',copyvalue[x]+V[ii])
                print(copyvalue[jj])
                copyvalue[jj] = max(copyvalue[x] + V[ii], copyvalue[jj])  #选中第ii个物品和不选中，取大的
                #输出被选中的物品编号
                if copyvalue[jj] > cc:
                    copydict[jj] = [ii]
                    y = round(jj - W[ii])
                    for hh in copydict[y]:
                        copydict[jj].append(hh)
        codedict = copydict.copy()#更新
        valuelist = copyvalue.copy()#更新
        print(ii,'ii',copyvalue)
    result = ''
    total_cost = 0
    for hcode in sorted(list(set(copydict[MW]))):
        result += '%d,%d;' % (hcode, copydict[MW].count(hcode))
        weight_s = W[hcode] * copydict[MW].count(hcode)
        total_cost += weight_s
    return '最大价值：', valuelist[-1],total_cost,result

result=CompletePack(weight, value, maxweight)
print(result)