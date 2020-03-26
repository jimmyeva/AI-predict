# -*- coding: utf-8 -*-
# @Time    : 2020/3/13 9:50
# @Author  : Ye Jinyu__jimmy
# @File    : forecast_model.py



import pandas as pd
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
from sklearn import preprocessing,metrics
import numpy as np
import time
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,train_test_split
import matplotlib
matplotlib.use('TkAgg')
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt

import features_engineering
import cx_Oracle
import datetime
import pymysql
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import math

'''当前是针对给点的数据集进行模型的训练和输出'''

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

#————————————————————热力图————————————
def heatmap(train_df):
    plt.subplots(figsize=(24,20))
    sns.heatmap(train_df.corr(),cmap='RdYlGn',annot=True,vmin=-0.1,vmax=0.1,center=0)



#————————————————————构建训练集与测试集————————————————————————
def df_division(data_frame,end_date):

    #--------用于训练使用----
    train_df            =  data_frame[data_frame['Account_date']  <= end_date]
    train_df = train_df[~train_df['sales_qty'].isin([0])]

    train, test         =  train_test_split(train_df, test_size=0.2, random_state=10)
    #拆分特征与标签，并将标签取对数处理
    ytrain              =  np.log1p(train['sales_qty'])
    ytest               =  np.log1p(test['sales_qty'])


    Xtrain              =  train.drop(['Account_date','sales_qty'],axis=1).values
    Xtest               =  test.drop(['Account_date','sales_qty'],axis=1).values
    #-----------------把需要预测的特征也准备好
    predict_df          = data_frame[data_frame['Account_date'] > end_date]
    new_start           = predict_df['Account_date'].min()
    end                 = predict_df['Account_date'].max()
    predict_df.sort_values('Account_date')
    Xpredict            = predict_df.drop([ 'Account_date','sales_qty'], axis=1).values

    return Xtrain,ytrain,Xtest,ytest,Xpredict, new_start, end

#————————————————————————获取需要预测的特征——————————————————————
# def predict_function(data_frame,end_date):
#     # print(data_frame)
#
#     return Xpredict,new_start,end

#——————————————定义评价函数，可以传入后面模型中替代模型本身的损失函数————————————

def mape(yhat, y):
    """    mape -- MAPE 评价指标    """
    n       = len(y)
    mape    = sum(np.abs((y - yhat) / y)) / n * 100
    return mape


def mae_value(yhat, y):
    '''   返回:
    mae -- MAE 评价指标'''
    n   = len(y)
    print('优化参数的总长度是：%d'%n)
    mae = sum(np.abs(y - yhat)) / n
    return mae


#定义评价函数，可以传入后面模型中替代模型本身的损失函数
def rmspe(y,yhat):
    return np.sqrt(np.mean((y - yhat) ** 2))


def rmspe_xg(yhat,y):
    y_       =   np.expm1(y.get_label())
    yhat_    =   np.expm1(yhat)
    return 'rmspe',rmspe(y_,yhat_)



# ————————————————————————初始模型构建参数设定————————————————————————————————
def training_function(Xtrain,ytrain,Xtest,ytest):
    params  ={'objective':'reg:linear',
           'booster':'gbtree',
           'eta':0.15,
           'max_depth':10,
           'subsample':0.8,
           'colsample_bytree':0.7,
           'silent':1,
           'seed':10}
    num_boost_round = 6000
    dtrain      =  xgb.DMatrix(Xtrain,ytrain)
    dvalid      =  xgb.DMatrix(Xtest,ytest)
    watchlist   =  [(dtrain,'train'),(dvalid,'eval')]

    # #模型训练
    print('Train a XGBoost model')
    start   = time.time()
    gbm     = xgb.train(params,dtrain,num_boost_round,evals=watchlist,
                 early_stopping_rounds=80,feval=rmspe_xg,verbose_eval=True) #

    end     = time.time()
    print('Train time is {:.2f} s.'.format(end-start))
    return gbm

#————————————————————模型优化————————————————————————————————
def modeling_optimize(ytest,yhat):
    print('weight correction')
    W = [(0.980 + (i / 1000)) for i in range(200)]
    S = []
    for w in W:
        error   = mae_value(np.expm1(ytest), np.expm1(yhat * w))
        print('mae_value for {:.3f}:{:.6f}'.format(w, error))
        S.append(error)
    Score       = pd.Series(S, index=W)
    Score.plot()
    BS          = Score[Score.values == Score.values.min()]
    a           = np.array(BS.index.values)
    W_ho        = a.repeat(len(ytest))
    print('Best weight for Score:{}'.format(BS))
    ##计算校正后整体数据的MAE得分
    yhat_new    = yhat * W_ho
    error       = mae_value(np.expm1(ytest), np.expm1(yhat_new))
    print('mae_value for weight corretion {:.6f}'.format(error))
    return yhat_new,BS


#——————————————————————对预测值进行修正————————————————————————
def revised_predict(data,predict_df):
    max_qty     = data['sales_qty'].max()
    min_qty     = data['sales_qty'].min()
    predict_df["forecast"].iloc[np.where(predict_df["forecast"] < min_qty)] = min_qty
    predict_df["forecast"].iloc[np.where(predict_df["forecast"] > max_qty)] = max_qty
    return predict_df


#————————————————定义函数用于训练模型并输出修正系数——————
def model_revised(Xtrain, ytrain, Xtest, ytest,Xpredict):
    gbm         = training_function(Xtrain, ytrain, Xtest, ytest)
    yhat        = gbm.predict(xgb.DMatrix(Xtest))
    yhat_new,BS = modeling_optimize(ytest, yhat)

    compare     = pd.DataFrame({'real:': np.expm1(ytest), 'forecast': np.expm1(yhat_new)})
    print('compare', compare)

    a           = np.array(BS.index.values)
    a.repeat(len(ytest))
    W_revise    = a.repeat(len(Xpredict))
    return gbm,W_revise




def exponential_smoothing(alpha, s):
    '''
    一次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回一次指数平滑模型参数， list
    '''

    s_temp = [0 for i in range(len(s))]
    s_temp[0] = ( s[0] + s[1] + s[2] ) / 3
    for i in range(1, len(s)):
        s_temp[i] = alpha * s[i] + (1 - alpha) * s_temp[i-1]
    return s_temp

def compute_single(alpha, s):
    '''
    一次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回一次指数平滑模型参数， list
    '''
    return exponential_smoothing(alpha, s)

def compute_double(alpha, s):
    '''
    二次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回二次指数平滑模型参数a, b， list
    '''
    s_single = compute_single(alpha, s)
    s_double = compute_single(alpha, s_single)

    a_double = [0 for i in range(len(s))]
    b_double = [0 for i in range(len(s))]

    for i in range(len(s)):
        a_double[i] = 2 * s_single[i] - s_double[i]                    #计算二次指数平滑的a
        b_double[i] = (alpha / (1 - alpha)) * (s_single[i] - s_double[i])  #计算二次指数平滑的b
    #----------------------------------------构建未来7日的预测------------------------------------------------------------
    pre_list = list()
    for i in range(0,7):
        pre_new = a_double[-1] + b_double[-1] * i
        pre_list.append(pre_new)

    return pre_list


def model_double(data_sr):
    fit2    = Holt(data_sr, exponential=True).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
    result  = fit2.forecast(7)
    l2, = plt.plot(list(fit2.fittedvalues) + list(fit2.forecast(7)), marker='.')
    plt.show()
    return result

#——————————————————————完整的单个SKU的预测的整合————————————————
def forecast_merge(train_df,end_date):

    train_df['sales_qty'].astype('int')
    #------------有些SKU前面各种处理后还是只有少量的销售--------------
    if len(train_df[~train_df['sales_qty'].isin([0])]) <=10:
        new_start  = (datetime.datetime.strptime(end_date, '%Y%m%d')  + datetime.timedelta(7)).strftime('%Y%m%d')
        print('train_df',train_df)
        #=============================================================================================>二阶指数预测
        ts         = train_df[['sales_qty']]
        pre_list   = compute_double(0.7, ts)
        predict_df = pd.DataFrame({'Account_date': pd.date_range(end_date, new_start, freq='D'),
                                   'forecast': np.array(pre_list)})
    else:
        Xtrain,ytrain,Xtest,ytest,Xpredict, new_start, end = df_division(train_df,end_date)
        gbm, W_revise                   = model_revised(Xtrain, ytrain, Xtest, ytest,Xpredict)

        predict_list = np.round(np.expm1(gbm.predict(xgb.DMatrix(Xpredict)) * W_revise))
        predict_df = pd.DataFrame({'Account_date': pd.date_range(new_start, end, freq='D'),
                                   'forecast':np.array(predict_list)})
        predict_df = revised_predict(train_df,predict_df)
    return predict_df



#——————————————————特征重要度画图————————————
def feature_importance(train_df,gbm):
        columns       = train_df.drop(['Account_date','sales_qty'],axis=1).columns
        feature_score = gbm.get_fscore()
        feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)

        fs = []
        for (key, value) in feature_score:
            fs.append("{0},{1}".format(key[1:], value))

        feature_list = list()
        fscore_list  = list()
        for i in range(len(fs)):
            temp            =   fs[i]
            temp_split      =   temp.split(',', 1)
            feature_list.append(int(temp_split[0]))
            fscore_list.append(int(temp_split[1]))

        importance_columns  =   list()
        for x in range(len(feature_list)):
            columns_index = feature_list[x]
            importance_columns.append(columns[columns_index])
        df = pd.DataFrame({'feature_list': feature_list, 'fscore_list': fscore_list,'feature_name':importance_columns})
        df['fscore_list'] = df['fscore_list'] / df['fscore_list'].sum()

        #
        feature_name      = df['feature_name'][:30]
        fscore_list       = df['fscore_list'][:30]

        fig               = plt.figure(figsize=(20, 10), facecolor='white')
        ax1               = fig.add_subplot(111)
        # 左轴
        ax1.bar(feature_name, fscore_list, width=0.5, align='center', label='real_qty', color="black")

        plt.xticks(feature_name, color='blue', rotation=90)
        plt.legend(loc='upper left', fontsize=10)
        # plt.text('2019-10-01', sum, text, fontdict={'size': 20, 'color': 'y'}, verticalalignment='top',
        #          horizontalalignment='left')
        ax1.set_xlabel('relative importance')
        # ax1.set_ylabel('relative importance')
        plt.title('XGBoost Feature Importance')
        # plt.xlabel('relative importance')
        plt.savefig('./fscore_list.jpg', dpi=600, bbox_inches='tight')
        plt.close()



if __name__ == '__main__':
    # train_df = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/compare_old_new/merge.csv',encoding='utf_8_sig')
    # end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    #
    # df       = train_df[['Account_date','sales_qty']]

    ts = list([31.0,152.0,128.0,34.0,67.0,47.0,5.0,])

    print(ts)
    pre_list      = compute_double(0.7,ts)
    print(pre_list)





























def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds) #, show_progress=False
        alg.set_params(n_estimators=cvresult.shape[0])


    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print('\n','Model Report')
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
