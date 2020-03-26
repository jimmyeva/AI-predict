# -*- coding: utf-8 -*-
# @Time    : 2020/2/28 14:02
# @Author  : Ye Jinyu__jimmy
# @File    : data_sci

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import SelectPercentile,f_classif

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
import warnings
warnings.filterwarnings('ignore')


#设置评价的指标
evaluation = pd.DataFrame({'Model': [],
                           'Details':[],
                           'Root Mean Squared Error (RMSE)':[],
                           'R-squared (training)':[],
                           'Adjusted R-squared (training)':[],
                           'R-squared (test)':[],
                           'Adjusted R-squared (test)':[],
                           '5-Fold Cross Validation':[]})


#Defining a Function to Calculate the Adjusted  R2
def adjustedR2(r2,n,k):
    return r2-(k-1)/(n-k)*(1-r2)


df = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/algorithm_model/merge.csv',encoding='utf_8_sig')


# print(df.columns)

#pearson correaltion matrix，做出相关性矩阵
df = df[['dayofweek','weekofyear','month','day','year','period_of_month',
            'period2_of_month','week_of_month','quarter','sales_qty']]
features = ['dayofweek','weekofyear','month','day','year','period_of_month',
            'period2_of_month','week_of_month','quarter','sales_qty']

# features = df.columns

mask = np.zeros_like(df[features].corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=25)
print(df[features])
print(df[features].corr())
sns.heatmap(df[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", #"BuGn_r" to reverse
            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9})
plt.show()

#——————————————————————查看如果是有单一变量的线性回归是什么样的效果——————————————————
# train_data,test_data = train_test_split(df,train_size = 0.8,random_state=3)
# #
# lr = linear_model.LinearRegression()
# X_train = np.array(train_data['day'], dtype = pd.Series).reshape(-1,1)
# y_train = np.array(train_data['sales_qty'], dtype=pd.Series)
# lr.fit(X_train,y_train)
#
# X_test = np.array(test_data['day'], dtype=pd.Series).reshape(-1,1)
# y_test = np.array(test_data['sales_qty'], dtype=pd.Series)
#
# pred = lr.predict(X_test)
# rmsesm = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))
# rtrsm = float(format(lr.score(X_train, y_train),'.3f'))
# rtesm = float(format(lr.score(X_test, y_test),'.3f'))
# cv = float(format(cross_val_score(lr,df[['day']],df['sales_qty'],cv=5).mean(),'.3f'))
#
# print ("Average Price for Test Data: {:.3f}".format(y_test.mean()))
# print('Intercept: {}'.format(lr.intercept_))
# print('Coefficient: {}'.format(lr.coef_))
#
# r = evaluation.shape[0]
# evaluation.loc[r] = ['Simple Linear Regression','-',rmsesm,rtrsm,'-',rtesm,'-',cv]
# print(evaluation)

#查看真实的数据与拟合的曲线
# plt.figure(figsize=(6.5,5))
# plt.scatter(X_test,y_test,color='darkgreen',label="Data", alpha=.1)
# plt.plot(X_test,lr.predict(X_test),color="red",label="Predicted Regression Line")
# plt.xlabel("day", fontsize=15)
# plt.ylabel("sales_qty", fontsize=15)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.legend()
#
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['top'].set_visible(False)
# plt.show()


#查看每一个特征与目标值之间的关系
# df1 = df[['dayofweek','weekofyear','month','day','year','period_of_month',
#             'period2_of_month','week_of_month','quarter','code','sales_qty']]
# # features = ['dayofweek','weekofyear','month','day','year','period_of_month',
# #             'period2_of_month','week_of_month','quarter','code','sales_qty']
# h = df1.hist(bins=25,figsize=(16,16),xlabelsize='10',ylabelsize='10',xrot=-15)
# sns.despine(left=True, bottom=True)
# [x.title.set_size(12) for x in h.ravel()]
# [x.yaxis.tick_left() for x in h.ravel()]
# plt.show()



'''
一些时间序列的画图工具
'''
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# 时间序列图
def draw_ts(timeseries):
    timeseries.plot()
    plt.show()


# 移动平均图
def draw_trend(timeseries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeseries.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean = pd.ewma(timeseries, span=size)

    timeseries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()


# 语义描述
def testStationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    return dfoutput


# 自相关和偏相关图，默认阶数为31阶
def draw_acf_pacf(ts, lags=31):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=31, ax=ax2)
    plt.show()


# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 14:52
# @Author  : Ye Jinyu__jimmy
# @File    : feature_engineing

from sklearn import preprocessing
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import math
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA

from matplotlib.pylab import rcParams
import psycopg2
import pymysql
import time
import datetime

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

'''先将所有的特征与数据进行合并,因为后面将进行树模型的学习和训练，因此不需要归一化'''


# ——————————————————————————————————先获取各个的原始销量与特征数据————————————————————————————————
def data_prepare():
    weather_feature = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/weather_sql_read.csv',
                                  encoding='utf_8_sig', index_col=0)
    # sales = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data//sales.csv',encoding='utf_8_sig',index_col=0)
    price = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data//price_df.csv', encoding='utf_8_sig',
                        index_col=0)
    time_df = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/time_df.csv', encoding='utf_8_sig',
                          index_col=0)
    sales_feature = pd.read_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data//new_df.csv', encoding='utf_8_sig',
                                index_col=0)
    merge_01 = pd.merge(weather_feature, sales_feature, on='Account_date', how='left').fillna(method='pad')
    merge_02 = pd.merge(merge_01, price, on=['Account_date', 'sales_qty'], how='left')
    merge = pd.merge(merge_02, time_df, on=['Account_date'], how='left')
    merge = merge.dropna(how='any')
    # i= 0
    # while i < int(len(merge.columns)):
    #     print(merge.columns[i])
    #     if merge.columns[i] == 'Account_date' or merge.columns[i] == 'code' or merge.columns[i] == 'sales_qty':
    #         pass
    #     else:
    #         print('规范化')
    #         std = preprocessing.MinMaxScaler()
    #         merge[merge.columns[i]] = std.fit_transform(merge[[merge.columns[i]]])
    #
    #     i += 1
    # merge = merge.fillna(method='pad',limit=7)
    # # merge = merge.drop(['code','Account_date'],axis=1)
    # merge= merge.dropna(axis=0,how='any')
    print(merge)
    return merge


# ————————————————————————————特征选择——————————————————————————
# 带L1惩罚项的逻辑回归作为基模型的特征选择


# L1惩罚项降维的原理在于保留多个对目标值具有同等相关性的特征中的一个，
# 所以没选到的特征不代表不重要。故，可结合L2惩罚项来优化。具体操作为：若一个特征在L1中的权值为1，
# 选择在L2中权值差别不大且在L1中权值为0的特征构成同类集合，
# 将这一集合中的特征平分L1中的权值，故需要构建一个新的逻辑回归模型：

class LR_method(LogisticRegression):
    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

        # 权值相近的阈值
        self.threshold = threshold
        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                                    fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                    class_weight=class_weight,
                                    random_state=random_state, solver=solver, max_iter=max_iter,
                                    multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        # 使用同样的参数创建L2逻辑回归
        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                     intercept_scaling=intercept_scaling, class_weight=class_weight,
                                     random_state=random_state, solver=solver, max_iter=max_iter,
                                     multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        # 训练L1逻辑回归
        super(LR_method, self).fit(X, y, sample_weight=sample_weight)
        self.coef_old_ = self.coef_.copy()
        # 训练L2逻辑回归
        self.l2.fit(X, y, sample_weight=sample_weight)
        cntOfRow, cntOfCol = self.coef_.shape
        # 权值系数矩阵的行数对应目标值的种类数目
        for i in range(cntOfRow):
            for j in range(cntOfCol):
                coef = self.coef_[i][j]
                # L1逻辑回归的权值系数不为0
                if coef != 0:
                    idx = [j]
                    # 对应在L2逻辑回归中的权值系数
                    coef1 = self.l2.coef_[i][j]
                    for k in range(cntOfCol):
                        coef2 = self.l2.coef_[i][k]
                        # 在L2逻辑回归中，权值系数之差小于设定的阈值，且在L1中对应的权值为0
                        if abs(coef1 - coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                            idx.append(k)
                    # 计算这一类特征的权值系数均值
                    mean = coef / len(idx)
                    self.coef_[i][idx] = mean
        return self


# rf = RandomForestRegressor()
# rf.fit(feature, label)
# print("Features sorted by their score:")
# print(sorted(zip(map(lambda x: "%.4f"%x, rf.feature_importances_), names), reverse=True))

# use linear regression as the model
# from sklearn.linear_model import LinearRegression
# from sklearn.feature_selection import RFE
# lr = LinearRegression()
# #rank all features, i.e continue the elimination until the last one
# rfe = RFE(lr, n_features_to_select=1)
# rfe.fit(feature,label)
# print(len(names))
# print("Features sorted by their rank:")
# print(sorted(zip(rfe.ranking_, names)))


# ————————————————————————————PCA降维——————————————————————————
def pca_dim():
    X = np.array(
        [[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4
    pca = PCA(n_components=2)  # 降到2维
    pca.fit(X)  # 训练
    newX = pca.fit_transform(X)  # 降维后的数据
    # PCA(copy=True, n_components=2, whiten=False)
    print(pca.explained_variance_ratio_)  # 输出贡献率
    print(newX)

    # print("===============================")
    # print("标准化，返回如下：")
    # print(data)
    # #
    # # merge.apply(lambda x : (x-np.min(x))/(np.max(x)-np.min(x)))
    # # print(merge)


# ————————————————————————————返回所有的测试——————————————
if __name__ == '__main__':

    s = '杭州配送中心'
    min = s[0:2]
    print(min)
    # merge = data_prepare()
    # merge.to_csv('D:/AI/xianfengsg/forecaset/warehouse/V2.0/data/merge.csv',encoding='utf_8_sig',index=0)





