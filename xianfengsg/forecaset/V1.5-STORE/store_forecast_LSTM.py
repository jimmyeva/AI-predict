# -*- coding: utf-8 -*-
# @Time    : 2019/12/11 10:50
# @Author  : Ye Jinyu__jimmy
# @File    : store_forecast_LSTM

import pandas as pd
from datetime import datetime
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import concatenate
from math import sqrt
import datetime
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', 500)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
import os
# 注：设置环境编码方式，可解决读取数据库乱码问题
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
import process_data
from tqdm import tqdm

import psycopg2

# load data
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')



'''利用sklearn的预处理模块对类别特征“风向”进行编码，当然也可以对该特征进行one-hot编码。 
接着对所有的特征进行归一化处理，然后将数据集转化为有监督学习问题，同时将需要预测的当前时刻（t）的天气条件特征移除'''
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # convert series to supervised learning
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def cs_to_sl(dataset):
    # load dataset
    dataset.set_index("Account_date", inplace=True)
    values = dataset.values
    # integer encode direction,对文本特征进行哑编码
    encoder = LabelEncoder()
    values[:, 1] = encoder.fit_transform(values[:, 1])
    values[:, 2] = encoder.fit_transform(values[:, 2])
    values[:, 7] = encoder.fit_transform(values[:, 7])
    values[:, 8] = encoder.fit_transform(values[:, 8])
    values[:, 9] = encoder.fit_transform(values[:, 9])
    values[:, 10] = encoder.fit_transform(values[:, 10])
    values[:, 11] = encoder.fit_transform(values[:, 11])
    # ensure all data is float

    values = values.astype('float32')
    #normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # # drop columns we don't want to predict

    reframed.drop(reframed.columns[[13, 14, 15,16,17,18,19,20,21,22,23]], axis=1, inplace=True)

    print(reframed.head(5))
    return reframed, scaler

'''构造LSTM模型'''
#-------------首先，我们需要将处理后的数据集划分为训练集和测试集。为了加速模型的训练，我们仅利用第一年数据进行训练，然后利用剩下的4年进行评估。 
# 下面的代码将数据集进行划分，然后将训练集和测试集划分为输入和输出变量，最终将输入（X）改造为LSTM的输入格式，即[samples,timesteps,features]


def train_test(reframed):
    # split into train and test sets
    values = reframed.values
    # new_data= forecast_data.values
    train_len = round((len(reframed) * 0.8))
    train = values[:train_len, :]
    test = values[train_len:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    # new_x = new_data.reshape((new_data.shape[0], 1, new_data.shape[1]))
    return train_X, train_y, test_X, test_y

# design network
# design network
def fit_network(train_X, train_y, test_X, test_y, scaler):
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    '''模型评估'''

    # make a prediction
    yhat = model.predict(test_X)
    print(test_X)
    print(yhat)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    inv_y = scaler.inverse_transform(test_X)
    inv_y = inv_y[:, 0]
    print(inv_yhat)
    print(inv_y)
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

#根据返回的需要预测的特征信息做
def predict_pro(dataset):
    # load dataset
    dataset.set_index("Account_date", inplace=True)
    print(dataset.columns)
    values = dataset.values
    # integer encode direction,对文本特征进行哑编码
    encoder = LabelEncoder()
    values[:, 0] = encoder.fit_transform(values[:, 0])
    values[:, 1] = encoder.fit_transform(values[:, 1])
    values[:, 2] = encoder.fit_transform(values[:, 2])
    values[:, 3] = encoder.fit_transform(values[:, 3])
    values[:, 4] = encoder.fit_transform(values[:, 4])
    values[:, 5] = encoder.fit_transform(values[:, 5])
    values[:, 6] = encoder.fit_transform(values[:, 6])
    # ensure all data is float
    values = values.astype('float32')
    #normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    predict_data = series_to_supervised(scaled, 1, 1)
    # # drop columns we don't want to predict
    predict_data.drop(predict_data.columns[[12,13, 14, 15,16,17,18,19,20,21]], axis=1, inplace=True)
    return predict_data





#--------------------------设置主函数用于进行计算
if __name__ == '__main__':

    today = datetime.date.today()
    end_date = today.strftime('%Y%m%d')
    training_data,predict_data = process_data.main(end_date)
    sku_code = set(training_data['goods_code'].values)
    # print(len(sku_code))
    # for code in tqdm(sku_code):
    #     print(code)
    sales_mid = training_data[training_data['goods_code'] == '16040']
    sales_mid = sales_mid.drop(['store_code','goods_code','goods_name'],axis=1)
    sales_mid.to_csv('./sales_mid.csv',encoding='utf_8_sig')
    reframed, scaler = cs_to_sl(sales_mid)
    forecast_data = predict_pro(predict_data)
    train_X, train_y, test_X, test_y= train_test(reframed)

    fit_network(train_X, train_y, test_X, test_y, scaler)
