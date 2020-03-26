# -*- coding: utf-8 -*-
# @Time    : 2019/11/27 14:53
# @Author  : Ye Jinyu__jimmy
# @File    : LSTM-PRICE
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
import time
import math
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import optimizers
import os

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


#####当激活函数为 sigmoid 或者 tanh 时，要把数据正则话，此时 LSTM 比较敏感 设定 99.2% 是训练数据，余下的是测试数据
def create_train_test_data(dataset,look_back ):

    ############正则化
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    print(dataset.shape)
    datasetX, datasetY = create_dataset(dataset, look_back)

    ##########分割数据集
    train_size = int(len(datasetX) * 0.992)


    ####test_size = len(dataset) - train_size
    ################X=t and Y=t+1 时的数据，并且此时的维度为 [samples, features]
    trainX, trainY, testX, testY = datasetX[0:train_size,:],datasetY[0:train_size],datasetX[train_size:len(datasetX),:],datasetY[train_size:len(datasetX)]

    ###################投入到 LSTM 的 X 需要有这样的结构： [samples, timesteps, features]，所以做一下变换
    trainX = trainX.reshape (trainX.shape[0],look_back ,1)
    testX = testX.reshape(testX.shape[0], look_back , 1)

    return scaler,trainX,trainY,testX,testY

def build_model(layers,seq_len):  #layers [1,50,100,1]
    model = Sequential()

    model.add(LSTM(layers[1], input_shape=(seq_len, layers[0]),return_sequences=True))
    #model.add(Dropout(0.2))

    model.add(LSTM(layers[2],return_sequences=False))
    #model.add(Dropout(0.2))

    model.add(Dense(units=layers[3], activation='tanh'))

    start = time.time()
    #rmsprop = optimizers.RMSprop(lr=0.0001)
    #model.compile(loss="mse", optimizer=rmsprop)
    #model.compile(loss="mse", optimizer="adagrad")
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model

#直接全部预测
def predict_point_by_point(model, data):
    predicted = model.predict(data)
    print('predicted shape:',np.array(predicted).shape)  #(412L,1L)
    predicted = np.reshape(predicted, (len(predicted),))
    return predicted
#结果展示
def plot_results(predicted_data, true_data,predicted_next):
    PredictionNextX=[i for i in range(len(predicted_data),(len(predicted_data)+len(predicted_next)))]
    fig = plt.figure(facecolor='white',figsize=(10,5))
    #ax = fig.add_subplot(111)
    #ax.plot(true_data, label='True Data')
    plt.plot(true_data,color="black", label='True Data')
    plt.plot(predicted_data,color="blue", label='Prediction')
    plt.plot(PredictionNextX,predicted_next,color="red", label='PredictionNext')
    plt.xlabel("Time(s)") #X轴标签
    plt.ylabel("precipitation")  #Y轴标签
    plt.title("amount of precipitation") #图标题
    plt.legend()
    plt.show()
    #plt.savefig(filename+'.png')

def predict_next(model,scaler,dataset_pre,look_back,next_num):
    dataset_pre = scaler.transform(dataset[len(dataset)-look_back:len(dataset), 0])
    next_predicted_list=[]
    for i in range(next_num):
        dataXNext = dataset_pre[len(dataset_pre)-look_back:len(dataset_pre)]
        dataPreNext = dataXNext.reshape(1,look_back,1)
        next_predicted = model.predict(dataPreNext)
        next_predicted_array = np.array(next_predicted[0])
        next_predicted_list.append(next_predicted[0][0])
        dataset_pre = np.concatenate((dataset_pre,next_predicted_array))
        print("未来",i,"天的降水量预测值为：",next_predicted[0][0])
    return scaler.inverse_transform(next_predicted_list)

if __name__=='__main__':
    global_start_time = time.time()
    nb_epoch  = 100#迭代次数
    seq_len = 30#步长
    batchSize=365
    lstmFirstLayer=10#第一层lstm每步神经元个数
    lstmSecondLayer=20#第二层lstm每步神经元个数
    future_predict_num=30#连续预测未来30个数据

    print('> Loading data... ')
    dataframe = pd.read_csv('dy_rain_20_20_56691_.csv', usecols=[0])
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    scaler,trainX,trainY,testX,testY = create_train_test_data(dataset,seq_len)

    print('X_train shape:',trainX.shape)  #(3709L, 50L, 1L)
    print('y_train shape:',trainY.shape)  #(3709L,)
    print('X_test shape:',testX.shape)    #(412L, 50L, 1L)
    print('y_test shape:',testY.shape)    #(412L,)
    #预测未来数据
    #predict_next(model,dataset,seq_len,10)

    print('> Data Loaded. Compiling...')
    #建模
    model = build_model([1, lstmFirstLayer, lstmSecondLayer, 1],seq_len)
    #模型训练
    history = model.fit(trainX,trainY,batch_size=batchSize,epochs=nb_epoch,validation_split=0.05,verbose=1,shuffle=True)

    #测试数据预测
    testPredict = predict_point_by_point(model, testX)
    #测试数据反归一化
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    #计算测试数据集的rmse
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict))
    print ('Test Score: %.2f RMSE' % (testScore))
    #预测未来数据
    next_predicted_list = predict_next(model,scaler,dataset,seq_len,future_predict_num)
    #结果展示
    plot_results(testPredict,testY[0],next_predicted_list)
    #保存模型
    hyperparams_name=str(seq_len)+"-"+str(lstmFirstLayer)+"-"+str(lstmSecondLayer)+"-"+str(testScore)
    model.save(os.path.join(
        'MODEL', '{}_cont.h5'.format(hyperparams_name)), overwrite=True)
