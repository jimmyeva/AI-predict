# -*- coding: utf-8 -*-
# @Time    : 2019/12/12 15:24
# @Author  : Ye Jinyu__jimmy
# @File    : forecast_lstm_store_new

import keras
import datetime
import process_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,PolynomialFeatures, MinMaxScaler
# tf.reset_default_graph()
np.set_printoptions(threshold=np.inf)
tf.compat.v1.reset_default_graph()
'''此函数用来实际测试在生产环境中的销量数据'''


#定义常量
rnn_unit    = 10       #hidden layer units
input_size  = 15
output_size = 1
lr          = 0.0006         #学习率
#——————————————————导入数据——————————————————————

#----------------------转成年月日特征
def date_feature(dataset):
    dataset["year"] = pd.to_datetime(dataset['Account_date']).dt.year  # 年
    dataset["month"] = pd.to_datetime(dataset['Account_date']).dt.month  # 月
    dataset["day"] = pd.to_datetime(dataset['Account_date']).dt.day
    return dataset

#---------------------*特征处理-----------------------
class time_series_feature():
    def period_of_month(self,day):
        if day in range(1, 11): return 1
        if day in range(11, 21): return 2
        if day in range(21, 32): return 3

    def period2_of_month(self,day):
        if day in range(1, 16): return 1
        if day in range(16, 32): return 2

    def week_of_month(self,day):
        if day in range(1, 8): return 1
        if day in range(8, 15): return 2
        if day in range(15, 22): return 3
        if day in range(22, 32): return 4

    def quarter(self,month):
        if month in range(1, 4): return 1
        if month in range(4, 7): return 2
        if month in range(7, 10): return 3
        if month in range(10, 13): return 4


    def time_subset(self,x):
        x["dayofweek"]          = x['Account_date'].apply(lambda x: x.dayofweek)
        x["weekofyear"]         = x["Account_date"].apply(lambda x: x.weekofyear)
        x['month']              = x['Account_date'].apply(lambda x: x.month)
        x['day']                = x['Account_date'].apply(lambda x: x.day)
        x['year']               = x['Account_date'].apply(lambda x: x.year)
        x['period_of_month']    = x['day'].apply(lambda x: time_series_feature.period_of_month(self,x))
        x['period2_of_month']   = x['day'].apply(lambda x: time_series_feature.period2_of_month(self,x))
        x['week_of_month']      = x['day'].apply(lambda x: time_series_feature.week_of_month(self,x))
        x['quarter']            = x['month'].apply(lambda x: time_series_feature.quarter(self,x))
        return x

#对编标变量进行标签编码
def label_feature(dataset):
    gle = LabelEncoder()
    Weekday_labels = gle.fit_transform(dataset['general'])
    # genre_mappings = {index: label for index, label in enumerate(gle.classes_)}
    dataset['general_feature'] = Weekday_labels
    Weekday_labels = gle.fit_transform(dataset['wind_direction'])
    # genre_mappings = {index: label for index, label in enumerate(gle.classes_)}
    dataset['wind_direction_feature'] = Weekday_labels
    dataset = dataset.drop(['wind_direction', 'general'], axis=1)
    return dataset

'''获取训练集'''
def get_train_data(data, batch_size=60, time_step=7): #,train_begin=0,train_end=600
    batch_index=[]
    data_train=data[:]  #train_begin:train_end
    mean = np.mean(data_train,axis=0)
    std = np.std(data_train,axis=0)
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集
    for i in range(len(normalized_train_data) - time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[ i:i + time_step,1:]
       y=normalized_train_data[ i:i + time_step,0,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y,mean,std



# 获取测试集
def get_test_data(predict_data,sales_data,time_step = 1,test_begin = -30):
    data_test=sales_data[test_begin:]
    max = np.max(data_test,axis=0)
    min = np.min(data_test, axis=0)
    mean = np.mean(data_test,axis=0)
    std = np.std(data_test,axis=0)

    #因为销售的数据是在数组的第一列
    Q1 = np.percentile(data_test, 25,axis=0)[0]
    Q3 = np.percentile(data_test, 75,axis=0)[0]
    IQR = Q3 - Q1
    threshold_up = Q3 + 1.5 * IQR
    threshold_down = Q3 - 1.5 * IQR

    mean_predict = np.mean(predict_data,axis=0)
    std_predict = np.std(predict_data,axis=0)
    normalized_test_data = (predict_data - mean_predict) / (std_predict + 0.001)

    size = (len(normalized_test_data) + time_step) // time_step  # 有size个sample
    test_x = []
    for i in range(size):
        x = normalized_test_data[i * time_step:(i + 1) * time_step,:]
        test_x.append(x.tolist())
        # test_x.append((predict_data[(i + 1) * time_step:]).tolist())

    return mean,std,max,min,threshold_up,threshold_down,test_x

#获取测试集
# def get_test_data(predict_data,sales_data,time_step=7,test_begin=680):
#     print('历史集总长是：' , len(sales_data))
#     data_test=sales_data[test_begin:]
#     max = np.max(data_test,axis=0)
#     min = np.min(data_test, axis=0)
#     mean=np.mean(data_test,axis=0)
#     std=np.std(data_test,axis=0)
#
#     normalized_test_data = (data_test - mean) / (std+0.001)  #标准化,注意做一次拉普拉斯平滑处理
#     size=(len(normalized_test_data)+time_step)//time_step  #有size个sample
#     print('size的长度是：',size)
#
#     print('mean','\n',mean)
#     print('std','\n',std)
#     print('data_test', '\n', data_test)
#     print('data_test[0]', '\n', data_test[0])
#     print('normalized_test_data', '\n', normalized_test_data)
#     test_x,test_y=[],[]
#     for i in range(size):
#         x=normalized_test_data[i*time_step:(i+1)*time_step,1:]
#         y=normalized_test_data[i*time_step:(i+1)*time_step,0]
#         test_x.append(x.tolist())
#         test_y.extend(y)
#         # test_x.append((normalized_test_data[(i+1)*time_step:,1:]).tolist())
#         # test_y.extend((normalized_test_data[(i+1)*time_step:,0]).tolist())
#     print('test_x','\n',test_x)
#     return mean,std,max,min,test_x,test_y



#——————————————————定义神经网络变量————————————————————
#输入层、输出层权重、偏置

weights={
         'in':tf.Variable(tf.random.normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random.normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }

#——————————————————定义神经网络变量——————————————————
def lstm(X):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    # with tf.name_scope("fw_side"), tf.variable_scope("fw_side",reuse = True):

    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit, reuse=tf.compat.v1.AUTO_REUSE)
    # cell = tf.keras.layers.LSTMCell(rnn_unit, reuse=tf.compat.v1.AUTO_REUSE)

    init_state=cell.zero_state(batch_size,dtype=tf.float32)

    output_rnn,final_states = tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    #
    # output_rnn, final_states = tf.keras.layers.RNN(cell, input_rnn, dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果


    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states



#——————————————————训练模型——————————————————
#-----考虑到真实的训练环境，这里把每批次训练样本数（batch_size）、
# 时间步（time_step）、训练集的数量（train_begin,train_end）设定为参数，使得训练更加机动。
def train_lstm(training_data,batch_size=40,time_step=7):
    X=tf.compat.v1.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.compat.v1.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y,mean,std=get_train_data(training_data,batch_size,time_step)
    pred,_=lstm(X)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)

    # module_file = tf.train.latest_checkpoint('./')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)
        #重复训练2001次
        for i in range(2001):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],
                                 feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],
                                            Y:train_y[batch_index[step]:batch_index[step+1]]})
            print(i,loss_)
            if i % 1000==0:
                print("保存模型：",saver.save(sess,'./model/stock_banana.model',global_step=i))
    return mean,std



#————————————————预测模型————————————————————
def prediction(predict_data,sales_data,time_step):
    X = tf.compat.v1.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.compat.v1.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean,std,max,min,threshold_up,threshold_down,test_x = get_test_data(predict_data,sales_data,time_step)
    pred,_ = lstm(X)
    saver=tf.train.Saver(tf.global_variables())
    #这里还需要一个用于预测的数据集，后面模型进行预测的数据集
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('./model')
        saver.restore(sess, module_file)
        test_predict=[]
        print('预测的数量是：',len(test_x)-1)
        for step in range(len(test_x)-1):
            prob = sess.run(pred,feed_dict={X:[test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        # test_y = np.array(test_y) * std[0] + mean[0]
        test_predict=np.array(test_predict)*std[0]+mean[0]

        # -----------------对预测的数据进行处理
        # test_predict = test_predict[np.where(test_predict < 0, test_predict, 0)]
        # test_predict = np.where(test_predict > max[0], test_predict, max[0])
        # test_predict = np.where(test_predict < min[0], test_predict, min[0])

        print('test_predict','\n',test_predict)
        print('前30天的最大销售额是：',max[0])
        print('前30天的最小销售额是：',min[0])
        print('上异常值是：',threshold_up)
        print('下异常值是：', threshold_down)
        print('平均值是：', mean[0])
        test_predict[test_predict < min[0]] = min[0]
        test_predict[test_predict < threshold_down] = mean[0]
        test_predict[test_predict < 0] = 0
        test_predict[test_predict > max[0]] = max[0]
        print('test_predict','\n',test_predict)
        # acc = np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差
        # print('acc',acc)
        #以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        # plt.plot(list(range(len(test_y))), test_y,  color='r')
        plt.show()
        return test_predict

#先获取当前的叫货目录内含有的sku，再针对已有SKU进行预测
def get_sku_list():

    pass

def main(end_date,name,region):
    training_data, predict_data = process_data.main(end_date,name,region)
    # training_data.to_csv('./training_data.csv',encoding='utf_8_sig')
    # predict_data.to_csv('./predict_data.csv',encoding='utf_8_sig')
    training_data_final = date_feature(training_data)
    # predict_data = pd.read_csv('./predict_data.csv',encoding='utf_8_sig')
    predict_data_final = date_feature(predict_data)
    sku_code = set(training_data['goods_code'].values)
    print('参与预测的sku数量是：'+ str(len(sku_code)))
    #设置新的dataframe用来储存每个SKU的数量
    for code in tqdm.tqdm(sku_code):
        print('正在进行计算的sku是：'+ code)
        sales_mid = training_data_final[training_data_final['goods_code'] == code] # '16010'
        sales_mid = sales_mid.drop(['store_code', 'goods_code', 'goods_name'], axis=1)  #  'Unnamed: 0',
        sales_mid = sales_mid.fillna(0)
        print('sales_mid',sales_mid)
        print('predict_data_final',predict_data_final)
        test_predict = main_function(sales_mid,predict_data_final)
        fianl_data = predict_data_final
        fianl_data['forecast_qty'] = pd.Series(test_predict)
    return

#定义主函数用于每一个城市的所有的sku的计算操作
def main_function(sales_mid,predict_data):
    get_time_series = time_series_feature()
    sales_time_series = get_time_series.time_subset(sales_mid)
    predict_time_series = get_time_series.time_subset(predict_data)
    sales_mid = label_feature(sales_time_series)
    predict_data = label_feature(predict_time_series)
    # sales_mid.to_csv('./sales_mid.csv',encoding='utf_8_sig',index=False)
    # predict_data.to_csv('./predict_data.csv',encoding='utf_8_sig',index=False)
    def data_prepare(df):

        data=df.iloc[:,1:].values  #取第2-10列
        return data
    # sales_mid.to_csv('./sales_mid.csv',encoding='utf_8_sig',index=False)
    sales = data_prepare(sales_mid)
    predict = data_prepare(predict_data)
    # print(sales)
    train_lstm(sales)
    test_predict = prediction(predict,sales,7)
    return test_predict

#设置函数用于计算分门店独立计算SKU




if __name__ == '__main__':
    today = datetime.date.today()
    end_date = today.strftime('%Y%m%d')
    name ='杭州'
    region = '杭州'
    test_predict = main(end_date,name,region)



