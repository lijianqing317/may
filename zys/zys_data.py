# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import  keras.utils as np_util
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense,Conv1D,MaxPool1D,\
    MaxPooling1D,GlobalAveragePooling1D,Dropout,Reshape,Flatten

from keras.optimizers import Adam
f=open('zyszl.csv')
df=pd.read_csv(f)
data=df.iloc[:,0:605].values
normalized_data=(data-np.min(data,axis=0))/(np.max(data,axis=0)-np.min(data,axis=0));
#normalized_data=(data)/np.sum(data,axis=0);
origin_data_max = np.max(data,axis=0)
origin_data_min = np.min(data,axis=0)
#print len(normalized_train_data),len(normalized_test_data)
train_test_cut = 99
data_train=normalized_data[0:train_test_cut]
data_test = normalized_data[train_test_cut:]

data_train_x =[]
data_train_y = []
data_test_x =[]
data_test_y = []
for i in range(len(data_train)):
    data_train_x.append(data_train[i][:603])
    data_train_y.append(data_train[i][603:])
for i in range(len(data_test)):
    data_test_x.append(data_test[i][:603])
    data_test_y.append(data_test[i][603:])

data_test_x=np.array(data_test_x)
data_test_y = np.array(data_test_y)
data_train_x =np.array(data_train_x)
data_train_y = np.array(data_train_y)
test_origin_y = data_test_y
train_origin_y = data_train_y
data_test_x.resize((len(data_test_x),3,201))
data_test_y.resize((len(data_test_y),1,1))
data_train_x.resize((len(data_train_x),3,201))
data_train_y.resize((len(data_train_y),1,1))

model = Sequential()
model.add(Conv1D(64,3,activation='relu',input_shape=(3,201)))
model.add(MaxPooling1D(pool_size=1,padding='valid',strides=3))
model.add(MaxPooling1D(pool_size=1,padding='valid',strides=3))
model.add(Dropout(rate=0.06))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))
model.compile(loss='mse',optimizer='sgd')
model.fit(np.array(data_train_x),np.array(data_train_y),epochs=200,batch_size=9)
model.save('./model.h5')
train_predict_y = model.predict(data_train_x)
test_predict_y = model.predict(data_test_x)

def getResult(origin,predict):
    aard = np.mean(np.abs(predict-origin)/origin*100)
    rmse = np.sqrt(np.mean(np.square(origin-predict)))
    return (aard,rmse)
train_aard,train_rmse = getResult(train_origin_y,train_predict_y)
test_aard,test_rmse = getResult(test_origin_y,test_predict_y)

print 'train_AARD: %s,train_RMSE: %s'%(train_aard,train_rmse)
print 'test_AARD: %s,test_RMSE: %s'%(test_aard,test_rmse)

ax1 =plt.subplot(2,1,1)
for i in range(len(train_origin_y)):
    plt.scatter(train_origin_y[i],train_origin_y[i],c='r',marker="*")
    plt.plot(train_predict_y[i],train_origin_y[i],c='g',marker="^")
plt.xlabel('train_origin')
plt.ylabel('train_predict')
ax2 = plt.subplot(2,1,2)
for i in range(len(test_origin_y)):
    plt.scatter(test_origin_y[i],test_origin_y[i],c='r',marker="*")
    plt.plot(test_origin_y[i],test_predict_y[i],c='g',marker="^")
plt.xlabel('test_origin')
plt.ylabel('test_predict')
plt.show()

















