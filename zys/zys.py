# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import  keras.utils as np_util

from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense,Conv1D,MaxPool1D,\
    MaxPooling1D,GlobalAveragePooling1D,Dropout,Reshape,Flatten

from keras.optimizers import Adam
f=open('zyszl.csv')
df=pd.read_csv(f)
data=df.iloc[:,0:605].values
normalized_data=(data-np.min(data,axis=0))/(np.max(data,axis=0)-np.min(data,axis=0));  #标准
#print normalized_data.max(),normalized_data.min()
'''test_y=np.array(test_y)*std[7]+mean[7]
test_predict=np.array(test_predict)*std[7]+mean[7]'''

#print len(normalized_train_data),len(normalized_test_data)

data_train=normalized_data[0:99]
data_test = normalized_data[20:]
data_train_x =[]
data_train_y = []
data_test_x =[]
data_test_y = []
for i in range(len(data_train)):
    #data_train_x.append(data_train[i][0:613])
    data_train_x.append(data_train[i][:603])
    data_train_y.append(data_train[i][603:])
for i in range(len(data_test)):
    #data_train_x.append(data_train[i][0:613])
    data_test_x.append(data_test[i][:603])
    data_test_y.append(data_test[i][603:])


#defination
'''model = Sequential()
model.add(Dense(1,input_dim=603))
model.add(Activation('tanh'))
model.compile(loss='mse',optimizer='sgd')
for i in range(5000):
    coss = model.train_on_batch(np.array(data_train_x),np.array(data_train_y))
    print coss
coss = model.evaluate(np.array(data_test_x),np.array(data_test_y),batch_size=19)
print coss
predict = model.predict(np.array(data_test_x))
index=[]
for i in range(len(predict)):
    index.append(i)

plt.scatter(index,predict)
plt.scatter(index,data_test_y)
plt.show()'''


data_test_x=np.array(data_test_x)
data_test_y = np.array(data_test_y)
yy = data_test_y
data_test_x.resize((len(data_test_x),3,201))
data_test_y.resize((len(data_test_y),1,1))
data_train_x =np.array(data_test_x)
data_train_y = np.array(data_test_y)
data_train_x.resize((len(data_train_x),3,201))
data_train_y.resize((len(data_train_y),1,1))


model = Sequential()
model.add(Conv1D(8,3,activation='relu',input_shape=(3,201)))
model.add(Activation("relu"))
model.add(Dense(1))
model.compile(loss='mse',optimizer='sgd')
model.fit(np.array(data_train_x),np.array(data_train_y),epochs=100,batch_size=4)
y = model.predict(data_test_x)
x = []
for i in range(19):
    x.append(i)
    plt.scatter(y[i],y[i],c='r',marker="^")
    plt.plot(y[i],yy[i],c='g',marker="^")
plt.xlabel('y')
plt.ylabel('predict')
plt.show()
print yy
#print 'aard',np.mean(abs(y-yy)/yy*100)
print 'wucha',np.mean(sum(abs(yy-y)))
'''
data_test_x = np.expand_dims(data_test_x, axis=3)
model = Sequential()
model.add(Conv1D(64, 3, activation='relu',input_shape=(3,201)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
data_test_x.resize((19,3,201))
data_test_y.resize((19,1,1))l
print data_test_x.shape,data_test_y.shape
model.fit(np.array(data_test_x), data_test_y)
score = model.evaluate(data_test_x, data_test_y, batch_size=1)'''















