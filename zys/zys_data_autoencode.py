# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
import  keras.utils as np_util
from keras.optimizers import RMSprop
from keras.models import Sequential,Model
from keras.layers import Activation, TimeDistributed, Dense,Conv1D,MaxPool1D,\
    MaxPooling1D,GlobalAveragePooling1D,Dropout,Reshape,Flatten

from keras.optimizers import Adam
f=open('zyszl.csv')
df=pd.read_csv(f)
data=df.iloc[:,0:605].values
normalized_data=(data-np.min(data,axis=0))/(np.max(data,axis=0)-np.min(data,axis=0));
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
#encode
model = Sequential()
model.add(Dense(301,input_dim=603,activation='relu'))
model.add(Dense(60,activation='relu'))
model.add(Dense(8,name='out_layer'))
#decode
model.add(Dense(60,activation='relu'))
model.add(Dense(301,activation='relu'))
model.add(Dense(603,activation='relu'))
model.compile(optimizer='adam',loss='mse')
model.fit(np.array(data_train_x),np.array(data_train_x),epochs=1000,batch_size=9)
model.save('./auto_model.h5')
encoder_model = Model(inputs=model.input,outputs=model.get_layer('out_layer').output)
train_feature = encoder_model.predict(np.array(data_train_x))
test_feature = encoder_model.predict(np.array(data_test_x))
#encoder_model.save('./encoder_model.h5')
#sio.savemat('./extract_feature',{'train_feature':train_feature,'test_feature':test_feature})














