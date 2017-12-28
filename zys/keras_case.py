import keras
import numpy as np
import struct
from keras.models import  Sequential,Model
from keras.layers import Dense,Activation
from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import RMSprop
image_path = '/home/lijq/IdeaProjects/tr/imagedata/'
image_train_data ='train-images.idx3-ubyte'
image_train_label = 'train-labels.idx1-ubyte'
image_test_data = 't10k-images.idx3-ubyte'
image_test_label = 't10k-labels.idx1-ubyte'
image_list =[image_train_data,image_train_label,image_test_data,image_test_label]
image_data_list =[]
for i in range(len(image_list)):
    print image_path+image_list[i]
    image_data_list.append(open('%s%s'%(image_path,image_list[i]),'rb').read())
def getimagedata(buffer):
    index = 0
    magic, nums,nums_rows,nums_cloumns = struct.unpack_from('>IIII',buffer,index)
    index +=struct.calcsize('>IIII')
    image_tra = []
    for im in range(0,nums):
        ims = struct.unpack_from('784B',buffer,index)
        index +=struct.calcsize('784B')
        ims = np.array(ims)
        image_tra.append(ims)
    image_tra = np.array(image_tra)
    return image_tra
def getlabeldata(buffer):
    index = 0
    label_list = []
    magic, label_num = struct.unpack_from('>II', buffer, index)
    index += struct.calcsize('>II')
    for i in range(label_num):
        label_item = int(struct.unpack_from('>B', buffer, index)[0])
        label_list.append(label_item)
        index += struct.calcsize('>B')
    label_list = np.array(label_list)
    return label_list
train_image_data = getimagedata(image_data_list[0])
train_image_label = getlabeldata(image_data_list[1])
test_image_data = getimagedata(image_data_list[2])
test_image_label = getlabeldata(image_data_list[3])
print len(train_image_data),len(train_image_label),len(test_image_data),len(test_image_label)

train_image_data= train_image_data.reshape(train_image_data.shape[0], -1) / 255.   # normalize
test_image_data = test_image_data.reshape(test_image_data.shape[0], -1) / 255.   # normalize
train_image_label = np_utils.to_categorical(train_image_label,num_classes=10)
test_image_label = np_utils.to_categorical(test_image_label,num_classes=10)

'''model = Sequential()
model.add(Dense(32,input_dim = 784,activation='relu'))
model.add(Dense(25,activation='relu'))
model.add(Dense(784,activation='relu',name = 'middle_layer'))
model.add(Dense(10,activation='softmax',name = 'out_layer'))
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='rmsprop')
model.fit(train_image_data,train_image_label,epochs=2,batch_size=32)
model.save('./minist_model1.h5')
model_reload = load_model('./minist_model1.h5')
loss,accuracy = model_reload.evaluate(test_image_data,test_image_label)
print "   loss:",loss,"acc:",accuracy'''
model = Sequential()
model.add(Dense(301,input_dim=784,activation='relu'))
model.add(Dense(60,activation='relu'))
model.add(Dense(10,name='out_layer'))
#decode
model.add(Dense(10,activation='relu'))
model.add(Dense(301,activation='relu'))
model.add(Dense(784,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='mse')
model.fit(train_image_data,train_image_label,epochs=1,batch_size=32)
model.save('./auto_model1.h5')
