from keras.models import   Sequential
from keras.layers import Dense,Convolution2D,Flatten,Activation,MaxPooling2D
from keras.optimizers import Adam
import struct
import numpy as np
import keras.utils as np_utils
#getData

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

train_image_data= train_image_data.reshape(-1,1,28,28) / 255.   # normalize
test_image_data = test_image_data.reshape(-1,1,28,28) / 255.   # normalize
train_image_label = np_utils.to_categorical(train_image_label,num_classes=10)
test_image_label = np_utils.to_categorical(test_image_label,num_classes=10)


# denifition a cnn model
model =Sequential()
model.add(
    Convolution2D(
        batch_input_shape=(None,1,28,28),
        filters=32,
        kernel_size=5,
        padding='same',
        data_format='channels_first'
    )
)
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same'
))

model.add(Convolution2D(filters=64,
                        kernel_size=5,
                        padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    padding='same',
    pool_size=2
))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

#OPTIMIZER
adam = Adam(lr=0.001)
#model compile
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
#train
#model.fit(train_image_data,train_image_label,batch_size=32,epochs=1)
#test
loss,accuracy = model.evaluate(test_image_data,test_image_label)
model.save('./cnn_minist.h5')
print 'loss:'+loss,"accuracy:"+accuracy

