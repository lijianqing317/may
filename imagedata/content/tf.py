import tensorflow as tf
import numpy as np
import struct
from keras.utils import np_utils
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

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.01)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])

l1 = add_layer(xs, 784, 30, activation_function=tf.nn.relu)
l2 = add_layer(l1, 30, 25, activation_function=tf.nn.relu)
l3 = add_layer(l2, 25, 10, activation_function=tf.nn.softmax)
#l2 = add_layer(l1, 30, 10, activation_function=tf.nn.softmax)
loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(l3),
                                              reduction_indices=[1])) # loss
correct_prediction = tf.equal(tf.argmax(ys,1), tf.argmax(l3,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
all = 60000
batch_size = all/32
for i in range(2):
    start = 0
    end = 0
    for j in range(31):
        end = start+batch_size
        sess.run(train_step, feed_dict={xs: train_image_data[start:end],
                                        ys: train_image_label[start:end]})
        start = end

        print sess.run(loss, feed_dict={xs: train_image_data[start:end],
                                                ys: train_image_label[start:end]})

'''batch_start=0
keep_prob = tf.placeholder("float")
batch_size=32
for i in range(200):
    sess.run(train_step, feed_dict={xs: train_image_data[batch_start:batch_start+batch_size],
                                        ys: train_image_label[batch_start:batch_start+batch_size]})
    batch_start +=batch_size
    ff={xs: train_image_data[batch_start:batch_start+batch_size],
                  ys: train_image_label[batch_start:batch_start+batch_size]}
    print(sess.run(loss, ff))'''





