# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:29:48 2016

@author: root
"""
import numpy
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import pyplot as plt

matfn=u'/home/lijq/IdeaProjects/tr/jiaotong.mat';
jiaotong=sio.loadmat(matfn) 
jiaotong_data=jiaotong['data']
plt.imshow(jiaotong_data[0])
print jiaotong_data
jiaotong_label=jiaotong['labels']
traindata=jiaotong_data[0:2499,:]
trainlabel=jiaotong_label[0:2499,:]
testdata=jiaotong_data[2500:2908,:]
testlabel=jiaotong_label[2500:2908,:]
print(traindata.shape)
print(trainlabel.shape)
print(testdata.shape)
print(testlabel.shape)
x = tf.placeholder(tf.float32, [None, 32,32,3])                        #输入的数据占位符
y_actual = tf.placeholder(tf.float32, shape=[None, 5])            #输入的标签占位符

#定义一个函数，用于初始化所有的权值 W
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
  
#定义一个函数，用于构建卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#定义一个函数，用于构建池化层
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#构建网络
W_conv1 = weight_variable([5, 5, 3, 32])      
b_conv1 = bias_variable([32])       
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)     #第一个卷积层
h_pool1 = max_pool(h_conv1)                                  #第一个池化层

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)      #第二个卷积层
h_pool2 = max_pool(h_conv2)                                   #第二个池化层

W_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])              #reshape成向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)    #第一个全连接层

keep_prob = tf.placeholder("float") 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                  #dropout层

W_fc2 = weight_variable([1024, 5])
b_fc2 = bias_variable([5])
y_predict=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)   #softmax层

cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))     #交叉熵
train_step = tf.train.GradientDescentOptimizer(5e-5).minimize(cross_entropy)    #梯度下降法
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                 #精确度计算
sess=tf.InteractiveSession()                          
sess.run(tf.global_variables_initializer())
for i in range(10):
    for j in range(50):
        batch_xs = traindata[j*50:j*50+49,:]
        batch_ys = trainlabel[j*50:j*50+49,:]
        train_step.run(feed_dict={x: batch_xs, y_actual: batch_ys, keep_prob: 0.5})
    test_acc=accuracy.eval(feed_dict={x: testdata, y_actual: testlabel, keep_prob: 1})
    print('step',i,"test accuracy",test_acc)
