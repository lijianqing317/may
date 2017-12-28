import struct
import matplotlib.pyplot as plt
import numpy as np
from numpy import matrix
import PIL.Image as Im
import scipy.io as sio
train_file = '/home/lijq/IdeaProjects/tr/imagedata/train-images.idx3-ubyte'
train_images = open(train_file,'rb')
buffers = train_images.read()
index = 0
magic, nums,nums_rows,nums_cloumns = struct.unpack_from('>IIII',buffers,index)
index +=struct.calcsize('>IIII')
'''images = struct.unpack_from('>784B',buffers,index)
index +=struct.calcsize('>784B')
ims = np.array(images).reshape(28,28)

fig = plt.figure()
fig.add_subplot(111)
plt.imshow(ims,cmap = 'gray')
plt.show()'''

'''for im in range(0,nums):
    ims = struct.unpack_from('>784B',buffers,index)
    index +=struct.calcsize('>784B')
    print index,im
    ims = np.array(ims,dtype='uint8')
    ims = ims.reshape(28,28)
    ims = Im.fromarray(ims)
    ims.save('./images/%s.bmp'%im,'bmp')'''
jt_data = sio.loadmat('./jiaotong.mat')
jt_d = jt_data['data']
jt_l = jt_data['labels']
jt_index = 0
'''for jt_image in range(0,len(jt_d)):
    jt_im =jt_d[jt_image]
    jt_im =jt_im*255
    jt_im = Im.fromarray(jt_im.astype(np.uint8))
    jt_index +=1
    label_index =np.argmax(jt_l[jt_image])
    jt_im.save('./jt_image/%s-%s.png'%(jt_index,label_index),'png')

    print jt_index,'-',label_index'''
my_image = matrix([[225,0,235,28],[78,39,100,4],[89,300,29,56]])
imag = Im.fromarray(my_image.astype(np.uint8))
imag.save('./my_image.png','png')





