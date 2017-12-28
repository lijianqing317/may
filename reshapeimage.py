import struct
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Im
train_file = '/home/lijq/IdeaProjects/tr/imagedata/train-images.idx3-ubyte'
train_images = open(train_file,'rb')
buffers = train_images.read()
index = 0
magic, nums,nums_rows,nums_cloumns = struct.unpack_from('>IIII',buffers,index)
index +=struct.calcsize('>IIII')
for im in range(0,nums):
    ims = struct.unpack_from('>784B',buffers,index)
    index +=struct.calcsize('>784B')
    print index,im
    ims = np.array(ims,dtype='uint8')
    ims = ims.reshape(28,28)
    ims = Im.fromarray(ims)
    plt.imshow(ims,cmap = 'gray')
    plt.show()
    imag = Im.fromarray(ims.astype(np.uint8))
    imag.save('./my_image.png','png')
