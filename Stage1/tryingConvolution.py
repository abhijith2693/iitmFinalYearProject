import sys
import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy
import cv2
import pylab
from PIL import Image
from matplotlib import pyplot as plt
sys.path.insert(1,'../headers/')

import mha

rng = numpy.random.RandomState(23455)
input = T.tensor4(name = 'input')

w_shp = (1,1,9,9)
w_bound = numpy.sqrt(9*9)
W = theano.shared(numpy.asarray(rng.uniform(low = -1.0/w_bound, high = 1.0/w_bound, size = w_shp), 
	dtype = input.dtype), name = 'W')

b_shp = (1,)
b= theano.shared(numpy.asarray(
	rng.uniform(low = -.5, high = .5, size = b_shp),
	dtype = input.dtype), name = 'b')

conv_out = conv.conv2d(input,W)

output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
f = theano.function([input],output)

T1C = mha.new()
T1C.read_mha('../patient1/BRATS_HG0001_T1C.mha')
img = T1C.data[:,:,100]
img = numpy.float32(img)/numpy.max(img)

img_ = img.reshape(1,1,160,216)
f_img = f(img_)

pylab.subplot(2, 1, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();
# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
pylab.subplot(2, 1, 2); pylab.axis('off'); pylab.imshow(f_img[0, 0, :, :])
#pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(f_img[0, 1, :, :])
pylab.show()

