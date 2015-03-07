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

flair = mha.new()
t1 = mha.new()
t2 = mha.new()
t1c = mha.new()
truth = mha.new()

#flair.read_mha('../patient1/BRATS_HG0001_FLAIR.mha')
#t1.read_mha('../patient1/BRATS_HG0001_T1.mha')
#t2.read_mha('../patient1/BRATS_HG0001_T2.mha')
t1c.read_mha('../patient1/BRATS_HG0001_T1C.mha')
truth.read_mha('../patient1/BRATS_HG0001_truth.mha')
img = t1c.data[:,:,100]

truth = truth.data[:,:,100]
edema = truth.copy()
core = truth.copy()
edema[truth == 2] = 0
core[truth == 1] = 0
core = core/2

plt.hist(img.flatten(1),alpha = 0.2, bins = 1168,range = (1,1168),label = 'brain')
plt.hist(img.flatten(1),alpha = 0.3, weights = edema.flatten(1), bins = 1168,range = (1,1168),label = 'edema')
plt.hist(img.flatten(1),alpha = 0.5, weights = core.flatten(1), bins = 1168,range = (1,1168),label= 'core')
plt.legend(loc='upper right')

#pylab.subplot(2, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
#hist = numpy.histogram(a,range = (10,1168),weights = None, density = None) 


plt.title('Histogram of tumour, core and whole brain for T1C- slice 100 ')
plt.show()
