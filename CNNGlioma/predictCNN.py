import os
import sys
import time
sys.path.insert(1,'../headers/')

from time import gmtime, strftime
import numpy
import cPickle
from PIL import Image

import nibabel as nib

import theano
import theano.tensor as T

from convnet3d import *

import readPatGlioma

###########################################
def inspect_inputs(i, node, fn):
    print i, node, "input(s) value(s):", [input[0] for input in fn.inputs],

def inspect_outputs(i, node, fn):
    print "output(s) value(s):", [output[0] for output in fn.outputs]
############################################    
#Predict final or intermediate ?
predictFinal = False
#network parameters
num_classes = 3
n_fmaps = (15,25,50,num_classes)
#n_fmaps = (3,3,3,num_classes)		#feature map description :
#n_fmaps[0] - number of feature maps after 1st convolution
#n_fmaps[1] - number of feature maps after 2nd convolution
#n_fmaps[2] - number of output neurons after hidden layer
#n_fmaps[3] - number of output classes							

fmap_sizes = (5,5,2,1)		
# 0th value is fmap_size after 1st convolution and so on
############################################
plen = 61
offset = 20
numPred = plen - offset + 1

test_pat_num = 20

############################################
#Details pertaining to Multiple Sclerosis Dataset
num_channels = 4


############################################
#Initial Settings

done_looping = False
rng = numpy.random.RandomState(23455)

if(not predictFinal):
	savedModel = open('Output/CNNmodel.pkl','r')
	genVariables = cPickle.load(savedModel)
	epoch,pat_idx,tim_idx,ix,iy,iz,itr,best_validation_loss,best_itr = genVariables
else :
	savedModel = open('Output/FinalModel.pkl','r')

layer3convW = cPickle.load(savedModel)
layer3convb = cPickle.load(savedModel)
layer2convW = cPickle.load(savedModel)
layer2convb = cPickle.load(savedModel)
layer1convW = cPickle.load(savedModel)
layer1convb = cPickle.load(savedModel)
layer0convW = cPickle.load(savedModel)
layer0convb = cPickle.load(savedModel)

print "Loaded Model."

######################################
##Loading Dataset to Shared space

tpat = readPatGlioma.new(test_pat_num)
test_data = theano.shared(numpy.asarray(tpat.data,dtype = theano.config.floatX),borrow = True)
test_truth = theano.shared(numpy.asarray(tpat.truth,dtype = 'int32'),borrow = True)
img_shape = tpat.truth.shape


p_shape = (plen,plen,plen)
#ignore_border = True
idx = T.lscalar()
idy = T.lscalar()
idz = T.lscalar()
time_idx = T.lscalar()

x = T.ftensor4('x')
z = T.itensor3('y')
y = z.reshape([numPred*numPred*numPred,])

layer0_input = x.reshape([1,num_channels,p_shape[0],p_shape[1],p_shape[2]])
layer0conv = ConvLayer(rng,
					   input = layer0_input,
					   filter_shape = (n_fmaps[0],num_channels,fmap_sizes[0],fmap_sizes[0],fmap_sizes[0]),
					   image_shape = (1,num_channels,p_shape[0],p_shape[1],p_shape[2]),
					   W_init = layer0convW,
					   b_init = layer0convb,
					   sparse_count = 0 )
newlen = layer0conv.outputlen

layer0pool = PoolLayer(layer0conv.output,
					   image_shape = (1,n_fmaps[0],newlen[0],newlen[1],newlen[2]),
					   pool_size = (2,2,2),
					   sparse_count = 0)
newlen = layer0pool.outputlen

layer1conv = ConvLayer(rng,
				       layer0pool.output,
				       filter_shape = (n_fmaps[1],n_fmaps[0],fmap_sizes[1],fmap_sizes[1],fmap_sizes[1]),
				       image_shape = (1,n_fmaps[0],newlen[0],newlen[1],newlen[2]), 
					   W_init = layer1convW,
					   b_init = layer1convb,
				       sparse_count = 1 )
newlen = layer1conv.outputlen

layer1pool = PoolLayer(layer1conv.output,
					   image_shape = (1,n_fmaps[1],newlen[0],newlen[1],newlen[2]),
					   pool_size = (2,2,2),
					   sparse_count = 1 )
newlen = layer1pool.outputlen

layer2conv = ConvLayer(rng,
				   input = layer1pool.output,
				   filter_shape = (n_fmaps[2],n_fmaps[1],fmap_sizes[2],fmap_sizes[2],fmap_sizes[2]),
				   image_shape = (1,n_fmaps[1],newlen[0],newlen[1],newlen[2]),
				   W_init = layer2convW,
				   b_init = layer2convb,
				   sparse_count = 3)
newlen = layer2conv.outputlen

layer3conv = ConvLayer(rng,
				   input = layer2conv.output,
				   filter_shape = (n_fmaps[3],n_fmaps[2],fmap_sizes[3],fmap_sizes[3],fmap_sizes[3]),
				   image_shape = (1,n_fmaps[2],newlen[0],newlen[1],newlen[2]),
				   W_init = layer3convW,
				   b_init = layer3convb,
				   sparse_count = 0,
				   softmax = 1)
newlen = layer2conv.outputlen

cost = layer3conv.negative_log_likelihood(y)
"""+ T.sum(layer3conv.W**2) + T.sum(layer3conv.b**2) + T.sum(layer2conv.W**2) + T.sum(layer2conv.b**2) + \
	   T.sum(layer1conv.W**2) + T.sum(layer1conv.b**2) + T.sum(layer0conv.W**2) + T.sum(layer0conv.b**2)  
"""
test_model = theano.function(inputs = [idx,idy,idz],
							 outputs = [layer3conv.errors(y),layer3conv.y_pred,layer3conv.p_y_given_x], 
							 givens = {x: test_data[:,idx:idx+plen,idy:idy+plen,idz:idz+plen],
									   z: test_truth[idx+offset/2:idx+plen-offset/2 +1, 
									  			     idy+offset/2:idy+plen-offset/2 +1,
									  			     idz+offset/2:idz+plen-offset/2 +1]})
""",
							 mode=theano.compile.MonitorMode(pre_func=inspect_inputs,
															 post_func=inspect_outputs))
"""
############################################

maxXvalue = img_shape[0]-plen
maxYvalue = img_shape[1]-plen
maxZvalue = img_shape[2]-plen

xvalues = numpy.arange(0,maxXvalue,numPred)
yvalues = numpy.arange(0,maxYvalue,numPred)
zvalues = numpy.arange(0,maxZvalue,numPred)

if((maxXvalue-1)%numPred != 0):
	xvalues = numpy.append(xvalues,maxXvalue-1)
if((maxYvalue-1)%numPred != 0):
	yvalues = numpy.append(yvalues,maxYvalue-1)
if((maxZvalue-1)%numPred != 0):
	zvalues = numpy.append(zvalues,maxZvalue-1)

############################################

print('Predicting for test patient')
start_time = time.clock()

Prediction = numpy.zeros(img_shape)
img_shape = numpy.insert(img_shape,0,num_classes)
PostProbs = numpy.zeros(img_shape)

for ix in numpy.arange(len(xvalues)):
	for iy in numpy.arange(len(yvalues)):
			for iz in numpy.arange(len(zvalues)):
				errors,pred,p_y_given_x = test_model(xvalues[ix],yvalues[iy],zvalues[iz])
				pred = pred.reshape([numPred,numPred,numPred])
				p_y_given_x = p_y_given_x.transpose(1,0).reshape([num_classes,numPred,numPred,numPred])
				Prediction[xvalues[ix]+offset/2:xvalues[ix]+plen-offset/2 +1,
						   yvalues[iy]+offset/2:yvalues[iy]+plen-offset/2 +1,
						   zvalues[iz]+offset/2:zvalues[iz]+plen-offset/2 +1] = pred
				PostProbs[:,xvalues[ix]+offset/2:xvalues[ix]+plen-offset/2 +1,
						   	yvalues[iy]+offset/2:yvalues[iy]+plen-offset/2 +1,
						   	zvalues[iz]+offset/2:zvalues[iz]+plen-offset/2 +1] = p_y_given_x


end_time = time.clock()
print >> sys.stderr, ('Prediction done and it took '+' %.2fm ' % ((end_time - start_time) / 60.))
#output nii file
affine = [[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
for ix in numpy.arange(num_classes):
	img = nib.Nifti1Image(PostProbs[ix],affine)
	img.set_data_dtype(numpy.float32)
	nib.save(img,'Output/postprobs' + str(ix+1) + '.nii')
img = nib.Nifti1Image(Prediction, affine)
img.set_data_dtype(numpy.int32)
nib.save(img,'Output/Prediction.nii')
