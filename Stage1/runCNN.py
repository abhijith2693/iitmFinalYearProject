"""
x = 16,36,56,76 


x-16 : x
x : x + plen-31 
x + plen-31 : x+plen - 32 +16



"""
import os
import sys
import time
sys.path.insert(1,'../headers/')

import numpy
import cPickle
from PIL import Image

import nibabel as nib

import theano
import theano.tensor as T

from convnet3d import *
from mlp import HiddenLayer
from logistic_sgd import *
from time import gmtime, strftime

import readPatMS
############################################

resumeTraining = False							# make this true to resume training from saved model	

############################################
# General  Hyper-parameters
learning_rate = 0.1 				
n_epochs = 2
patience = 10000
patience_increase = 2
improvement_threshold = 0.995
validation_frequency = 1000
saving_frequency = 10

#network parameters
n_fmaps = (15,25,25,2)		#feature map description :
#n_fmaps[0] - number of feature maps after 1st convolution
#n_fmaps[1] - number of feature maps after 2nd convolution
#n_fmaps[2] - number of output neurons after hidden layer
#n_fmaps[3] - number of output classes							

fmap_sizes = (5,5,5,1)		
# 0th value is fmap_size after 1st convolution and so on
############################################
#"""
#Image details and settings for Multiple Sclerosis
plen = 51
offset = 32
numPred = plen - offset + 1

valid_pat_num = 5
test_pat_num = 5
valid_tstamp = 1
test_tstamp = 2

############################################
#Details pertaining to Multiple Sclerosis Dataset
num_channels = 4
num_patients = 4
TStamps = (4,4,5,4)			
img_shape = (181,217,181)

############################################
#Initial Settings

done_looping = False
rng = numpy.random.RandomState(23455)

if(resumeTraining)
	savedModel = file('CNNmodel','wb')
	genVariables = cPickle.load(savedModel)
	epoch,pat_idx,tim_idx,x,y,z,itr,best_validation_loss,best_itr = genVariables
	layer3convW = cPickle.load(savedModel)
	layer3convb = cPickle.load(savedModel)
	layer2convW = cPickle.load(savedModel)
	layer2convb = cPickle.load(savedModel)
	layer1convW = cPickle.load(savedModel)
	layer1convb = cPickle.load(savedModel)
	layer0convW = cPickle.load(savedModel)
	layer0convb = cPickle.load(savedModel)
	
else
	epoch,pat_idx,tim_idx,x,y,z,itr,best_validation_loss,best_itr = [0,0,0,0,0,0,0,numpy.inf,0]
######################################
p_shape = (plen,plen,plen)
#ignore_border = True
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

test_model = theano.function([x,z],[layer3conv.errors(y),layer3conv.y_pred])
valid_model = theano.function([x,z],layer3conv.errors(y))

params = layer3conv.params + layer2conv.params + layer1conv.params + layer0conv.params
masks = layer3conv.masks + layer2conv.masks + layer1conv.masks + layer0conv.masks
grads = T.grad(cost,params)
#update only sparse elements
updates = [(param_i,param_i-learning_rate*grad_i*mask_i) for param_i,grad_i,mask_i in zip(params,grads,masks)]
train_model = theano.function([x,z],cost,updates = updates)

############################################

maxXvalue = img_shape[0]-plen
maxYvalue = img_shape[1]-plen
maxZvalue = img_shape[2]-plen

xvalues = numpy.arange(0,maxXvalue,numPred)
yvalues = numpy.arange(0,maxYvalue,numPred)
zvalues = numpy.arange(0,maxZvalue,numPred)
############################################
strftime("start time is %Y-%m-%d %H:%M:%S", gmtime())
start_time  = time.clock()

while(epoch < n_epochs) and (not done_looping):
	while(pat_idx < num_patients):
		while(tim_idx < TStamps[pat_idx]):
			pat = readPatMS.new(pat_idx+1,tim_idx+1)
			print (' epoch : %i, patient : %i, time stamp : %i \n' %(epoch,pat_idx+1,tim_idx+1))
			while(x < maxXvalue):
				while(y < maxYvalue):
					while(z < maxZvalue):
						costTrain = train_model(pat.data[:,x:x+plen,y:y+plen,z:z+plen],
											  	pat.truth[x+offset/2:x+plen-offset/2 +1,
											  		      y+offset/2:y+plen-offset/2 +1,
											  		      z+offset/2:z+plen-offset/2 +1])
						
						if(itr%saving_frequency == 0)
							save_file = file('CNNmodel.pkl', 'wb')
							genVariables = [epoch,pat_idx,tim_idx,x,y,z+numPred,itr+1,best_validation_loss,best_itr]
							cPickle.dump(genVariables,save_file,protocol = cPickle.HIGHEST_PROTOCOL) 
							for i in xrange(len(params)):
								cPickle.dump(params[i].get_value(borrow=True), save_file, protocol = cPickle.HIGHEST_PROTOCOL)    
							save_file.close()
						
						itr = itr + 1
						if itr % 100 == 0 :
							print 'training at iteration : %i ' % (itr+1) 

						if(itr%validation_frequency == 0):
							vpat = readPatMS.new(valid_pat_num,valid_tstamp)
							valid_losses = numpy.zeros([len(xvalues),len(yvalues),len(zvalues)])
							vx,vy,vz = [0,0,0]	
							for x in xvalues:
								for y in yvalues:
									for z in zvalues:
										valid_losses[vx/numPred,vy/numPred,vz/numPred] = valid_model(vpat.data[:,vx:vx+plen,vy:vy+plen,vz:vz+plen],
																			   						 vpat.truth[vx+offset/2:vx+plen-offset/2 +1,
																			   			  						vy+offset/2:vy+plen-offset/2 +1,
																			   			  						vz+offset/2:vz+plen-offset/2 +1])
							this_validation_loss = numpy.mean(valid_losses)
							
							if this_validation_loss < best_validation_loss:
								if this_validation_loss < best_validation_loss * improvement_threshold:
									patience = max(patience, itr * patience_increase)
									best_validation_loss = this_validation_loss
									best_itr = itr	

						if patience <= itr :
							done_looping = True
							break
						z = z + numPred
					y = y + numPred
				x = x + numPred			
			tim_idx = tim_idx + 1				
		pat_idx = pat_idx + 1					
	epoch = epoch + 1			


end_time = time.clock()
print('Optimization complete.')
strftime("end time is %Y-%m-%d %H:%M:%S", gmtime())
print >> sys.stderr, ('Training took '+' %.2fm ' % ((end_time - start_time) / 60.))
print('Best validation score of %f %% obtained at iteration %i, '
	   %(best_validation_loss*100.,best_itr + 1))

# Save model to file after whole optimization is done
save_file = file('FinalModel.pkl', 'wb')
for i in xrange(len(params)):
	cPickle.dump(params[i].get_value(borrow=True), save_file, protocol = cPickle.HIGHEST_PROTOCOL)    
save_file.close()

print('Predicting for test patient')
start_time = time.clock()
tpat = readPatMS.new(test_pat_num,test_tstamp)
Prediction = numpy.zeros(img_shape)

for x in xvalues:
	for y in yvalues:
			for z in zvalues:
				errors,pred = test_model(pat.data[:,x:x+plen,y:y+plen,z:z+plen],
									 pat.truth[x+offset/2:x+plen-offset/2 +1,
									   	       y+offset/2:y+plen-offset/2 +1,
											   z+offset/2:z+plen-offset/2 +1])
				pred = pred.reshape([numPred,numPred,numPred])
				Prediction[x+offset/2:x+plen-offset/2 +1,
						   y+offset/2:y+plen-offset/2 +1,
						   z+offset/2:z+plen-offset/2 +1] = pred

end_time = time.clock()
print >> sys.stderr, ('Prediction done and it took '+' %.2fm ' % ((end_time - start_time) / 60.))
#output nii file
affine = [[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
img = nib.Nifti1Image(Prediction, affine)
img.set_data_dtype(numpy.float32)
nib.save(img,'prediction.nii')