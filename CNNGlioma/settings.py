#Glioma Settings


######################################################################################################################################
#######												Settings for Training												     #########
######################################################################################################################################

resumeTraining = False						# make this true to resume training from saved model	

############################################
# General  Hyper-parameters
learning_rate = 0.01 #e-04 				
n_epochs = 100
patience = 30000 
patience_increase = 2
improvement_threshold = 0.995
validation_frequency = 1000
saving_frequency = 1000

num_classes = 3
#network parameters
n_fmaps = (15,25,50,num_classes)		#feature map description :
#n_fmaps = (3,3,3,num_classes)			#feature map for debugging
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

############################################
#Details pertaining to Glioma Dataset
num_channels = 4
num_patients = 18
img_shape = (160,216,176)

valid_pat_num = 19
test_pat_num = 20

######################################################################################################################################
#######												Settings for Predicting												     #########
######################################################################################################################################