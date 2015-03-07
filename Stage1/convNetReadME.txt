Image based conv nets

Details for Glioma data - Brats 2013 dataset
input image patch
1 file per batch
4 channels
image size 176,160,216
patch size 31x31x31

first layer :
do normal convolution :
use 5x5x5 filter
new image size = 172,156,212 
patch size 27x27x27

and pooling with stride 1
use 2x2x2 filter
new image size = 171,155,211
patch size 14x14x14

second layer
do convolution with sparse kernel with 1 gap
use 5x5x5 filter = 9x9x9 filter
new image size = 163,147,203
patch size = 10x10x10

and pooling layer with stride 1
use 2x2x2 = 3x3x3
new image size = 161,145,201
patch size = 5x5x5

hidden layer 
use convolution 17x17x17 and tanh
new image size = 145x129x185

use softmax



