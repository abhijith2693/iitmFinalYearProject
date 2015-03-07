import os
import sys
import time

import theano
import theano.tensor as T
import theano.tensor.nlinalg as LA
import numpy
from theano.printing import pp

import mha
sys.path.insert(1,'../headers/')


flair = mha.new('../patient1/BRATS_HG0001/BRATS_HG0001_FLAIR.mha') 
t1 = mha.new('../patient1/BRATS_HG0001/BRATS_HG0001_T1.mha') 
t2 = mha.new('../patient1/BRATS_HG0001/BRATS_HG0001_T2.mha') 
t1c = mha.new('../patient1/BRATS_HG0001/BRATS_HG0001_T1C.mha') 
truth = mha.new('../patient1/BRATS_HG0001_truth.mha')





