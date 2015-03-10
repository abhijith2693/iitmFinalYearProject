import os
import numpy
import nibabel as nib

IMAGES_PATH = '../MS_norm'
TRUTH_PATH = '../MS_atropos'

###############################################

class new():
	def __init__(self, pat_idx=None, time_idx = None):
		Patstr = str(pat_idx).zfill(2)
		TimStr = str(time_idx).zfill(2)
		flair = nib.load(IMAGES_PATH + '/training' + Patstr + '/' + TimStr + '/' + 'N_training' + Patstr + '_' + TimStr + '_flair_pp.nii').get_data()
		t1 = nib.load(IMAGES_PATH + '/training' + Patstr + '/' + TimStr + '/' + 'N_training' + Patstr + '_' + TimStr + '_mprage_pp.nii').get_data()
		pd = nib.load(IMAGES_PATH + '/training' + Patstr + '/' + TimStr + '/' + 'N_training' + Patstr + '_' + TimStr + '_pd_pp.nii').get_data()
		t2 = nib.load(IMAGES_PATH + '/training' + Patstr + '/' + TimStr + '/' + 'N_training' + Patstr + '_' + TimStr + '_t2_pp.nii').get_data()
		self.truth = nib.load(TRUTH_PATH + '/training' + Patstr + '/' + TimStr + '/' + 'truth7mask1.nii').get_data()
		self.data = numpy.array([flair,t1,pd,t2])
		