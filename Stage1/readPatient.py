import mha
import numpy

IMAGES_FOLDER_PATH = '/home/abhijith/BRATS/Images'
TRUTH_FOLDER_PATH = '/home/abhijith/BRATS/Truth'

###############################################

class new():
	def __init__(self,pat_idx=None):
		flair = mha.new(IMAGES_FOLDER_PATH + '/BRATS_HG00' + str(pat_idx).zfill(2) + '/BRATS_HG00' + str(pat_idx+1).zfill(2) + '_FLAIR.mha')
		t1 = mha.new(IMAGES_FOLDER_PATH + '/BRATS_HG00' + str(pat_idx).zfill(2)  + '/BRATS_HG00' + str(pat_idx+1).zfill(2) + '_T1.mha')
		t2 = mha.new(IMAGES_FOLDER_PATH + '/BRATS_HG00' + str(pat_idx).zfill(2)  + '/BRATS_HG00' + str(pat_idx+1).zfill(2) + '_T2.mha')
		t1c = mha.new(IMAGES_FOLDER_PATH + '/BRATS_HG00' + str(pat_idx).zfill(2)  + '/BRATS_HG00' + str(pat_idx+1).zfill(2) + '_T1C.mha')
		self.truth = mha.new(TRUTH_FOLDER_PATH + '/BRATS_HG00' + str(pat_idx+1).zfill(2) + '_truth.mha')
		self.data = numpy.array([flair.data,t1.data,t2.data,t1c.data])