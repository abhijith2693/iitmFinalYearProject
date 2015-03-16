import mha
import numpy

IMAGES_FOLDER_PATH = '/media/brain/1A34723D34721BC7/GliomaData/BRATS2013/Images'
TRUTH_FOLDER_PATH = '/media/brain/1A34723D34721BC7/GliomaData/BRATS2013/Truth'

###############################################

class new():
	def __init__(self,pat_idx=None):
		fnames = [22,24,25,26,27]
		if(pat_idx > 15):
			pat_idx = fnames[pat_idx-16] 
		flair = mha.new(IMAGES_FOLDER_PATH + '/BRATS_HG00' + str(pat_idx).zfill(2) + '/BRATS_HG00' + str(pat_idx).zfill(2) + '_FLAIR.mha')
		t1 = mha.new(IMAGES_FOLDER_PATH + '/BRATS_HG00' + str(pat_idx).zfill(2)  + '/BRATS_HG00' + str(pat_idx).zfill(2) + '_T1.mha')
		t2 = mha.new(IMAGES_FOLDER_PATH + '/BRATS_HG00' + str(pat_idx).zfill(2)  + '/BRATS_HG00' + str(pat_idx).zfill(2) + '_T2.mha')
		t1c = mha.new(IMAGES_FOLDER_PATH + '/BRATS_HG00' + str(pat_idx).zfill(2)  + '/BRATS_HG00' + str(pat_idx).zfill(2) + '_T1C.mha')
		self.truth = mha.new(TRUTH_FOLDER_PATH + '/BRATS_HG00' + str(pat_idx).zfill(2) + '_truth.mha').data
		self.data = numpy.array([flair.data,t1.data,t2.data,t1c.data])

