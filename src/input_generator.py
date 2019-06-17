from keras.utils import Sequence
from keras import backend as K
import numpy as np
import netdata
import os


class input_generator(Sequence):
	'Generates data for waveNILM'
	def __init__(self, batch_size=1, out_dim = 2, feature_length = 2, sample_inds = [0,0], sample_size = 1440,
				 shuffle=False, noise_mode  = 0,path = '../data/sample_size_1440_app_inds_[9_10]_meta_inds_[1]', test = False):
		'Initialization'
		self.sample_inds = sample_inds #sample numbers for this input_generator
		self.sample_size = sample_size #size of each sample
		self.batch_size = batch_size
		self.feature_length = feature_length # Ammount of features for each sample
		self.out_dim = out_dim # Ammount of output appliamces/sub-meters
		self.shuffle = shuffle # Whether to shuffle samplesafter every epoch
		self.path = path # Where to load files from
		self.noise_mode = noise_mode # See netdata.py for details
		self.test = test # Flag for separating testing data from training data
		self.indexes = [] # Placeholder for indexes
		self.on_epoch_end() # Begin by runnning the end of epoch routine to shuffle/create indexes
		
		
		

	def __len__(self):
		'Denotes the number of batches per epoch'
		if len(self.sample_inds)==2:
			return int(np.floor((self.sample_inds[1]-self.sample_inds[0]) / self.batch_size))
		else:
			return int(np.floor(len(self.sample_inds)//self.batch_size))

	def __getitem__(self, index):
		'Loads one batch of data from samples created by make_data_folder'
		
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Placeholders for batch data and labels
		X = np.empty((self.batch_size, self.sample_size, self.feature_length))
		y = np.empty((self.batch_size,self.sample_size, self.out_dim))
		
		# Load data
		for i,ind in enumerate(indexes):
			# load sample and place in placeholder
			x = np.load(self.path +('/Input_Sample_No_%s.npy' %ind))
			x = x if x.any() else K.epsilon()
			X[i,] = x
			

			# load target data
			ytmp = np.load(self.path +('/Output_Sample_No_%s.npy' %ind))
			
			# prevent all zero inputs for training
			if not self.test:
				for j in range(ytmp.shape[2]):
					ytmp[:,:,j] = ytmp[:,:,j] if ytmp[:,:,j].any() else K.epsilon()
			
			# fill placeholder with fixed data	
			y[i,] = ytmp
		return X, y
		
	   
	def on_epoch_end(self):
		'shuffle indexes after each epoch'
		if self.indexes == []: # Create indexes if they don't already exist
			if len(self.sample_inds) == 2:
				self.indexes = np.arange(self.sample_inds[0],self.sample_inds[1],dtype = 'int32')
			else:
				self.indexes = self.sample_inds
		if self.shuffle == True: # Shuffle indexes if needed
			np.random.shuffle(self.indexes)  



def make_data_folder(in_pth,
	time_inds = [], # which timesteps to include in the data folder
	sample_size = 1440, # number of timesteps per sample
	app_inds = [9,10], # Which output appliances to use
	noise_mode = 0, # See netdata.py for details
	effective_sample_size = 0, # Effective sample size after account for overlap, 0 denotes no overlap
	meter_max = [2**7,2**14,2**12,2**14], # Normalization coeffecient for each electrical feature
	source = '', # Dataset source file location
	agg_ind = [1]): # Location of aggregate data in dataset array
	
	'A Function for creating the data folder to be used by input_generator'
	'Note that this function is written with the assumption the entire dataset can be loaded to memory'
	'Consider rewriting this to create the data folder sequentially for larger datasets'
	
	data = netdata.NILMdata(fill = True, time_inds = time_inds,noise_mode = noise_mode, source = in_pth + '/' + source,agg_ind = agg_ind)
	data.arrange_for_waveNILM(sample_size, time_inds = time_inds,app_inds = app_inds, use_time = False,effective_sample_size = effective_sample_size,meter_max = meter_max)
	
	pth = in_pth + ('/sample_size_%s_app_inds_%s_effective_sample_size_%s_noise_mode_%s_data_source_%s' %(sample_size,app_inds,effective_sample_size,noise_mode,source[:-4])).replace(',','_')
	try:
		if not os.path.isdir(pth):
				os.mkdir(pth)
	except:
		raise BaseException('unable to create directory for saving data, please check savepath = %s is appropriate (all sub-directories exist, writing permissions etc.)' %pth)

	assert data.NILMgt.any(), 'Ground Truth is empty'
	for i in range(data.NILMin.shape[0]):
		np.save(pth +('/Input_Sample_No_%s.npy' %i),data.NILMin[i,:,:].reshape(1,data.NILMin.shape[1],data.NILMin.shape[2]))
		out = data.NILMgt[i,:,:].reshape(1,sample_size,data.NILMgt.shape[2])
		
		np.save(pth +('/Output_Sample_No_%s.npy' %i),out)
		
	

 