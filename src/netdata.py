import numpy as np 
import pickle


def load_data(datasource = '../data/AMPds2_In_P_Out_P.dat'):
		with open (datasource, 'rb') as f:
			data = pickle.load(f)
		return data


class NILMdata(object):
	'data class for NILM data for waveNILM'
	def __init__(self,
		fill = False, # Whether or not to fill class with data upon initialization
		source = '', # Data source
		time_inds = [], # Time inds for loading only a portion of the data source
		noise_mode = 0, # Data noise mode, see more details below
		agg_ind = [1], # Which index to use for aggregate measurement from data file.
		gr_tru_ign_ind = [23], #Indices to ignore, in case there we want to disaggregate everything but some indicies are unused.
		):

		
		self.noise_mode = noise_mode
		self.T = np.array([])
		self.agg_pow = np.array([])
		self.gr_tru = np.array([])
		self.NILMin = np.array([])
		self.NILMgt = np.array([])
		self.NILMhat = np.array([])
		self.readyforNILM = {}
				
		if fill:
			if source:
				data  = load_data(datasource = source)
			else:
				data = load_data()
			gr_tru_ind = np.arange(agg_ind[-1]+1,data.shape[1]) # What is this, do I need it?
			gr_tru_ind = np.setdiff1d(gr_tru_ind, gr_tru_ign_ind)

			if not time_inds:
				self.T = data[:,0,:]
				self.agg_pow = data[:,agg_ind,:]
				self.gr_tru = data[:,gr_tru_ind,:]
			else:
				self.T = data[time_inds[0]:time_inds[1],0,:]
				self.agg_pow = data[time_inds[0]:time_inds[1],agg_ind,:]
				self.gr_tru = data[time_inds[0]:time_inds[1],gr_tru_ind,:]
			
		
					 
	
	def arrange_for_waveNILM(self,sample_size,
		time_inds = [], #start and end indices for this data
		app_inds = [], # Which appliances to select, choose [] to use real data and all of the available appliances us
		use_time = True, # Whether or not time data will be included in the data
		rescale_meta = True, # Whether or not meta data will be rescaled
		meter_max = 2**14, #rescaling factor so that powers are in the range [0,1]	
		effective_sample_size = 0): # effective sample size after taking into account ovelap to support "skip_out_of_receptive_field"

		'Arranging data to fit the input requirements of waveNILM'	
		
		epsilon = 1e-8 
		if not time_inds:
			time_inds = [0,len(self.T)]
				
		# Converting time to be time of day (cyclical every 24 hours) in days (values range between 0 and 1)
		T = (self.T[time_inds[0]:time_inds[1]] % (60*60*24))/60 /1440.0
		
		if not app_inds: # When appliances are manually given to the network
			app_inds = np.arange(self.gt_tru.shape[1])
		
		
		if self.noise_mode == 0:
		# noise_mode = 0, is a denoised scenario where we choose appliances and creat aggregate data by adding them up
			X0 = np.zeros((time_inds[1]-time_inds[0],self.agg_pow.shape[1]))
			for i in range(self.agg_pow.shape[1]):
				X0[:,i]  = self.gr_tru[time_inds[0]:time_inds[1],app_inds,i].sum(1)/meter_max[i]   
			Y = self.gr_tru[time_inds[0]:time_inds[1],app_inds,0]/meter_max[0]			   
		
		
		elif self.noise_mode == 2: #Modeled Noise
		# noise_mode = 2, is a modeled noise, where we give the aggregate noise measurement (all other appliances) 
		# to the network as ground truth for training.
			X0 = np.zeros((time_inds[1]-time_inds[0],self.agg_pow.shape[1]))
			for i in range(self.agg_pow.shape[1]):
				X0[:,i]  = self.agg_pow[time_inds[0]:time_inds[1],i].squeeze()/meter_max[i]
			Y0 = self.gr_tru[time_inds[0]:time_inds[1],app_inds,0]/meter_max[0]
			N = X0[:,0]-Y0.sum(1)			
			Y = np.append(Y0,N.reshape(len(N),1),axis = 1)
		
		elif self.noise_mode == 1:
		# noise_mode = 1 is the standard noisy scenario where the inputs are actual aggregate measurements
			X0 = np.zeros((time_inds[1]-time_inds[0],self.agg_pow.shape[1]))
			for i in range(self.agg_pow.shape[1]):		
				X0[:,i]  = self.agg_pow[time_inds[0]:time_inds[1],i,].squeeze()/meter_max[i]
			Y = self.gr_tru[time_inds[0]:time_inds[1],app_inds,0]/meter_max[0]
		
		else:
			raise ValueError('Invalid noise mode value, proper values are 0,1,2 only')					

		X = X0
		
		# Arrange data to samples X sample_size X features
		if effective_sample_size == 0: # with no overlap
			X = X[:X.shape[0]//sample_num*sample_num,:]
			X = X.reshape(sample_num,X.shape[0]//sample_num,X.shape[1])
		
		
			Y = Y[:Y.shape[0]//sample_num*sample_num,:]
			Y = Y.reshape(sample_num,Y.shape[0]//sample_num,Y.shape[1])
		
		# With overlap to account for  "skip_out_of_receptive_field"'s effect.
		else:
			
			sample_num = (X.shape[0]-sample_size)//(effective_sample_size) + 1
			x = np.zeros((sample_num,sample_size,X.shape[1]))
			y = np.zeros((sample_num,sample_size,Y.shape[1]))
			
			for i in range(sample_num):
				x[i,:,:] = X[i*effective_sample_size:i*effective_sample_size+sample_size,:]
				y[i,:,:] = Y[i*effective_sample_size:i*effective_sample_size+sample_size,:]
			X = x
			Y = y
			
		
		self.NILMin = X 
		self.NILMgt = Y  
		self.readyforNILM = {'sample_size':sample_size,
				'time_inds':time_inds,
				'app_inds':app_inds,
				'effective_sample_size': effective_sample_size}


	def copy(self):
		'Create a new copy of the current instance of NILMdata'
		new_instance = NILMdata()
		new_instance.T = self.T
		new_instance.agg_pow = self.agg_pow
		new_instance.gr_tru = self.gr_tru
		new_instance.meta  = self.meta
		new_instance.NILMin = self.NILMin
		new_instance.NILMgt = self.NILMgt
		new_instance.NILMhat = self.NILMhat
		new_instance.readyforNILM = self.readyforNILM
		

		return new_instance
