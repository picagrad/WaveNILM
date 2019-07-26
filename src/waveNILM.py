import time
import input_generator
from keras.models import Model, load_model
from keras.layers import *
from keras.callbacks import *
from keras import metrics
from keras import objectives
from keras.activations import *
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras import backend as K
import numpy as np
import datetime,os,pickle
from sacred import Experiment
import os
from sacred.observers import FileStorageObserver


''' 
Main script to run training and testing of WaveNILM as seen in "WaveNILM - A causal neural network for power disaggergation from the complex power signal"
Published in ICASSP 2019.

This script both trains and tests the non-autoregressive version of waveNILM, as was used in the paper
Using SACRED for experiment management, all parameters defined in set_params().

Usage, from terminal:
"Python waveNILM.py with {optimizer_name} {Experiment paramters}"


List of parameters (and default values):
	
	# Data parameters:
	data_len = 1440*720 # Length of data to be used for training and testing
	val_spl = 0.1  # split ration for validation, note that currently, validation is used for testing as well.
	input_path = '../data' # Location of data files
	data_source = 'AMPds2_In_P_Out_P.dat' # Name of default data source file
	agg_ind = [1] # Location of aggregate measurement in source file
	noise_mode = 0 # Noise mode: 0 - denoised, 1 - noisy, 2 - noisy with modeled noise
	noise_scale = 10 # Weight of noise for noise mode 2
	
	app_inds = [6,8,9,12,14] # Which appliances to disaggregate
	# app_inds = [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
	
	#Network Parameters
	nb_filters = [512,256,256,128,128,256,256,256,512] # Number of nodes per layer, use list for varying layer width.
	depth = 9 # Number of dilated convolutions per stack
	stacks = 1 # Number of dilated convolution stacks
	residual = False # Whether network is residual or not (requires uniform number of nodes per layer, for now)
	use_bias = True # Whether or not network uses bias
	activation = ReLU() # activation - use function handle
	callbacks = [LearningRateScheduler(scheduler, verbose = 1)]	 # Callbacks for training
	dropout = 0.1 # Dropout value
	mask = True # Whether to use masked output
	
	# Training Parameters:
	n_epochs = 300 # Number of training epochs
	batch_size = 50 # Number of samples per batch, each sample will have sample_size timesteps
	sample_size = 1440 # Number of timesteps (minutes for AMPds2) per sample
	savepath = '../data/comparison' # Folder to save models during/after training.
	save_flag = False  # Flag to save best model at each iteration of cross validation
	shuffle  = True # Shuffle samples every epoch
	verbose = 2 # Printing verbositiy, because of sacred, use only 0 or 2
	res_l2 = 0. # l2 penalty weight
	use_receptive_field_only = True # Whether or not to ignore the samples without full input receptive field
	loss = objectives.mae # Loss, use function handle
	all_metrics = [estimated_accuracy] #Metrics, use function handle
	optimizer = {} # Placeholder for optimizer
	cross_validate = True # Cross validation parameters
	splice = [0,1,2,3,4,5,6,7,8,9]
	use_metric_per_app = False # Whether to display metrics for each appliances separately
'''


ex = Experiment('Name')
ex.observers.append(FileStorageObserver.create('../data/sacred_runs'))

@ex.config
def base_config():
	# Data parameters:
	data_len = 1440*720 # Length of data to be used for training and testing
	val_spl = 0.1  # split ration for validation, note that currently, validation is used for testing as well.
	input_path = '../data' # Location of data files
	data_source = 'AMPds2_In_P_Out_P.dat' # Name of default data source file
	agg_ind = [1] # Location of aggregate measurement in source file
	noise_mode = 0 # Noise mode: 0 - denoised, 1 - noisy, 2 - noisy with modeled noise
	noise_scale = 10 # Weight of noise for noise mode 2
	
	app_inds = [6,8,9,12,14] # Which appliances to disaggregate
	# app_inds = [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
	
	#Network Parameters
	nb_filters = [512,256,256,128,128,256,256,256,512] # Number of nodes per layer, use list for varying layer width.
	depth = 9 # Number of dilated convolutions per stack
	stacks = 1 # Number of dilated convolution stacks
	residual = False # Whether network is residual or not (requires uniform number of nodes per layer, for now)
	use_bias = True # Whether or not network uses bias
	activation = ReLU() # activation - use function handle
	callbacks = [LearningRateScheduler(scheduler, verbose = 1)]	 # Callbacks for training
	dropout = 0.1 # Dropout value
	mask = True # Whether to use masked output
	
	# Training Parameters:
	n_epochs = 300 # Number of training epochs
	batch_size = 50 # Number of samples per batch, each sample will have sample_size timesteps
	sample_size = 1440 # Number of timesteps (minutes for AMPds2) per sample
	savepath = '../data/comparison' # Folder to save models during/after training.
	save_flag = False  # Flag to save best model at each iteration of cross validation
	shuffle  = True # Shuffle samples every epoch
	verbose = 2 # Printing verbositiy, because of sacred, use only 0 or 2
	res_l2 = 0. # l2 penalty weight
	use_receptive_field_only = True # Whether or not to ignore the samples without full input receptive field
	loss = objectives.mae # Loss, use function handle
	all_metrics = [estimated_accuracy] #Metrics, use function handle
	optimizer = {} # Placeholder for optimizer
	cross_validate = True # Cross validation parameters
	splice = [0,1,2,3,4,5,6,7,8,9]
	use_metric_per_app = False # Whether to display metrics for each appliances separately

@ex.named_config
def adam():
	optimizer = {
		'optimizer': 'adam',
		'params': 
		{'lr': 0.001,
		'beta_1':0.9,
		'beta_2':0.999,
		'decay': 0.,
		'amsgrad':False,
		'epsilon': 1e-8}
		
	}

@ex.named_config
def sgd():
	optimizer = {
		'optimizer': 'sgd',
		'params': {
		'lr': 0.01,
		'momentum': 0.9,
		'decay': 0.,
		'nesterov': True,
		'epsilon': None
		}
	}
		
def estimated_accuracy(y_true,y_pred):
	' NILM metric as described in paper'
	return 1 - K.sum(K.abs(y_pred-y_true))/(K.sum(y_true)+K.epsilon())/2
	 
def scheduler(epoch, curr_lr):
	' Learning rate scheduler '
	if epoch<  50:
		pass
	elif (epoch+1) % 10 == 0:
		curr_lr = curr_lr*0.98			
	return(curr_lr)

@ex.capture
def get_meter_max(input_path, data_source):
	' Find the "meter maximum value" - the closest power of 2 to the maximum aggregate measurement.'
	' This is like saying we set the bit-depth of our measurements so that the maximum power still fits'
	with open(input_path  + '/' + data_source,'rb') as f:
		data = pickle.load(f)
	
	meter_max = []
	for i in range(data.shape[-1]):
		meter_max.append(2** np.ceil(np.log2(data[:,1:,i].max())))
	return meter_max
	
	
		
def make_class_specific_loss(loss,cl_ind, name):
	'Wrap any loss or meteric so that it gives the loss only for one output dimension, meaning one sub-meter'
	
	def wrapper(y_true,y_pred):
		y_true = K.expand_dims(y_true[:,:,cl_ind],axis=2)
		y_pred = K.expand_dims(y_pred[:,:,cl_ind],  axis=2)
		
		return loss(y_true,y_pred)
	
	wrapper.__name__ = loss.__name__ + name
	
	return wrapper


@ex.capture
def get_class_weights(noise_mode,app_inds,noise_scale):
	'Give class weights so that if we are in noise mode 2  (see netdata.py for details)'
	' we still get balanced class weights. Otherwise - give equal weight to each class.'
	' Can be altered for unbalanced dataset to allow better training'
		
	if noise_mode==2:
		cw = np.ones((1,len(app_inds)))
		cw = np.append(cw,1/noise_scale)
		cw= cw/cw.sum()
		return cw
	
	return np.ones((1,len(app_inds)))/len(app_inds)
		
@ex.capture
def extract_and_duplicate(tensor,reps=1,batch_size=0,sample_size=0):
	'Copy input once for every class for multiplication.'
	'Conisder reimplementing to save memeory'
	
	tensor = K.reshape(tensor[:,:,0],(batch_size,sample_size,1))
	if reps>1:
		tensor = Concatenate()([tensor for i in range(reps)])
	return tensor

def load_waveNILM(config_path, model_path,print_flag):
	'Load a presaved waveNILM model'
	
	ex.add_config(config_path)
	inmodel = load_model(model_path, custom_objects={'kl':kl, 'estimated_accuracy':estimated_accuracy,'abs_error':abs_error,'target_power':target_power,'adj_estimated_accuracy':adj_estimated_accuracy,'adj_abs_error':adj_abs_error})
	if print_flag:
		inmodel.summary()
	return inmodel

@ex.command
def save_network_copy(out_path,in_path):
	'save a waveNILM copy'
	inmodel = load_model(in_path, custom_objects={'kl':kl, 'estimated_accuracy':estimated_accuracy,'abs_error':abs_error,'target_power':target_power,'adj_estimated_accuracy':adj_estimated_accuracy,'adj_abs_error':adj_abs_error})
	w_in = inmodel.get_weights()
	inmodel.summary()
	model = create_net()
	model.compile('adam','mae')
	model.summary()
	try: 
		model.set_weights(w_in)
	except:
		raise BaseException('Input model and output model don''t have same shape')
		
	model.save(out_path)

@ex.capture
def create_net(batch_size,depth,sample_size,app_inds,nb_filters,use_bias,res_l2,residual,stacks,activation,noise_mode,dropout,mask):
	
	#Create WaveNILM network
	
	meter_max = get_meter_max()
	
	# If constant amount of convolution kernels - create a list
	if len(nb_filters)==1:
		nb_filters = np.ones(depth,dtype='int')*nb_filters[0]		  
	
	# Input layer
	inpt = Input(batch_shape = (batch_size,sample_size,len(meter_max)))
	
	# Initial Feature mixing layer
	out = Conv1D(nb_filters[0], 1, padding='same',use_bias = use_bias,kernel_regularizer=l2(res_l2))(inpt) 
	
	skip_connections = [out]
	
	# Create main wavenet structure
	for j in range(stacks):
		for i in range(depth):
			# "Signal" output
			signal_out = Conv1D(nb_filters[i], 2, dilation_rate=2 ** i, padding='causal', 
				use_bias=use_bias, kernel_regularizer=l2(res_l2))(out)
			signal_out = activation(signal_out)
			
			# "Gate" output
			gate_out = Conv1D(nb_filters[i], 2, dilation_rate=2 ** i, padding='causal', 
				use_bias=use_bias,kernel_regularizer=l2(res_l2))(out)
			gate__out = sigmoid(gate_out)
			
			# Multiply signal by gate to get gated output
			gated = Multiply()([signal_out, gate_out])
			
			# Create residual if desired, note that currently this can only be supported for entire network at once
			# Consider changing residual to 2 lists  - split and mearge, and check for each layer individually
			if residual:
				# Making copies of previous layer nodes if the number of filters  doesn't match
				prev_ind = max(i-1,0)
				if not nb_filters[i] == nb_filters[prev_ind]:
					out = Lambda(extract_and_duplicate,arguments = {'reps':nb_filter[i]/nb_filter[prev_ind],'batch_size':batch_size,'sample_size':sample_size})(out)
				
				# Creating residual
				out = Add()([out,gated])
			else:
				out = gated

			# Droupout for regularization
			if dropout!=0:
				out = Dropout(dropout)(out)
			skip_connections.append(out)

	out = Concatenate()(skip_connections)
	
	# Masked output final layer
	if mask:
		# Create copies of desired input property (power, current, etc.) for multiplication with mask
		pre_mask = Lambda(extract_and_duplicate,arguments = {'reps':len(app_inds)+noise_mode//2,'batch_size':batch_size,'sample_size':sample_size})(inpt)
		# Create mask
		mask = TimeDistributed(Dense(len(app_inds)+noise_mode//2, activation = 'tanh'))(out)
		# Multiply with mask
		out = Multiply()([pre_mask,mask])

		## Optional residual mask instead of multiplicative mask
		# out = Add()([pre_mask,mask])
		# out = LeakyReLU(alpha = 0.1)(out)
	
	#Standard output final layer
	else:
		out = TimeDistributed(Dense(len(app_inds)+noise_mode//2,activation = 'linear'))(out)
		out = LeakyReLU(alpha = 0.1)(out)

	model = Model(inpt,out)
	
	return model


@ex.capture(prefix='optimizer')
def make_optimizer(optimizer, params):
	# Optimizer setup using sacred prefix
	if optimizer == 'sgd':
		optim = SGD(**params)
	elif optimizer == 'adam':
		optim = Adam(**params)
	else:
		raise ValueError('Invalid config for optimizer.optimizer: ' + optimizer)
	return optim
	


@ex.capture
def compile_model(model,use_receptive_field_only,loss,all_metrics,noise_mode, use_metric_per_app,app_inds):
	#Fix cost and metrics according to skip_out_of_receptive_field and compile model
	
	optim = make_optimizer()
	
	# Skipping any inputs that may contain zero padded inputs for loss calculation (and performance evaluation)
	if use_receptive_field_only:
		loss = skip_out_of_receptive_field(loss)
		all_metrics = [skip_out_of_receptive_field(m) for m in all_metrics]
	
	# creating specific copy of each metric for each appliance
	if use_metric_per_app and len(app_inds)>1:
		ln = len(all_metrics)
		for i in range(len(app_inds)):
			name = '_for_appliance_%d' % app_inds[i]
			for j in range(ln):
				all_metrics.append(make_class_specific_loss(all_metrics[j],i,name))
			
	model.compile(loss = loss, metrics = all_metrics, optimizer = optim)
	

@ex.capture(prefix='optimizer')
def reset_weights(model,lr):
	# Resetting weights of model for next iteration of cross validation
	session = K.get_session()
	for layer in model.layers: 
		if hasattr(layer, 'kernel_initializer'):
			layer.kernel.initializer.run(session=session)
	K.set_value(model.optimizer.lr, lr) #resetting learning rate (which is updated during training by optimizer)

@ex.capture
def data_splice(splice_ind, effective_sample_size,data_len, val_spl,sample_size,batch_size):
	
	# dividing data into training and validation set, note that validation set (which can be used as test set if needed)
	val_len = val_spl*((data_len-sample_size)//effective_sample_size)
	
	val_len = val_len//batch_size*batch_size
	val_start = int(np.floor(splice_ind*val_len))
	val_end = int(np.floor(splice_ind*val_len+val_len))
	val_ind = np.arange(val_start,val_end,dtype='int')
	trn_ind = np.delete(np.arange((data_len-sample_size)//effective_sample_size,dtype='int'),val_ind)
	
	return trn_ind,val_ind
	
@ex.capture
def compute_receptive_field(depth,stacks):
	return (stacks*2**depth)

@ex.capture
def skip_out_of_receptive_field(func):
	# Skipping any inputs that may contain zero padded inputs for loss calculation (and performance evaluation)
	receptive_field = compute_receptive_field()

	def wrapper(y_true, y_pred):
		y_true = y_true[:, receptive_field - 1:, :]
		y_pred = y_pred[:, receptive_field - 1:, :]
		return func(y_true, y_pred)

	wrapper.__name__ = func.__name__

	return wrapper

@ex.capture
def create_data_and_run(model,input_path,sample_size,app_inds,n_epochs,data_len,val_spl,savepath,batch_size, shuffle,save_flag,depth,verbose,callbacks,cross_validate,noise_mode,splice, data_source, agg_ind):
	
	# Preparing data folder according to input paramters
	effective_sample_size = int((sample_size - compute_receptive_field()))
	pth = input_path + ('/sample_size_%s_app_inds_%s_effective_sample_size_%s_noise_mode_%s_data_source_%s' %(sample_size,app_inds,effective_sample_size,noise_mode,data_source[:-4])).replace(',','_')
	print('Data source is %s' %(input_path +'/' + data_source))
	
	meter_max = get_meter_max()
	print('Normalization is: %s ' %meter_max)
	# If data folder doesn't exist, or is empty, create a new one
	if not os.path.isdir(pth):
		input_generator.make_data_folder(input_path,sample_size = sample_size, app_inds = app_inds, effective_sample_size = effective_sample_size,noise_mode = noise_mode, source = data_source, meter_max = meter_max, agg_ind = agg_ind)
	elif os.listdir(pth) == []:
		input_generator.make_data_folder(input_path,sample_size = sample_size, app_inds = app_inds, effective_sample_size = effective_sample_size,noise_mode = noise_mode, source = data_source, meter_max = meter_max, agg_ind = agg_ind)
	
	
	# Cross validated training
	if not cross_validate:
		splice = [0]
	# Collect time stamp to avoid overwriting previous models
	now = datetime.datetime.now().strftime('/%Y_%m_%d_%H%M')
	spth = savepath + now

	histories = []
	paths = []
	for i in splice:
		
		if i>0:
			reset_weights(model)
		# Create training and validation generators for current data splice
		trn_inds, val_inds = data_splice(i,effective_sample_size)
		trn_gen = input_generator.input_generator(batch_size = batch_size, out_dim = len(app_inds) + noise_mode//2 , feature_length = len(meter_max),sample_inds = trn_inds, path = pth, sample_size = sample_size, shuffle = shuffle)
		val_gen = input_generator.input_generator(batch_size = batch_size, out_dim = len(app_inds) + noise_mode//2, feature_length =  len(meter_max),sample_inds = val_inds, path = pth, sample_size = sample_size, shuffle = shuffle, test = True)

		# Creating time specific folder to avoid overwriting previous runs
		if save_flag:
			try:
				if not os.path.isdir(spth):
					os.mkdir(spth)
				wpth = spth +('/Data_splice_%d_model.hdf5' %i).replace('[]','None')
				
				# Add callback to save model with best validation accuracy
				callbacks.append(ModelCheckpoint(filepath=wpth , verbose=1, save_best_only=True, monitor = 'val_estimated_accuracy'))
			except:
				raise BaseException('unable to create directory %s\n for saving data, please check savepath is appropriate (all sub-directories exist, writing permissions etc.)' % spth)
		else:
			wpth = ''
		
		# Getting appliance weights (for use when modeling noise with lower importance or for unbalanced data)
		cw = get_class_weights()
		print('Appliance Weights:',cw)
			
		# Declare current appliance indices and callbacks for trianing
		print('Appliance indices %s data splice: %d' %(app_inds,i))
		print('Callbacks are: %s' %callbacks) 
		
		# Train model
		history = model.fit_generator(trn_gen,epochs = n_epochs, validation_data = val_gen, verbose = verbose , use_multiprocessing = True, callbacks =  callbacks)	
		
		# Keep loss and metric history for this splice of cross validation
		histories.append(history)
		paths.append(wpth)
		callbacks = callbacks[:-1]  
	return histories, spth
	


@ex.automain
def my_main(save_flag):
	#Initialize network and compile
	model =  create_net()
	model.summary()
	compile_model(model)
	
	# Train and test network
	history,spth = create_data_and_run(model)
	
	# Keep histories for each iteration of network
	histories = [h.history for h in history]
	
	# Save to directory if needed
	if save_flag:
		pth = spth +'/Loss_history.dat'
		file = open(pth,'wb')
		pickle.dump(histories,file)
		file.close()
  
