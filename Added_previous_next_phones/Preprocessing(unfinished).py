import gzip
import cPickle
import numpy as np
import os
import os.path
import sys
import theano
import theano.tensor as T
import time
import matplotlib.pylab as plt
from scipy.io import wavfile
import scipy.stats

# DATASET

def onehot(x,max):
	onehot = np.zeros(max,dtype=theano.config.floatX)
	onehot[x] = 1
	return onehot
	
def previous(list, index):
	
	previous_index = index
	value = list[index]
	previous_value = value
	
	while ((previous_value == value) and (previous_index >= 1)):
		previous_index -= 1
		previous_value = list[previous_index]
	
	return previous_value, index-previous_index
	
def next(list, index):
	
	next_index = index
	value = list[index]
	next_value = value
	
	while ((next_value == value) and (next_index < len(list)-1)):
		next_index += 1
		next_value = list[next_index]
	
	return next_value, next_index -index
	

# TODO: if I want to work with GPU, I have to put the dataset into shared variables
class TIMIT(object):

	# Mean and standard deviation of the acoustic samples from the whole
	# dataset (train, valid, test).
	_mean = 0.0035805809921434142
	_std = 542.48824133746177

	def __init__(self, which_set, n_samples_in, n_samples_out, batch_size):
				 
		self.n_samples_in = n_samples_in
		self.n_samples_out = n_samples_out
		self.batch_size = batch_size

		# Load data from disk
		self._load_data(which_set)

		# Standardize data
		for i, sequence in enumerate(self.raw_wav):
			self.raw_wav[i] = (sequence - TIMIT._mean) / TIMIT._std

		# get info on the set
		self.num_phones = np.max([np.max(sequence) for sequence in self.phones]) + 1
		self.num_phonemes = np.max([np.max(sequence) for sequence in self.phonemes]) + 1
		self.num_words = np.max([np.max(sequence) for sequence in self.words]) + 1
		self.num_speakers = 630

	def _load_data(self, which_set):

		# Create file paths
		timit_base_path = os.path.join(os.environ["PYLEARN2_DATA_PATH"], "timit/readable")
		raw_wav_path = os.path.join(timit_base_path, which_set + "_x_raw.npy")
		phonemes_path = os.path.join(timit_base_path, which_set + "_x_phonemes.npy")
		phones_path = os.path.join(timit_base_path, which_set + "_x_phones.npy")
		words_path = os.path.join(timit_base_path, which_set + "_x_words.npy")
		speaker_path = os.path.join(timit_base_path, which_set + "_spkr.npy")

		# Load data        
		self.raw_wav = np.load(raw_wav_path)
		self.phonemes = np.load(phonemes_path)
		self.phones = np.load(phones_path)
		self.words = np.load(words_path)
		self.speaker_id = np.asarray(np.load(speaker_path), 'int')

		# # pre-compute previous_phones and samples_to_previous_phone
		# self.previous_phone = np.load(phones_path)
		# self.samples_to_previous_phone = np.load(phones_path)
		# for i in xrange(len(self.phones)):
			# for j in xrange(len(self.phones[i])):
				# self.previous_phone[i][j], self.samples_to_previous_phone[i][j] = previous(self.phones[i],j)
		
		# pre-compute next_phones and samples_to_next_phone
		self.next_phone = np.load(phones_path)
		self.samples_to_next_phone = np.load(phones_path)
		
		for i in xrange(len(self.phones)):
			j = 0
			while j < len(self.phones[i])-1:
				
				#print len(self.phones[i])
				
				self.next_phone[i][j], self.samples_to_next_phone[i][j] = next(self.phones[i],j)
				
				# print j
				# print self.phones[i][j]
				# print self.next_phone[i][j]
				# print self.samples_to_next_phone[i][j]
				
				for k in range(1,self.samples_to_next_phone[i][j]) :
				
					self.next_phone[i][j+k] = self.next_phone[i][j]
					self.samples_to_next_phone[i][j+k] = self.samples_to_next_phone[i][j+k-1]-1
				
				j+= self.samples_to_next_phone[i][j]
				
				# print j
				# print self.phones[i][j]
				# print self.next_phone[i][j]
				# print self.samples_to_next_phone[i][j]
				
				# raw_input("Press enter to continue")
				
				# print str(j+1) + '/' + str(len(self.phones[i]))
			print str(i+1) + '/' + str(len(self.phones))
			#print self.phones[i]
			#print self.next_phone[i]
			
		# Open the file and overwrite current contents
		save_file = open(which_set+'_next_phone.npy', 'wb')
		# save the file
		np.save(save_file, self.next_phone)
		# close the file
		save_file.close()
		
		# Open the file and overwrite current contents
		save_file = open(which_set+'_samples_to_next_phone.npy', 'wb')
		# save the file
		np.save(save_file, self.samples_to_next_phone)
		# close the file
		save_file.close()
		
	def get_batch(self):
		
		batch_x = np.zeros((self.batch_size, self.n_samples_in + 3 * self.num_phones + 2), dtype=theano.config.floatX)
		batch_y = np.zeros((self.batch_size, self.n_samples_out), dtype=theano.config.floatX)
		
		# for each element of the batch, we take a random part of a random sentence
		for k in xrange(self.batch_size):
		
			# choose a random sentence
			sentence_id = np.random.random_integers(len(self.raw_wav) - 1)
			
			# choose a random start acoustic sample
			start = np.random.random_integers(len(self.raw_wav[sentence_id]) - self.n_samples_in - self.n_samples_in - 1)
			
			# get the previous samples
			batch_x[k][0 : self.n_samples_in] = self.raw_wav[sentence_id][start : start + self.n_samples_in]
			
			# get the current phone
			batch_x[k][self.n_samples_in:self.n_samples_in+self.num_phones]=onehot(self.phones[sentence_id][start+self.n_samples_in], self.num_phones)
			
			# get the previous phone and the number of samples after previous phone
			batch_x[k][self.n_samples_in+self.num_phones:self.n_samples_in+2*self.num_phones]=onehot(self.previous_phone[sentence_id][start+self.n_samples_in],self.num_phones)
			batch_x[k][self.n_samples_in+2*self.num_phones] = self.samples_to_previous_phone[sentence_id][start+self.n_samples_in]/2000.
			
			# get the next phone and the number of samples before next phone
			batch_x[k][self.n_samples_in+2*self.num_phones+1:self.n_samples_in+3*self.num_phones+1]=onehot(self.next_phone[sentence_id][start+self.n_samples_in],self.num_phones)
			batch_x[k][self.n_samples_in+3*self.num_phones+1] = self.samples_to_next_phone[sentence_id][start+self.n_samples_in]/2000.

			# get the next samples
			batch_y[k] = self.raw_wav[sentence_id][start + self.n_samples_in:start + self.n_samples_in + self.n_samples_out]
			
			# print "previous"
			# print previous_phone
			# print samples_to_previous_phone
			# print "next"
			# print next_phone
			# print samples_to_next_phone
			# raw_input("Press enter to continue")
		
		# print "batch x"
		# print batch_x[10][self.n_samples_in-5:self.n_samples_in]
		# print batch_x[10][self.n_samples_in:self.n_samples_in + 5]
		# print "batch y"
		# print batch_y[10]
			
		return batch_x, batch_y
	
# MODEL
	
class HiddenLayer(object):

	def __init__(self, n_in, n_out):
	
		# initial values of parameters
		rng = np.random.RandomState(1234)
		low=-np.sqrt(6. / (n_in + n_out))
		high=np.sqrt(6. / (n_in + n_out))
		size=(n_in, n_out)
		W_values = np.asarray(rng.uniform(low=low,high=high,size=size),dtype=theano.config.floatX)
		b_values = np.zeros(n_out, dtype=theano.config.floatX)
		b_values = b_values + 1.
			
		# creation of shared symbolic variables
		# shared variables are the state of the built function
		# in practice, we put them in the GPU memory
		self.W = theano.shared(value=W_values, name='W', borrow=True)
		self.b = theano.shared(value=b_values, name='b', borrow=True)
		self.params = [self.W, self.b]
		
	# output
	def fprop(self, input):
		return T.maximum(0,T.dot(input, self.W) + self.b)
		
class OutputLayer(object):

	def __init__(self, n_in, n_out):
	
		# initial values of parameters
		W_values = np.zeros((n_in, n_out), dtype=theano.config.floatX)
		b_values = np.zeros(n_out, dtype=theano.config.floatX)
			
		# creation of shared symbolic variables
		self.W = theano.shared(value=W_values, name='W', borrow=True)
		self.b = theano.shared(value=b_values, name='b', borrow=True)
		self.params = [self.W, self.b]
		
	# output
	def fprop(self, input):
		return T.dot(input, self.W) + self.b
		
class MLP(object):

	layer = []

	def __init__(self, n_input, n_output, n_hidden, n_hidden_layers):

		self.n_hidden_layers = n_hidden_layers
		
		# Create MLP layers	
		if n_hidden_layers == 0 :
			self.layer.append(OutputLayer(n_in=n_input, n_out=n_output))

		else :
			self.layer.append(HiddenLayer(n_in=n_input, n_out=n_hidden))

			for k in range(1,n_hidden_layers):
				self.layer.append(HiddenLayer(n_in=n_hidden, n_out=n_hidden))

			self.layer.append(OutputLayer(n_in=n_hidden, n_out=n_output))

		# MLP parameters
		self.params = self.layer[0].params
		
		for k in range(1,n_hidden_layers+1):
			self.params = self.params + self.layer[k].params
			
	# declare output
	def fprop(self, input):
		output = self.layer[0].fprop(input)
		for k in range(1,self.n_hidden_layers+1):
			output = self.layer[k].fprop(output)
		return output
			
	def save_params(self, path):        
		
		# Open the file and overwrite current contents
		save_file = open(path, 'wb')
		
		# write all the parameters in the file
		for k in xrange(self.n_hidden_layers+1):
			cPickle.dump(self.layer[k].W.get_value(borrow=True), save_file, -1)
			cPickle.dump(self.layer[k].b.get_value(borrow=True), save_file, -1)
		
		# close the file
		save_file.close()
		
	def load_params(self, path): 
		
		# Open the file
		save_file = open(path)
		
		# read an load all the parameters
		for k in xrange(self.n_hidden_layers+1):
			self.layer[k].W.set_value(cPickle.load(save_file), borrow=True)
			self.layer[k].b.set_value(cPickle.load(save_file), borrow=True)

		# close the file
		save_file.close()
			
# TRAINING

def train(learning_rate, batch_size, n_train_batch, n_valid_batch, n_test_batch, n_samples_in, n_samples_out, n_hidden, n_hidden_layers, load = None):

	print '... loading the dataset'
	
	# create and load the 3 sets
	train_set = TIMIT(which_set='train', n_samples_in=n_samples_in, n_samples_out=n_samples_out,  batch_size= batch_size)
	valid_set = TIMIT(which_set='valid', n_samples_in=n_samples_in, n_samples_out=n_samples_out,  batch_size= batch_size)
	test_set = TIMIT(which_set='test', n_samples_in=n_samples_in, n_samples_out=n_samples_out,  batch_size= batch_size)

	print '... creating the model'
	
	# input and output variables
	x = T.matrix('x')
	y = T.matrix('y')
	
	# creation of model
	model = MLP(n_input = n_samples_in + 3*train_set.num_phones + 2, n_output = n_samples_out, n_hidden = n_hidden, n_hidden_layers = n_hidden_layers)	
	
	# load previous parameters
	if load != None :
		model.load_params(load)
	
	# cost function = MSE
	cost = 0.5*T.mean((model.fprop(x) - y)**2)
	
	print '... computing the gradients'
	
	# compute gradients
	gparams = []
	for param in model.params:
		gparam = T.grad(cost, param)
		gparams.append(gparam)
		
	# compute updates
	updates = []
	for param, gparam in zip(model.params, gparams):
		updates.append((param, param - learning_rate * gparam))
	
	print '... building the model'
	
	# before the build, you work with symbolic variables
	# after the build, you work with numeric variables
	train_model = theano.function(inputs=[x,y], outputs=cost, updates=updates)
	test_model = theano.function(inputs=[x,y],outputs=cost)
		        
	print '... training the model'
	
	# initialize variables
	start_time = time.clock()
	epoch = 1
	stop = 0
	train_loss = 0
	validation_loss = 0
	test_loss = 0
	best_validation_loss = 1.
	
	while stop < 10 :

		# train the model on all training examples
		train_loss = 0
		for i in xrange(n_train_batch) : 
			batch_x, batch_y = train_set.get_batch()
			train_loss += np.mean(train_model(batch_x, batch_y))
		train_loss /= n_train_batch
		print('epoch %i, train MSE %f' %(epoch, train_loss))
		
		# test it on the validation set
		validation_loss = 0
		for i in xrange(n_valid_batch) :
			batch_x, batch_y = valid_set.get_batch()
			validation_loss += np.mean(test_model(batch_x,batch_y))
		validation_loss /= n_valid_batch
		print('epoch %i, validation MSE %f' %(epoch, validation_loss))
		
		# if validation loss get worse
		if validation_loss >= best_validation_loss : 
			stop = stop + 1
		
		# if validation loss get better
		else :
			best_validation_loss = validation_loss
			stop = 0
			
			# test it on the test set
			test_loss = 0
			for i in xrange(n_test_batch) :
				batch_x, batch_y = test_set.get_batch()
				test_loss += np.mean(test_model(batch_x,batch_y))
			test_loss /= n_test_batch
			print('epoch %i, test MSE %f' %(epoch, test_loss))
			
			# save the best parameters
			model.save_params('/u/courbarm/best_params_mlp_300_1_epoch_'+str(epoch)+'.pkl')
		
		epoch = epoch + 1
	
	end_time = time.clock()
	print('best validation MSE %f' %(best_validation_loss))
	print('test MSE %f'%(test_loss))
	print('the code ran for %i seconds'%(end_time - start_time))
	
# MAIN

if __name__ == "__main__":

	#train(learning_rate=0.002, batch_size = 32, n_train_batch = 20000, n_valid_batch = 2000, n_test_batch = 2000,
	#		n_samples_in = 160, n_samples_out = 1, n_hidden = 300, n_hidden_layers = 1)#, load = 'best_params_mlp_800_1_epoch_6.pkl')
	
	test_set = TIMIT(which_set='test', n_samples_in=160, n_samples_out=1,  batch_size= 32)
	