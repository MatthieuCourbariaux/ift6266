import cPickle
import numpy as np
import theano
from scipy.io import wavfile
from random import randint
from MLP_TIMIT_SGD import MLP, TIMIT, onehot, previous, next
import matplotlib.pylab as plt

# parameters
_mean = 0.0035805809921434142
_std = 542.48824133746177
n_samples_in = 160
n_samples_out = 1
n_hidden = 800
n_hidden_layers = 2
parameters_path = "best_params_mlp_800_800_1_epoch_34.pkl"
fs = 16000
sentence_id = 0

print "load the phones from the test set ..."

# load the test set
test_set = TIMIT(which_set='test', n_samples_in=n_samples_in, n_samples_out=n_samples_out,  batch_size= 32)

# batch_x, batch_y = test_set.get_batch()
# print batch_x
# y = predict(batch_x)
# print y

# get the phones
phones = test_set.phones[sentence_id]

# print "Plot and wav original sentence ..."

# sentence = test_set.raw_wav[sentence_id]
# sentence = (sentence*_std) + _mean
# sentence = np.asarray((sentence/max(abs(sentence)))*2**15, dtype=np.int16)
# plt.plot(sentence)
# plt.ylabel('original sentence')
# plt.show()
# wavfile.write("original.wav", fs, sentence)

print "load and build the model ..."

# create the model
model = MLP(n_input = n_samples_in + 3*test_set.num_phones + 2, n_output = n_samples_out, n_hidden = n_hidden, n_hidden_layers = n_hidden_layers)

# load the parameters
# print model.layer[1].W.get_value(borrow=True)
model.load_params(parameters_path) # loading changes the weights
# print model.layer[1].W.get_value(borrow=True)

# build the model
x = theano.tensor.dmatrix('x')
predict = theano.function(inputs=[x], outputs=model.fprop(x))

print "Synthesize speech ..."
	
# initialize x0 to 0
x0 = np.asmatrix(np.zeros((1,len(phones))))

# start with train example
# sentence = test_set.raw_wav[sentence_id]
# x0[0,0:input_size] = sentence[6500:6500+input_size] # remove 6500 : only here for debug purpose
# print x0[0,0:input_size]

# predict the rest of the sentence
input = np.asmatrix(np.zeros((1,n_samples_in+test_set.num_phones*3+2)))
i = n_samples_in

while i<len(phones)-n_samples_in:

	# get the previous samples
	input[0,0:n_samples_in] = x0[0,i-n_samples_in:i]
	
	# get the current phone
	input[0,n_samples_in:n_samples_in+test_set.num_phones] = onehot(phones[i+n_samples_in], test_set.num_phones)
	
	# get the previous phone and the number of samples after previous phone
	previous_phone, samples_to_previous_phone = previous(phones, i+n_samples_in)
	input[0,n_samples_in+test_set.num_phones:n_samples_in+2*test_set.num_phones]=onehot(previous_phone,test_set.num_phones)
	input[0,n_samples_in+2*test_set.num_phones] = samples_to_previous_phone/2000.
	
	# get the next phone and the number of samples before next phone
	next_phone, samples_to_next_phone = next(phones, i+n_samples_in)
	input[0,n_samples_in+test_set.num_phones*2+1:n_samples_in+3*test_set.num_phones+1]=onehot(next_phone,test_set.num_phones)
	input[0,n_samples_in+2*test_set.num_phones+1] = samples_to_next_phone/2000.
	
	# Use the MLP
	output = predict(input)
	# print output
	# raw_input("Press enter to continue")
	x0[0,i:n_samples_out+i] = output
	# x0[0,i:n_samples_out+i] = np.random.normal(output, 0.0001)
	i += n_samples_out
	
x0 = (x0.T*_std) + _mean

print "Plot and save the generated sentence..."

# get image and sound
x0a = np.asarray((x0/max(abs(x0)))*2**15, dtype=np.int16)
plt.plot(x0a)
plt.ylabel('generated sentence')
plt.show()
wavfile.write("generated.wav", fs, x0a)