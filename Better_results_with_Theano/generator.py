import cPickle
import numpy as np
import theano
from scipy.io import wavfile
from random import randint
from MLP_TIMIT_SGD import MLP, TIMIT
import matplotlib.pylab as plt

# parameters
_mean = 0.0035805809921434142
_std = 542.48824133746177
input_size = 160
output_size = 1
fs = 16000
sentence_id = 0

print "load and build the model ..."

# create the model
model = MLP(n_input = input_size + 62, n_output = output_size, n_hidden = 300, n_hidden_layers = 1)

# load the parameters
# print model.layer[1].W.get_value(borrow=True)
model.load_params("best_params_epoch_33.pkl") # loading changes the weights
# print model.layer[1].W.get_value(borrow=True)

# build the model
x = theano.tensor.dmatrix('x')
predict = theano.function(inputs=[x], outputs=model.fprop(x))

print "load the phones from the test set ..."

# load the test set
test_set = TIMIT(which_set='test', n_samples_in=input_size, n_samples_out=output_size,  batch_size= 32)

# batch_x, batch_y = test_set.get_batch()
# print batch_x
# y = predict(batch_x)
# print y

# get the phones
phones = test_set.phones[sentence_id]

# convert phones to one hot vectors
phones_onehot = np.asmatrix(np.zeros((len(phones),test_set.num_phones)))
for i, pi in enumerate(phones):
	phones_onehot[i,pi] = 1

# print "Plot and wav original sentence ..."

# sentence = test_set.raw_wav[sentence_id]
# sentence = (sentence*_std) + _mean
# sentence = np.asarray((sentence/max(abs(sentence)))*2**15, dtype=np.int16)
# plt.plot(sentence)
# plt.ylabel('original sentence')
# plt.show()
# wavfile.write("original.wav", fs, sentence)

print "Synthesize speech ..."
	
# initialize x0 to 0
x0 = np.asmatrix(np.zeros((1,len(phones))))

# start with train example
# sentence = test_set.raw_wav[sentence_id]
# x0[0,0:input_size] = sentence[6500:6500+input_size] # remove 6500 : only here for debug purpose
# print x0[0,0:input_size]

# predict the rest of the sentence
input = np.asmatrix(np.zeros((1,input_size+62)))
i = input_size
while i<len(phones)-input_size:
	input[0,0:input_size] = x0[0,i-input_size:i]
	input[0,input_size:input_size+62] = phones_onehot[i+input_size]
	output = predict(input)
	# print output
	# raw_input("Press enter to continue")
	x0[0,i:output_size+i] = output
	# x0[0,i:output_size+i] = np.random.normal(output,0.02)
	i += output_size
x0 = (x0.T*_std) + _mean

print "Plot and wav generated sentence..."

# get image and sound
x0a = np.asarray((x0/max(abs(x0)))*2**15, dtype=np.int16)
plt.plot(x0a)
plt.ylabel('generated sentence')
plt.show()
wavfile.write("generated.wav", fs, x0a)