import cPickle as pickle
import numpy as np
import theano
from scipy.io import wavfile
from random import randint
import sys
sys.path.append('research/code/pylearn2/datasets/')
import timit
import matplotlib.pylab as plt

# parameters
_mean = 0.0035805809921434142
_std = 542.48824133746177
input_size = 160
output_size = 1
fs = 16000
sentence_id = 5
	
# load and build the model
mdl = pickle.load(open("mlp_62+160_300_300_1.pkl"))
X = theano.tensor.dmatrix('X')
P = theano.tensor.dmatrix('P')
y = mdl.fprop((X,P))
predict = theano.function([X, P], y)

# load the test set
dataset = timit.TIMIT(which_set='test', frame_length=input_size)
dataset._load_data(which_set='test')

# get the phones
phones = dataset.phones[sentence_id]

# convert phones to one hot vectors
phones_onehot = np.asmatrix(np.zeros((len(phones),62)))
for i, pi in enumerate(phones):
	phones_onehot[i,pi] = 1
	
# get the raw data 
sentence = dataset.raw_wav[sentence_id]

# get image and sound
plt.plot(sentence)
plt.ylabel('original sentence')
plt.show()
wavfile.write("original.wav", fs, sentence)

# start with 0
x0 = np.asmatrix(np.zeros((1,len(phones))))

# start with the raw data
# x0[0,0:input_size] = (sentence[0:input_size] - _mean) / _std

# predict the rest of the sentence
i = input_size
while i<len(phones) :
	x0[0,i:output_size+i] = np.random.normal(predict(x0[0,i-input_size:i], phones_onehot[i]), 1.)
	# x0[0,i:output_size+i] = predict(x0[0,i-input_size:i], phones_onehot[i])
	i += output_size
x0 = (x0.T*_std) + _mean

# print "160 a 210"
# print x0[0,160:210]
# print "8000 a 8050"
# print x0[0,8000:8050]
# i = input_size
# frame = np.asmatrix(np.zeros((1,input_size)))
# while i<len(sentence) :
	# frame[0] = (sentence[i-input_size:i] - _mean) / _std
	# x0[0,i:output_size+i] = predict(frame, phones_onehot[i])
	# i += output_size
# x0 = (x0.T*_std) + _mean
# print "160 a 210"
# print x0[0,160:210]
# print "8000 a 8050"
# print x0[0,8000:8050]

# get image and sound
x0a = np.asarray((x0/max(abs(x0)))*2**15, dtype=np.int16)
plt.plot(x0a)
plt.ylabel('generated sentence')
plt.show()
wavfile.write("generated.wav", fs, x0a)












