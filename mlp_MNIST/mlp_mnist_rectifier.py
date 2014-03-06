
import cPickle
import gzip
import os
import sys
import numpy
import theano
import theano.tensor as T

def onehot(x,numclasses=None):

    if x.shape==():
        x = x[None]
    if numclasses is None:
        numclasses = x.max() + 1
    result = numpy.zeros(list(x.shape) + [numclasses], dtype="int")
    z = numpy.zeros(x.shape)
    for c in range(numclasses):
        z *= 0
        z[numpy.where(x==c)] = 1
        result[...,c] += z
    return result	

def shared_dataset(data_xy, borrow=True):

	data_x, data_y = data_xy
	data_y = onehot(data_y)
	shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
	shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
	return shared_x, shared_y
	
def load_data(dataset):

	f = gzip.open(dataset, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()

	test_set_x, test_set_y = shared_dataset(test_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	train_set_x, train_set_y = shared_dataset(train_set)
	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)]
	return rval
	
class HiddenLayer(object):

	def __init__(self,rng,input, n_in,n_out):
				
		# initialisation of parameters	
		low=-numpy.sqrt(6. / (n_in + n_out))
		high=numpy.sqrt(6. / (n_in + n_out))
		size=(n_in, n_out)
		W_values = numpy.asarray(rng.uniform(low=low,high=high,size=size),dtype=theano.config.floatX)
		b_values = numpy.zeros(n_out, dtype=theano.config.floatX)
	
		# creation of shared symbolic variables
		self.W = theano.shared(value=W_values, name='W', borrow=True)
		self.b = theano.shared(value=b_values, name='b', borrow=True)
		self.params = [self.W, self.b]	
		
		# forward propagate symbolic expression
		self.output = T.maximum(T.dot(input, self.W) + self.b,0)	

class OutputLayer(object):

	def __init__(self,input,n_in,n_out):
				
		# initialisation of parameters
		W_values = numpy.zeros((n_in, n_out), dtype=theano.config.floatX)
		W_values = W_values + 0.000001
		b_values = numpy.zeros(n_out, dtype=theano.config.floatX)
		b_values = b_values + 0.000001
	
		# creation of shared symbolic variables
		self.W = theano.shared(value=W_values, name='W', borrow=True)
		self.b = theano.shared(value=b_values, name='b', borrow=True)
		self.params = [self.W, self.b]	
		
		# forward propagate symbolic expression
		self.output = T.maximum(T.dot(input, self.W) + self.b,0)
		
class MLP(object):
			
	def __init__(self, rng, input, target, n_in, n_h1, n_h2, n_out):
		
		self.h1 = HiddenLayer(rng=rng,input=input,n_in=n_in, n_out=n_h1)
		self.h2 = HiddenLayer(rng=rng,input=self.h1.output,n_in=n_h1, n_out=n_h2)
		self.outputLayer = OutputLayer(input=self.h2.output,n_in=n_h2,n_out=n_out)
		self.params = self.h1.params + self.h2.params + self.outputLayer.params
		self.errors = T.mean(T.neq(T.argmax(self.outputLayer.output, axis=1), T.argmax(target)))		
		self.mse = 0.5 * T.mean((self.outputLayer.output - target) ** 2)
		
def mlp_training(dataset='mnist.pkl.gz',learning_rate=0.1, n_h1 = 10, n_h2 = 10,n_epochs=10):

	print '... loading data'
	
	datasets = load_data(dataset)
	
	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]
		
	n_train = train_set_x.get_value(borrow=True).shape[0]
	n_valid = valid_set_x.get_value(borrow=True).shape[0]
	n_test = test_set_x.get_value(borrow=True).shape[0]
		
	print '... building the model'
		
	index = T.lscalar()  
	x = T.matrix('x')
	y = T.matrix('y')
	
	rng = numpy.random.RandomState(1234)
	classifier = MLP(rng=rng, input=x, target=y, n_in=28 * 28, n_h1= n_h1, n_h2= n_h2, n_out=10)
	cost = classifier.mse
	
	test_model = theano.function(inputs=[index], outputs=classifier.errors, givens={
				x: test_set_x[index:index+1],
		        y: test_set_y[index:index+1]})

	validate_model = theano.function(inputs=[index],outputs=classifier.errors, givens={
		        x: valid_set_x[index:index+1],
		        y: valid_set_y[index:index+1]})
		        
	gparams = []
	for param in classifier.params:
		gparam = T.grad(cost, param)
		gparams.append(gparam)
		
	updates = []
	for param, gparam in zip(classifier.params, gparams):
		updates.append((param, param - learning_rate * gparam))
	
	train_model = theano.function(inputs=[index], outputs=cost, updates=updates, givens={
		        x: train_set_x[index:index+1],
		        y: train_set_y[index:index+1]})
		        
	print '... training the model'

	for epoch in xrange(n_epochs):

		# train the model on all training examples
		for i in xrange(n_train) : train_model(i)
		
		# test it on the validation set
		validation_losses = [validate_model(i) for i in xrange(n_valid)]
		validation_loss = numpy.mean(validation_losses)
		print('epoch %i, validation error %f %%' %(epoch, validation_loss * 100.))

	# test it on the test set
	test_losses = [test_model(i) for i in xrange(n_test)]
	test_loss = numpy.mean(test_losses)
	print('epoch %i, test error %f %%' %(epoch, test_loss * 100.))
      
if __name__ == '__main__':
	mlp_training()







	
