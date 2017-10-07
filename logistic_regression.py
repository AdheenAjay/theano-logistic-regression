# ref: http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/theano-tutorial/intro_theano/logistic_regression.ipynb

import os
import requests
import gzip
import six
from six.moves import cPickle

if not os.path.exists('mnist.pkl.gz'):
	r = requests.get('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
	with open('mnist.pkl.gz', 'wb') as data_file:
		data_file.write(r.content)

with gzip.open('mnist.pkl.gz', 'rb') as data_file:
	if six.PY3:
		train_set, valid_set, test_set = cPickle.load(data_file, encoding='latin1')
	else:
		train_set, valid_set, test_set = cPickle.load(data_file)

train_set_x, train_set_y = train_set
valid_set_x, valid_set_y = valid_set
test_set_x, test_set_y = test_set

####The model

import numpy
import theano
from theano import tensor

#length of each training vector
n_in = 28*28

#num of classes
n_out = 10

x = tensor.matrix('x')
W = theano.shared(value = numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
					name = 'W',
					borrow = True)
b = theano.shared(value = numpy.zeros((n_out,), dtype=theano.config.floatX),
					name = 'b',
					borrow = True)

#
p_y_given_x = tensor.nnet.softmax(tensor.dot(x, W) + b)
y_pred = tensor.argmax(p_y_given_x, axis=1)

#defining a loss function
y = tensor.lvector('y')
log_prob = tensor.log(p_y_given_x)
log_likelyhood = log_prob[tensor.arange(y.shape[0]) , y]
loss = - log_likelyhood.mean()


# training model
g_W, g_b = theano.grad(cost=loss, wrt=[W,b])

learning_rate = numpy.float32(0.13)
new_W = W - learning_rate * g_W
new_b = b - learning_rate * g_b

train_model = theano.function(inputs=[x,y], outputs=loss, updates=[(W, new_W), (b, new_b)])

#testing model
misclass_nb = tensor.neq(y_pred, y)
misclass_rate = misclass_nb.mean()
test_model = theano.function(inputs=[x,y], outputs=misclass_rate)

#training process
batch_size = 500
# number of minibatches for training, validating and testing
n_train_batches = train_set_x.shape[0]//batch_size
n_valid_batches = valid_set_x.shape[0]//batch_size
n_test_batches = test_set_x.shape[0]//batch_size

def get_miniBatch(i, dataset_x, dataset_y):
	start_idx = i*batch_size
	end_idx = (i+1) * batch_size
	batch_x = dataset_x[start_idx:end_idx]
	batch_y = dataset_y[start_idx:end_idx]
	return (batch_x, batch_y)

##early stop parameters
#max number of epochs 
n_epochs = 10
#wait time
patience = 500
#wait this much longer when a new best is found
patience_increase = 2
# a relative improvement of this much is considered significant
improvement_threshold = 0.995


# go through this many minibatches before checking the network on the validation set;
# in this case we check every epoch
validation_frequency = min(n_train_batches, patience/2)

import timeit
from six.moves import xrange

best_validation_loss = numpy.inf 
test_score = 0.
start_time = timeit.default_timer()

done_looping = False
epoch = 0

while(epoch < n_epochs) and (not done_looping) :
	epoch = epoch + 1
	for minibatch_index in xrange(n_train_batches):
		minibatch_x, minibatch_y = get_miniBatch(minibatch_index, train_set_x, train_set_y)
		minibatche_avg_cost = train_model(minibatch_x, minibatch_y)

		#iteration number 
		iter = (epoch - 1) * n_train_batches + minibatch_index
		if (iter + 1) % validation_frequency == 0:
			#compute zro-onve loss  on validation set
			validation_losses = []
			for i in xrange(n_valid_batches):
				valid_xi, valid_yi = get_miniBatch(i, valid_set_x, valid_set_y)
				validation_losses.append(test_model(valid_xi, valid_yi))
			this_validation_loss = numpy.mean(validation_losses)
			print('epoch %i, minibatch %i/%i, validation error %f %%' %
				(epoch, minibatch_index + 1, n_train_batches, this_validation_loss*100.))

			#if we got the best validation score till now
			if this_validation_loss < best_validation_loss:
				if this_validation_loss < best_validation_loss * improvement_threshold:
					patience = max(patience, iter*patience_increase)

				best_validation_loss = this_validation_loss
				
				test_losses = []
				for i in xrange(n_test_batches):
					test_xi, test_yi = get_miniBatch(i, train_set_x, train_set_y)
					test_losses.append(test_model(test_xi, test_yi))

				test_score = numpy.mean(test_losses)
				print(' epoch %i, minibatch %i/%i, test error of best model %f %%' %
					(epoch, minibatch_index+1, n_train_batches, test_score*100.))

				#save the best model
				numpy.savez('best_model.npz', W=W.get_value(), b=b.get_value())

		if patience <= iter:
			done_looping = True
			break

end_time = timeit.default_timer()

print('Optimization complete with best validation score of %f %% , '
	'with test perfromance %f %%' %(best_validation_loss*100., test_score*100.))

print('The code ran for %d epochs with %f epochs/sec' %
	(epoch, 1.*epoch/(end_time - start_time)))


#visualization
import matplotlib.pyplot as plt
from utils import tile_raster_images
import pylab

plt.clf()

#increase the size of the figure
plt.gcf().set_size_inches(15, 10)

plot_data = tile_raster_images(W.get_value(borrow=True).T, img_shape=(28,28),
	tile_shape=(2,5), tile_spacing=(1,1))

plt.imshow(plot_data, cmap='Greys', interpolation='none')
plt.axis('off')

plt.show()






