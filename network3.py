"""network3.py
~~~~~~~~~~~~~~

A Theano-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.

Because the code is based on Theano, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.

This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).

"""

#### Libraries
# Standard library
import cPickle
import pickle
import gzip
import sys
import os.path

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error #have to sqrt after

import timeit

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


#### Constants
GPU = False
if GPU:
    print "Trying to run under a GPU.  If this is not desired, then modify "+\
        "network3.py\nto set the GPU flag to False."
    try: theano.config.device = 'gpu'   
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU.  If this is not desired, then the modify "+\
        "network3.py to set\nthe GPU flag to True."

theano.config.optimizer = 'None'
net_metadata = [] # we want to store all our weights and permunations here so we can restart where we left off, just a list of arrays

#### Load the MNIST data
def load_data_shared(filename, perm = None):

    f = open('data/' + filename + '.pckl')
    # f = open('data/1458partial60k.pckl')
    training_data, validation_data, test_data = pickle.load(f)
    f.close()
    # return [training_data, validation_data, test_data]

    if perm != None:
        permutation = perm
    else:
        permutation = np.random.permutation(len(training_data[0][0]))
    # permuation = [1,  2,  4,  0,  8,  16, 3,  4,  6,  7,  9,  23, 10, 11, 14, 17, 18, 21, 12, 13, 15, 19, 20, 22]
    print 'current permuation:', permutation
    net_metadata.append(permutation)
    for i in xrange(0,len(training_data[0])):
        training_data[0][i] = training_data[0][i][permutation]
    for i in xrange(0,len(validation_data[0])):
        validation_data[0][i] = validation_data[0][i][permutation]
    for i in xrange(0,len(test_data[0])):
        test_data[0][i] = test_data[0][i][permutation]
    print training_data[0][0]

    print 'finished shuffling'

    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
        #return shared_x, shared_y
    print np.count_nonzero(training_data[1] == 1), 'positives' #friendly reminder that this number will be less than the output from the data parser, since that one includes CV data
    return [shared(training_data), shared(validation_data), shared(test_data), training_data[1]]

#### Main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        for i in xrange(0, len(layers)):
            net_metadata.append(layers[i].w)
            net_metadata.append(layers[i].b)
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, training_outputs, weight_file, variable_updates = False, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data #try to push changes harder made by positive examples,
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        print 'size(training_data):',size(training_data)
        print 'size(vaidation_data):',size(validation_data)
        print 'size(training_data):', size(test_data)
        print np.count_nonzero(training_outputs == 1), 'positives'

        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches #amplify cost function for positive ex, so net has to make more drastic changes
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]
        """
        amplify cost function for positive ex, so net has to make more drastic changes
        I'm too dumb to find a better solution

        best to use smaller batches, ie larger batch sizes, so less negative examples will be amiplified
        """
        costP = 5*self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches 
        gradsP = T.grad(costP, self.params)
        updatesP = [(param, param-eta*grad)
                   for param, grad in zip(self.params, gradsP)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        train_mbP = theano.function(
            [i], costP, updates=updatesP,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function( # find a way to get the 'y' for the current example
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        test_mb_f1 = theano.function(
            [i], self.layers[-1].f1score(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        # Do the actual training

        best_validation_accuracy = 0.0
        best_roc_score = 0.0
        best_rmse = 1.0
        best_iteration = 0.0
        test_accuracy = 0.0
        start = timeit.default_timer();
        for epoch in xrange(epochs):
            for minibatch_index in xrange(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                batch_outputs =  training_outputs[minibatch_index * mini_batch_size: (minibatch_index + 1) * mini_batch_size ]
                if variable_updates:
                    if 1 in batch_outputs and best_roc_score < .93: # after some point, drastically increasing costs will mess up training?
                        cost_ij = train_mbP(minibatch_index)
                    else:
                        cost_ij = train_mb(minibatch_index)
                else:
                    cost_ij = train_mb(minibatch_index)

                if (iteration+1) % num_training_batches == 0:
                    allActual = []
                    allPredicted = []
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in xrange(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.5%}".format(epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:  
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in xrange(num_test_batches)])
                            print('The corresponding test accuracy is {0:.5%}'.format(test_accuracy))
                    actualOutput = [test_mb_f1(i)[0] for i in xrange(num_test_batches)]
                    for x in xrange (0, len(actualOutput)):
                        for y in xrange (0, mini_batch_size):
                            allActual.append(actualOutput[x][y])
                    predictedOutput = [test_mb_f1(i)[1] for i in xrange(num_test_batches)]
                    for x in xrange (0, len(predictedOutput)):
                        for y in xrange (0, mini_batch_size):
                            allPredicted.append(predictedOutput[x][y])
                    roc_score = roc_auc_score(allActual, allPredicted)
                    rmse = mean_squared_error(allActual, allPredicted)**0.5
                    if roc_score > best_roc_score:
                        best_roc_score = roc_score
                    if rmse < best_rmse:
                        best_rmse = rmse
                    print('Current ROC score is {0:.5%}'.format(roc_score))
                    print('The best ROC score is {0:.5%}'.format(best_roc_score))
                    #save the metadata ocassionally 
                    for x in xrange (0, len(self.layers)):
                        print 'updating ', type(self.layers[x])
                        net_metadata[2 * x + 1] = self.layers[x].w
                        net_metadata[2 * x + 2] = self.layers[x].b
                        f = open('weights/' + weight_file + '.pckl', 'w')
                        pickle.dump(net_metadata, f)
                        f.close()



        stop = timeit.default_timer();
        # print allActual
        # print allPredicted
        print("Finished training network.")
        print "Took ", "{:0>8}".format(datetime.timedelta(seconds=(stop - start)))
        print "F1 score ", f1_score(allActual, allPredicted)
        print "ROC AUC ", best_roc_score
        # print "ROC AUC ", roc_auc_score(allActual[len(allActual) - 500: len(allActual)], allPredicted[len(allPredicted) - 500: len(allPredicted)])
        print "RMSE ", best_rmse
        # print "RMSE ",mean_squared_error(allActual[len(allActual) - 500: len(allActual)], allPredicted[len(allPredicted) - 500: len(allPredicted)])**0.5
        print("Best validation accuracy of {0:.5%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.5%}".format(test_accuracy))

#### Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid, weights = None, biases = None):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        # initialize weights and biases

        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        if weights != None:
            self.w = weights
        else: 
            self.w = theano.shared(
                np.asarray(
                    # np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                    np.ones(filter_shape),
                    dtype=theano.config.floatX), name='w',
                borrow=True)
        if biases != None:
            self.b = biases
        else: 
            self.b = theano.shared(
                np.asarray(
                    np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                    dtype=theano.config.floatX), name='b',
                borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
        pooled_out = downsample.max_pool_2d(
            input=conv_out, ds=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dropout in the convolutional layers

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0, weights = None, biases = None):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases

        #Lets save the weights so we dont have to retrain every time
        if weights != None:
            self.w = weights
        else:
            self.w = theano.shared(
                np.asarray(
                    np.random.normal(
                        loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                    dtype=theano.config.floatX),
                name='w', borrow=True)

        if biases != None:
            self.b = biases
        else:
            self.b = theano.shared(
                np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                           dtype=theano.config.floatX),
                name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0, weights = None, biases = None):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        if weights != None:
            self.w = weights
        else:
            self.w = theano.shared(
                np.zeros((n_in, n_out), dtype=theano.config.floatX),
                name='w', borrow=True)

        if biases != None:
            self.b = biases
        else:
            self.b = theano.shared(
                np.zeros((n_out,), dtype=theano.config.floatX),
                name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

    def f1score(self,y):
        "Return the F1 score for the mini-batch."
        return (y, self.y_out)

    def getTrainY(self):
        "Return the F1 score for the mini-batch."
        return (y)


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
