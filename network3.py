"""network3.py
Neural network and convolution classes used for running models.
This is modified code based on Michael Nielsen's sample code: (https://github.com/mnielsen/neural-networks-and-deep-learning)

Extra features include weight saving, logging, epoch and learning rate tuning, ROC_AUC evaluation, and various activation functions.
Code is stil in development to reflect my current reserach. 
"""

#### Libraries
# Standard library
import cPickle
import pickle
import gzip
import sys
import os
import atexit


# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
# from theano.tensor.signal import downsample
from theano.tensor.signal.pool import pool_2d

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error #have to sqrt after

import timeit
import datetime

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


print '\n[device =',theano.config.device, 'floatX =', theano.config.floatX, ']\n'

net_metadata = [] # we want to store all our weights and permunations here so we can restart where we left off, just a list of arrays
weightFile = 0

def save_params():
    f = open('weights/' + weightFile + '.pckl', 'wb')
    pickle.dump(net_metadata, f)
    f.close()

def save_at_exit():
    print 'Saving metadata before exit'
    save_params()

# atexit.register(save_at_exit)

#Load Data
def load_data_shared(filename, perm = None):

    f = open('data/' + filename + '.pckl')
    training_data, validation_data, test_data = pickle.load(f)
    f.close()    

    #want to shuffle train and cv set for any n-fold cross validation
    print 'Generating new test-validation combination'
    
    fullData = (np.concatenate((training_data[0],validation_data[0])),np.concatenate((training_data[1],validation_data[1])))
    cvPerm = np.random.permutation(len(fullData[0]))

    newX = fullData[0][cvPerm]
    newY = fullData[1][cvPerm]

    training_data_S = (newX[0:int(len(newX)*.8)], newY[0:int(len(newY)*.8)])
    validation_data_S = (newX[int(len(newX)*.8):], newY[int(len(newY)*.8):])

    if perm != None:
        permutation = perm
    else:
        permutation = np.random.permutation(len(training_data_S[0][0]))
    
    #print 'current permuation:', permutation
    net_metadata.append(permutation)

    # We orignally tried randomly shuffling the data (permuation) to see if convolution layers would develop strong features.

    for i in xrange(0,len(training_data_S[0])):
        training_data_S[0][i] = training_data_S[0][i][permutation]
    for i in xrange(0,len(validation_data_S[0])):
        validation_data_S[0][i] = validation_data_S[0][i][permutation]
    for i in xrange(0,len(test_data[0])):
        test_data[0][i] = test_data[0][i][permutation]
    print 'finished shuffling'

    print np.count_nonzero(training_data_S[1] == 1), 'training positives'
    print np.count_nonzero(validation_data_S[1] == 1), 'validation positives'
    print np.count_nonzero(test_data[1] == 1), 'test positives'


    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data_S), shared(validation_data_S), shared(test_data), training_data_S[1]]

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
        self.actualOutput = []
        self.predictedOutput = []

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, training_outputs, weight_file = None, variable_updates = False, log_file=None, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data #try to push changes harder made by positive examples,
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        global weightFile, net_metadata
        weightFile = weight_file

        # compute number of minibatches for training, validation and testing
        print 'size of the training data:',size(training_data)
        print 'size of the vaidation data:',size(validation_data)
        print 'size of the testing data:', size(test_data)

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
        updatesOne = [(param, param- grad) for param, grad in zip(self.params, grads)] #if we are at .5 ROC just keep eta at 1 to move it along
        """
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
        train_mbOne = theano.function(
            [i], cost, updates=updatesOne,
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
        validate_mb_f1 = theano.function(
            [i], self.layers[-1].f1score(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        train_mb_f1 = theano.function(
            [i], self.layers[-1].f1score(self.y),
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        # Do the actual training

        best_validation_roc = 0.0
        best_roc_score = 0.0
        best_test_roc_score = 0.0
        best_rmse = 1.0
        test_roc_score = 0.0
        validation_roc_score = 0.0
        cost_ij = 0.0
        start = timeit.default_timer();

        if log_file is not None:
            try:
                os.remove(log_file + '.txt')
            except OSError:
                pass

        def update_params():
            if weight_file is not None:
                # print 'updating parameters'
                for x in xrange (0, len(self.layers)):
                    # print 'updating ', type(self.layers[x])
                    net_metadata[2 * x + 1] = self.layers[x].w
                    net_metadata[2 * x + 2] = self.layers[x].b
        def train_roc():
            allActual = []
            allPredicted = []
            #validation
            self.actualOutput = [train_mb_f1(i)[0] for i in xrange(num_training_batches)]
            for x in xrange (0, len(self.actualOutput)):
                for y in xrange (0, mini_batch_size):
                    allActual.append(self.actualOutput[x][y])
            self.predictedOutput = [train_mb_f1(i)[1] for i in xrange(num_training_batches)]
            for x in xrange (0, len(self.predictedOutput)):
                for y in xrange (0, mini_batch_size):
                    allPredicted.append(self.predictedOutput[x][y])
            return roc_auc_score(allActual, allPredicted)
        def validation_roc():
            allActual = []
            allPredicted = []
            #validation
            self.actualOutput = [validate_mb_f1(i)[0] for i in xrange(num_validation_batches)]
            for x in xrange (0, len(self.actualOutput)):
                for y in xrange (0, mini_batch_size):
                    allActual.append(self.actualOutput[x][y])
            self.predictedOutput = [validate_mb_f1(i)[1] for i in xrange(num_validation_batches)]
            for x in xrange (0, len(self.predictedOutput)):
                for y in xrange (0, mini_batch_size):
                    allPredicted.append(self.predictedOutput[x][y])
            return roc_auc_score(allActual, allPredicted)

        def test_roc():
            allActual = []
            allPredicted = []
            self.actualOutput = [test_mb_f1(i)[0] for i in xrange(num_test_batches)]
            for x in xrange (0, len(self.actualOutput)):
                for y in xrange (0, mini_batch_size):
                    allActual.append(self.actualOutput[x][y])
            self.predictedOutput = [test_mb_f1(i)[1] for i in xrange(num_test_batches)]
            for x in xrange (0, len(self.predictedOutput)):
                for y in xrange (0, mini_batch_size):
                    allPredicted.append(self.predictedOutput[x][y]) 
            return (roc_auc_score(allActual, allPredicted), mean_squared_error(allActual, allPredicted)**0.5)           


        sinceLastTest = timeit.default_timer()

        validation_roc_score = validation_roc()
        print '\n', 'Initial validation ROC score is',validation_roc_score
        test_roc_score, best_rmse = test_roc()
        best_roc_score = test_roc_score
        update_params()
        save_params()
        print 'Current test ROC score is', test_roc_score, '\n'

        times = open('scores/' + log_file + '.txt', 'a')
        times.write('epoch' + '\t' + 'time elapsed' + '\t' + 'training_roc' + '\t' + 'validation_roc' + '\t' + 'test_roc' '\t' + 'best_test_roc' + '\n')
        times.close()


        for epoch in xrange(epochs):
            for minibatch_index in xrange(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 100000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                    stop = timeit.default_timer();
                    print "{:0>8}".format(datetime.timedelta(seconds=(stop - start))), 'elapsed'
                batch_outputs =  training_outputs[minibatch_index * mini_batch_size: (minibatch_index + 1) * mini_batch_size ]
                if variable_updates:
                    if 1 in batch_outputs and best_roc_score < .93: # after some point, drastically increasing costs will mess up training?
                        cost_ij = train_mbP(minibatch_index)
                    else:
                        cost_ij = train_mb(minibatch_index)
                else:
                    # if best_roc_score <= .5:
                    #     cost_ij = train_mbOne(minibatch_index)
                    # else:
                    cost_ij = train_mb(minibatch_index)
                # if (iteration+1) % num_training_batches == 0:

                if (timeit.default_timer() - sinceLastTest)/60 >= 2: #time till next test in minutes
                    
                    #cross validation
                    training_roc_score = train_roc()
                    validation_roc_score = validation_roc()
                    print '\n', 'Epoch', epoch, ': Current training ROC score is',training_roc_score
                    print 'Current validation ROC score is',validation_roc_score
                    #testing
                    
                    test_roc_score, rmse = test_roc()
                    if validation_roc_score > best_roc_score:
                        best_roc_score = validation_roc_score
                        best_test_roc_score = test_roc_score
                        update_params()
                        save_params()
                        f = open('weights/network.pckl', 'wb')
                        pickle.dump(net_metadata, f)
                        f.close()
                    if rmse < best_rmse:
                        best_rmse = rmse
                    print 'Current test ROC score is', test_roc_score
                    print 'The best test ROC score is', best_test_roc_score
                    print 'weight file =',weight_file, ' log file = ', log_file, '\n' 

                    stop = timeit.default_timer();
                    
                    times = open('scores/' + log_file + '.txt', 'a')
                    times.write(str(epoch) + '\t' + str((stop - start)/60) + '\t' + str(training_roc_score) + '\t' + str(validation_roc_score) + '\t' + str(test_roc_score)+ '\t' + str(best_test_roc_score)+ '\n')
                    times.close()
                    
                    #save the metadata ocassionally 
                    sinceLastTest = timeit.default_timer()
            if epoch in xrange(0,11):
                print 'Epoch', epoch, 'with cost of', cost_ij



        stop = timeit.default_timer();
        # print allActual
        # print allPredicted
        print("Finished training network.")
        print "Took ", "{:0>8}".format(datetime.timedelta(seconds=(stop - start)))
        print "F1 score ", f1_score(allActual, allPredicted)
        print "ROC AUC ", best_roc_score
        print "RMSE ", best_rmse

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
                    np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                    # np.ones(filter_shape),
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
        pooled_out = pool_2d(
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
            # self.w = theano.shared(
            #     np.asarray(
            #         np.random.normal(
            #             loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
            #         dtype=theano.config.floatX),
            #     name='w', borrow=True)
            #tanh initialization
            # self.w = theano.shared(
            #     np.asarray(
            #         np.random.uniform(
            #             low=-(float(6)/(n_in + n_out))**.5, high=(float(6)/(n_in + n_out))**.5, size=(n_in, n_out)),
            #         dtype=theano.config.floatX),
            #     name='w', borrow=True)
            self.w = theano.shared(
                np.asarray(
                    np.random.uniform(
                        low=-4*(float(6)/(n_in + n_out))**.5, high=4*(float(6)/(n_in + n_out))**.5, size=(n_in, n_out)),
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
        # self.output = 1.7159*tanh((float(2)/3)*T.dot(self.inpt, self.w) + self.b)
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