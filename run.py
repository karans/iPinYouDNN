# def SGD(self, training_data, epochs, mini_batch_size, eta, validation_data, test_data, training_outputs, weight_file = None, variable_updates = False, log_file=None, lmbda=0.0):

# import network3, pickle
# from network3 import Network, ReLU
# from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
# file = open('weights/50PercentNeg1HL.pckl')
# metadata = pickle.load(file)
# file.close()
# training_data, validation_data, test_data, training_outputs = network3.load_data_shared(filename = '50PercentNeg1HL', perm = metadata[0])
# mini_batch_size = 500
# net = Network([
# 		FullyConnectedLayer(n_in=464, n_out=400, weights = metadata[1], biases = metadata[2]),
#         FullyConnectedLayer(n_in=400, n_out=400, weights = metadata[3], biases = metadata[4]),
#         SoftmaxLayer(n_in=400, n_out=2, weights = metadata[5], biases = metadata[6])], mini_batch_size)
# net.SGD(training_data, 10000000, mini_batch_size, .1, 
#             validation_data, test_data, training_outputs, weight_file = '50PercentNeg1HL', variable_updates = False, log_file = '50PercentNeg1HL')

import network3, pickle
from network3 import Network, ReLU
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data, training_outputs = network3.load_data_shared(filename = '50PercentNeg')
mini_batch_size = 500
net = Network([
		FullyConnectedLayer(n_in=464, n_out=200000),
        SoftmaxLayer(n_in=200000, n_out=2)], mini_batch_size)
net.SGD(training_data, 100000, mini_batch_size, .1, 
            validation_data, test_data, training_outputs, weight_file = '50PercentNeg1HL', variable_updates = False, log_file = '50PercentNeg1HL')

# import network3, pickle
# from network3 import Network, ReLU
# from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
# file = open('weights/50PercentNeg1HL.pckl')
# metadata = pickle.load(file)
# file.close()
# training_data, validation_data, test_data, training_outputs = network3.load_data_shared(filename = '50PercentNeg', perm = metadata[0])
# mini_batch_size = 1
# net = Network([
# 		FullyConnectedLayer(n_in=464, n_out=400, weights = metadata[1], biases = metadata[2]),
#         SoftmaxLayer(n_in=400, n_out=2, weights = metadata[3], biases = metadata[4])], mini_batch_size)
# net.SGD(training_data, 10000000, mini_batch_size, 1, 
#             validation_data, test_data, training_outputs, weight_file = '50PercentNeg1HL', variable_updates = True, log_file = '50PercentNeg1HL')
