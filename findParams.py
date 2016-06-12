import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from multiprocessing import Process, Queue


mini_batch_size = 10


# def train(filterSize, hiddenLayers): #do da multiprocessing
# for filterSize in xrange(2,5):
for hiddenLayers in xrange(40, 100, 20):
	training_data, validation_data, test_data = network3.load_data_shared()
	height = 4
	width = 6
	# print '------Training for ', filterSize, ' x ', filterSize, 'filterSize and ', hiddenLayers, 'hiddenLayers------'
	# net = Network([
	# 		ConvPoolLayer(image_shape=(mini_batch_size, 1, height, width), 
	# 					  filter_shape=(20, 1, filterSize, filterSize), 
	# 					  poolsize=(1, 1)),
	# 		FullyConnectedLayer(n_in=20* (height - filterSize + 1) * (width - filterSize + 1), n_out=hiddenLayers),
	# 		SoftmaxLayer(n_in=hiddenLayers, n_out=2)], mini_batch_size)
	net = Network([
	        FullyConnectedLayer(n_in=24, n_out=hiddenLayers),
	        SoftmaxLayer(n_in=hiddenLayers, n_out=2)], mini_batch_size)
	net.SGD(training_data, 60, mini_batch_size, 0.1, 
				validation_data, test_data)
		# print '------Finished Training for ', filterSize, ' x ', filterSize, 'filterSize and ', hiddenLayers, 'hiddenLayers------'

# p1 = Process(target=train, args=(2,40))
# p2 = Process(target=train, args=(3,40))
# p1.start()
# p2.start()
