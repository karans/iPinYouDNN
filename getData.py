import bz2
import pickle
import cPickle
import numpy as np
import random
import sys
from progressbar import *

def vectorized_result(j):
	e = np.zeros((2, 1))
	e[j] = 1.0
	return e

def getData(inputFile, isTestData, debug, nexamples = None):
	testText = inputFile.readlines()
	posText = posfile.readlines()
	posTextTest = posTestFile.readlines()
	print 'finished reading lines'

	#Convert the string into array format and split it up
	testString = ''.join(str(x) for x in testText)
	posString = ''.join(str(x) for x in posText)
	posTextString = ''.join(str(x) for x in posTextTest)

	array = testString.split()
	array = array[0:5400000] # due to small number of positive examples, we sample from the first 10k examples

	#add the manually pulled pos examples form train data
	posArray = posString.split()

	#add the manually pulled pos examples form test data
	posTestArray = posTextString.split()

	posTestArray = np.delete(posTestArray, np.arange(28,len(posTestArray),29))
	posTestArray = np.delete(posTestArray, np.arange(27,len(posTestArray),28))

	print 'array size', len(array), 'memory used', float(sys.getsizeof(array))/1000000 , 'mb'
	array.extend(posArray)
	array.extend(posTestArray)
	print 'array size', len(array), 'memory used', float(sys.getsizeof(array))/1000000 , 'mb'
	print 'finished array creation'

	#grab random training examples, not just the first n ones
	samples = []
	newArray = []
	if nexamples != None:
		nexamples = nexamples -1
		random.seed()
		if isTestData:
			samples = random.sample(range(1,len(array)/29),nexamples + 1)
			for i in range(0, len(samples)):
				for j in range(0,29):
					newArray.append(array[samples[i * 29]+j])
		else:
			samples = random.sample(range(1,len(array)/27),nexamples + 1)
			for i in range(0, len(samples)):
				for j in range(0,27):
					newArray.append(array[(samples[i]*27) + j])
		array = newArray
	print 'grabbed random samples'

	trainingResults = [array[x] for x in xrange(0, len(array), 27)]
	trainingResults = [int(x) for x in trainingResults]
	print trainingResults[0:10]


	print '----Removing hashed variables----'
	print 'click count:', trainingResults.count(1)
	if isTestData:
		array = np.delete(array, np.arange(28,len(array),29))
		print 'removed nclicks'
		array = np.delete(array, np.arange(27,len(array),28))
		print 'removed nconversation'

	array = np.delete(array, np.arange(3,len(array),27))
	print 'removed bidid'
	array = np.delete(array, np.arange(5,len(array),26))
	print 'removed ipinyouid'
	array = np.delete(array, np.arange(10,len(array),25))
	print 'removed domain'
	array = np.delete(array, np.arange(10,len(array),24))
	print 'removed url'
	array = np.delete(array, np.arange(11,len(array),23))
	print 'removed slotid'
	array = np.delete(array, np.arange(16,len(array),22))
	print 'removed creative'
	array = np.delete(array, np.arange(18,len(array),21))
	print 'removed keypage'
	#also removed usertags as i dont know how this information would be relevant to advertisers themselves
	array = np.delete(array, np.arange(19,len(array),20))
	print 'removed usertag'

	#Get the actual values before removing them, 0 - no click, 1 - clicked
	# trainingResults = [array[x] for x in xrange(19, len(array), 19)]

	array = np.delete(array, np.arange(0,len(array),19))
	print 'removed click'
	array = np.delete(array, np.arange(17,len(array),18))
	print 'removed advertiser'
	array = np.delete(array, np.arange(0,len(array),17))
	print 'removed day'
	array = np.delete(array, np.arange(0,len(array),16))
	print 'removed hour'

	array[array == 'null'] = 0

	print '\narray deletion complete\n'

	#remove labels
	# array = array[17:len(array)]


	#split useragent into os and browser and get numeric values
	oses = ["windows", "ios", "mac", "android", "linux", "other"]
	browsers = ["chrome", "sogou", "maxthon", "safari", "firefox", "theworld", "opera", "ie", "other"]

	print 'parsing os browser'
	i = 0;
	for index in pbarOS(xrange(2, len(array), 15)): #24 19
		indexi = index + i
		if debug:
			print array[indexi]
		os = array[indexi][0:array[indexi].index('_')]
		browser = array[indexi][array[indexi].index('_') + 1: len(array[indexi])]
		# print os, browser

		osNumber = oses.index(os) + 1
		browserNumber = browsers.index(browser) + 1

		# print osNumber, browserNumber
		array[indexi] = osNumber
		array = np.insert(array, indexi + 1, browserNumber)
		i += 1

	# print array[len(array) - 42: len(array)]
	print 'os browser parsing complete\n\n'

	print 'parsing timestamp'
	i = 0
	#split time YYYY/MM/DD/HH:MM:SS:MSMSMS to its components
	for index in pbarTime(xrange(0, len(array), 16)): #20
		indexi = index + i
		if debug:
			print array[indexi]
		year = array[indexi][0:4]
		month = array[indexi][4:6]
		day = array[indexi][6:8]
		hour = array[indexi][8:10]
		m = array[indexi][10:12]
		sec = array[indexi][12:14]
		ms = array[indexi][14:17]

		# print osNumber, browserNumber
		array[indexi] = year
		array = np.insert(array, indexi + 1, month)
		array = np.insert(array, indexi + 2, day)
		array = np.insert(array, indexi + 3, hour)
		array = np.insert(array, indexi + 4, m)
		array = np.insert(array, indexi + 5, sec)
		array = np.insert(array, indexi + 6, ms)
		i += 6
	print 'timestamp parsing complete\n\n'

	print 'parsing ip'
	i = 0
	#split ip address xxx.xxx.xxx
	for index in pbarIP(xrange(10, len(array), 22)): #26
		indexi = index + i
		if debug:
			print array[indexi]
		address = array[indexi]
		sec1 = address[0:address.index('.')]
		address = address[address.index('.') + 1: len(address)]	
		sec2 = address[0:address.index('.')]
		address = address[address.index('.') + 1: len(address)]
		sec3 = address[0:address.index('.')]
		
		# print osNumber, browserNumber
		array[indexi] = sec1
		array = np.insert(array, indexi + 1, sec2)
		array = np.insert(array, indexi + 2, sec3)
		i += 2
	print 'ip parsing complete\n\n'

	array = [int(x) for x in array]
	# trainingResults = [int(x) for x in trainingResults]

	#TODO: normalize all varaibles for training
	for i in range(0,24):
		minVal = array[i]
		maxVal = array[i]
		totalVal = 0
		for index in xrange(i, len(array), 24):
			if array[index] < minVal:
				minVal = array[index]
			if array[index] > maxVal:
				maxVal = array[index]
			totalVal += array[index]
		deviation = maxVal - minVal
		average = float(totalVal)/(len(array)/24)
		if deviation == 0:
			deviation = 1
		# print 'deviation:', deviation, 'average:', average
		for index in xrange(i, len(array), 24):
			array[index] = (array[index] - average)/deviation
	print 'mean normalization complete'

	print 'set size:', len(array)/24
	print 'click count:', trainingResults.count(1)

	#convert it into usable input arrays
	trainingResults = np.array(trainingResults)
	trainingResults = trainingResults.astype(np.int)

	array = np.array(array)
	array = array.astype(np.float32)
	array = array.reshape(len(array)/24, 24)

	trainingSet = (array, trainingResults)
	if debug:
		print trainingSet, type(trainingSet)
	return trainingSet


widgets = [Percentage(),
               ' ', Bar(),
               ' ', ETA(), '                                                  ']

pbarOS = ProgressBar(widgets = widgets)
pbarTime = ProgressBar(widgets = widgets)
pbarIP = ProgressBar(widgets = widgets)


#Open and read file for advertiser 1458
testfile = open('/Users/karansamel/Documents/GitRepos/make-ipinyou-data/1458/train.log.txt', 'r')
posfile = open('/Users/karansamel/Documents/GitRepos/make-ipinyou-data/1458/posData.txt', 'r')
posTestFile = open('/Users/karansamel/Documents/GitRepos/make-ipinyou-data/1458/posDataTest.txt', 'r')

data = getData(testfile, False, False,50000)
testfile.close()

#Separates data, only works if in multiples of 5, might start with ~60,000, suggested breakdowns are 60% training, 20% cross validation, 20% testing
training_data = (data[0][0: int(len(data[0]) * .5)], data[1][0: int(len(data[1]) * .5)])
validation_data = (data[0][int(len(data[0]) * .5): int(len(data[0]))], data[1][int(len(data[1]) * .5): int(len(data[1]))])
test_data = (data[0][int(len(data[0]) * .5): int(len(data[0]))], data[1][int(len(data[1]) * .5): int(len(data[1]))])

# USE TO SAVE AND RESTORE LARGE DATA SETS
# f = open('allData.pckl', 'w')
# pickle.dump([training_data, validation_data, test_data], f)
# f.close()

# f = open('allData.pckl')
# training_data, validation_data, test_data = cPickle.load(f)
# f.close()

# testfile = open('/Users/karansamel/Documents/GitRepos/make-ipinyou-data/1458/test.log.txt', 'r')
# getData(testfile, True, False,10000)
# testfile.close()

import network3, pickle
from network3 import Network, ReLU
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
file = open('weights/1458-200k.pckl')
metadata = pickle.load(file)
file.close()
training_data, validation_data, test_data, training_outputs = network3.load_data_shared(filename = '1458partial200k', perm = metadata[0])
mini_batch_size = 10
net = Network([
		FullyConnectedLayer(n_in=464, n_out=200, weights = metadata[1], biases = metadata[2]),
        FullyConnectedLayer(n_in=200, n_out=200, weights = metadata[3], biases = metadata[4]),
        SoftmaxLayer(n_in=200, n_out=2, weights = metadata[5], biases = metadata[6])], mini_batch_size)
net.SGD(training_data, 2000, mini_batch_size, 1, 
            validation_data, test_data, training_outputs, '1458-MLP')


import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data, training_outputs = network3.load_data_shared()
mini_batch_size = 10
net = Network([
        FullyConnectedLayer(n_in=464, n_out=200),
        FullyConnectedLayer(n_in=200, n_out=200),
        SoftmaxLayer(n_in=200, n_out=2)], mini_batch_size)
net.SGD(training_data, 200, mini_batch_size, .1, 
            validation_data, test_data, training_outputs)



############################## 1 CONV ############################## 
import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10
net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 16, 29), 
                  filter_shape=(1, 1, 5, 6), 
                  poolsize=(2, 2)),
        FullyConnectedLayer(n_in=1 * 6 * 12, n_out=70),
        SoftmaxLayer(n_in=70, n_out=2)], mini_batch_size)
net.SGD(training_data, 200, mini_batch_size, .1, 
            validation_data, test_data)


import network3, pickle
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
file = open('weightsCNN1.pckl')
metadata = pickle.load(file)
file.close()
training_data, validation_data, test_data = network3.load_data_shared(perm = metadata[0])
mini_batch_size = 10
net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 16, 29), 
                  filter_shape=(1, 1, 5, 6), 
                  poolsize=(2, 2), weights = metadata[1], biases = metadata[0]),
        FullyConnectedLayer(n_in=1 * 6 * 12, n_out=70, weights = metadata [1], biases = metadata[2]),
        SoftmaxLayer(n_in=70, n_out=2, weights = metadata[3], biases =  metadata[4])], mini_batch_size)
net.SGD(training_data, 200, mini_batch_size, .1, 
            validation_data, test_data)
############################## 2 CONV ############################## 

import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10
net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 16, 29), 
                  filter_shape=(1, 1, 2, 2), 
                  poolsize=(1, 1)),
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 15, 28), 
                  filter_shape=(1, 1, 2, 2), 
                  poolsize=(1, 1)),
        FullyConnectedLayer(n_in=1 * 14 * 27, n_out=300),
        SoftmaxLayer(n_in=300, n_out=2)], mini_batch_size)
net.SGD(training_data, 1200, mini_batch_size, .1, 
            validation_data, test_data)

import network3, pickle
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
file = open('weights/1458-1.pckl')
metadata = pickle.load(file)
file.close()
training_data, validation_data, test_data = network3.load_data_shared(perm = metadata[0])
mini_batch_size = 10
net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 16, 29), 
                  filter_shape=(1, 1, 2, 2), 
                  poolsize=(1, 1), weights = metadata[1], biases = metadata[2]),
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 15, 28), 
                  filter_shape=(1, 1, 2, 2), 
                  poolsize=(1, 1), weights = metadata[3], biases = metadata[4]),
        FullyConnectedLayer(n_in=1 * 14 * 27, n_out=300, weights = metadata[5], biases = metadata[6]),
        SoftmaxLayer(n_in=300, n_out=2, weights = metadata[7], biases = metadata[8])], mini_batch_size)
net.SGD(training_data, 1200, mini_batch_size, .1, 
            validation_data, test_data)

import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data, training_outputs = network3.load_data_shared()
mini_batch_size = 10
net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 16, 29), 
                  filter_shape=(1, 1, 2, 2), 
                  poolsize=(1, 1)),
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 15, 28), 
                  filter_shape=(1, 1, 2, 2), 
                  poolsize=(1, 1)),
        FullyConnectedLayer(n_in=1 * 14 * 27, n_out=300),
        SoftmaxLayer(n_in=300, n_out=2)], mini_batch_size)
net.SGD(training_data, 1200, mini_batch_size, .1, 
            validation_data, test_data, training_outputs)


############################## 1 CONV OLD ############################## 

import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10
net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 4, 6), 
                      filter_shape=(20, 1, 3, 3), 
                      poolsize=(1, 1)),
        FullyConnectedLayer(n_in=20*2*4, n_out=70),
        SoftmaxLayer(n_in=70, n_out=2)], mini_batch_size)
net.SGD(training_data, 200, mini_batch_size, .1, 
            validation_data, test_data)

############################## 0 CONV OLD ############################## 


import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10
net = Network([
        FullyConnectedLayer(n_in=464, n_out=200),
        SoftmaxLayer(n_in=200, n_out=2)], mini_batch_size)
net.SGD(training_data, 200, mini_batch_size, .1, 
            validation_data, test_data)


import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data, training_outputs = network3.load_data_shared()
mini_batch_size = 10
net = Network([
        FullyConnectedLayer(n_in=464, n_out=200),
        SoftmaxLayer(n_in=200, n_out=2)], mini_batch_size)
net.SGD(training_data, 200, mini_batch_size, .1, 
            validation_data, test_data, training_outputs)

############################## 0 CONV ############################## 

import network3, pickle
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
file = open('weights.pckl')
metadata = pickle.load(file)
file.close()
training_data, validation_data, test_data = network3.load_data_shared(perm = metadata[0])
mini_batch_size = 10
net = Network([
        FullyConnectedLayer(n_in=464, n_out=200, weights = metadata[1], biases = metadata[2]),
        SoftmaxLayer(n_in=200, n_out=2, weights = metadata[3], biases = metadata[4])], mini_batch_size)
net.SGD(training_data, 200, mini_batch_size, .1, 
            validation_data, test_data)


import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10
net = Network([
	FullyConnectedLayer(n_in=784, n_out=100),
	SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, 
            validation_data, test_data)


# click	weekday	hour	bidid								timestamp			logtype	ipinyouid		useragent	IP			
# 0		6		00		81aced04baad90f9358aa39a4521cd6f	20130606000104828	1		Vhk7ZAnxDIuOjCn	windows_ie	115.45.195.*

# region	city	adexchange	domain					url									urlid	slotid							slotwidth	slotheight	slotvisibility	slotformat	slotprice	creative							bidprice	payprice	keypage								advertiser	usertag	
# 216		219		1			trqRTJkrBoq7JsNr5SqfNX	f41292b3547399af082eccc2ad28f23c	null	mm_34022157_3445226_11175096	336			280			2				1			0			77819d3e0b3467fe5c7b16d68ad923a1	300			51			bebefa5efe83beee17a3d245e7c5085b	1458		10006,10110

# click	weekday	hour	bidid	timestamp	logtype	ipinyouid	useragent	IP	region	city	adexchange	domain	url	urlid	slotid	slotwidth	slotheight	slotvisibility	slotformat	slotprice	creative	bidprice	payprice	keypage	advertiser	usertag	nclick	nconversation



# click	weekday	hour	bidid	timestamp	logtype	ipinyouid	useragent	IP	region	city	adexchange	domain	url	urlid	slotid	slotwidth	slotheight	slotvisibility	slotformat	slotprice	creative	bidprice	payprice	keypage	advertiser	usertag	nclick	nconversation
# 0	6	00	83d080b8bbb0be814ed561c407a1cc13	20130613000101634	1	VhTdLndvPoTLQYE	windows_ie	221.194.182.*	3	8	2	trqRTvKJBQpvjqK0uG	3feb136feef0b22c2d07f550a179ce84	null	3582857028	336	280	2	0	5	77819d3e0b3467fe5c7b16d68ad923a1	300	118	bebefa5efe83beee17a3d245e7c5085b	1458	13866,10075,10024,10076,10063,10120,10129,10115,10146,10111	0	0



#Now we have to convert the array into the usable form by removing unwanted variables and assigning numeric values
#Lets start by removing hashed values wont give us any information
# 0    1     2   3    4      5      6           7       8       9  10   11   12   13     14   15         16    17        18         19             20          21       22  23 
# year month day hour minute second millisecond logtype browser OS IP_1 IP_2 IP_3 region city adexchange urlid slotwidth slotheight slotvisibility slotformat slotprice bid payprice

# proposed matrix
# 1  2  4  0  8  16
# 3  4  6  7  9  23
# 10 11 14 17 18 21
# 12 13 15 19 20 22


[284  37 252 282 221 251 427 319 116 453 351 205  44 110 290 207 167 279
 363 436  90 273 145 366 301 139 121 133 342 160 216 232 437 412 410 297
 316 113  98  36 244 315 460 323 317 158 417 285 308 131 355 374 450 346
  88 231 122 385 261 151  70 338 137 439  28 388 397 102 275  56 383 250
 321 283 403 461 191 227 367  69 237  93 215 240 458 173 196  24 274 161
 164 377 272 277 313 263 242 109 138 452  53 349 456 309  26 415 219 289
 114 245   4 395  97 210 335 378 197 162 100 416  40 449 195 424  84  76
 387 299 350 327 325   5 345  68 203  38   1  19  51 310 302 320 418 426
 359 347  10 322  33 447 132  61  58 118  62 411 257 182 184  13 348 130
  64 129 269  32 266  71 376 214 103 409 361 365  99 217 414 101 344 294
 394 222 159 448 168 198 281  41 104 228 144 268  65 241   2 459 303  11
 391 463  47 364 286 402 146 135 149 280 451 413 204  54 353 209 399  75
 333 428 384 230  55 233 115 262 454 300 433 108  14 405  45 318 373  86
  17 295 371  82 128  39  52  92 393 156 189 141 432 169 440  12 288 199
 236 352 147 334 123  95 341  79 180 296 382 336 206 111  66   6 194 421
 455 136 389 330 362 332 304 331 213 254 239 153 124  50  25 298 312 201
  83 343 140 287  27  96 154 337 442 178 370 340 105  91 444  30 369  46
 380 235  31 117 225 247 354 179 425 390  22 148 386   8 314 392 375  43
 229 190 445 172 306 379 224 126 200  49  15 202 434 291 166 163 305 422
  60 143 292 134 238 259  29 174 435 256   0 258 462 324 127 270 119 400
   3  72 307  48  87 253 187  73 420  89 404  23 396  80  85  78 248  59
 276 278 152 246  63 293 398 311 372  21 125 401 265  20  67 112 185  16
  42 381  57 457 255  35 157 430 267 193 107  77 326 358  34 186 356 423
 234 171 181 368 357 220 170 446 120 183 271 208  74 249 431   7 212 406
 243 142 177 264 150 226 438 407 429 155 339  81 211 218 106 176 443 360
 419 328 175 165 329   9  94 408 188 223 441 192 260  18]

