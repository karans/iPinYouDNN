import bz2
import pickle
import cPickle
import numpy as np
import random
import itertools
import sys
import timeit
from progressbar import *
# from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing.pool import ThreadPool


def vectorized_result(j):
	e = np.zeros((2, 1))
	e[j] = 1.0
	return e

def getData(inputFile, isTestData, debug, section = None, nexamples = None):
	start = timeit.default_timer();
	testText = inputFile.readlines()
	posText = posFile.readlines()
	# posTextTest = posTestFile.readlines()
	print 'finished reading lines'

	#Convert the string into array format and split it up
	testString = ''.join(str(x) for x in testText)

	posString = ''
	if isTestData == False:
		posString = ''.join(str(x) for x in posText)
	# posTextString = ''.join(str(x) for x in posTextTest)

	array = testString.split()
	# if isTestData == False:
	# 	array = array[0:1450000]
	# else:
	# 	array = array[0:205200] # due to small number of positive examples, we sample from the first 10k examples

	# array[array == 'null'] = 0 # in case we run into any problems 
	array = [0 if x == 'null' else x for x in array]

	# #add the manually pulled pos examples form train data
	posArray = posString.split()
	posArray = [0 if x == 'null' else x for x in posArray]

	# #add the manually pulled pos examples form test data
	# posTestArray = posTextString.split()

	# posTestArray = np.delete(posTestArray, np.arange(28,len(posTestArray),29))
	# posTestArray = np.delete(posTestArray, np.arange(27,len(posTestArray),28))

	# if isTestData == False:
	# 	array.extend(posTestArray)
	# else:
	# 	array.extend(posArray)

	#There are the labels at the top, get rid of those

	samples = []
	newArray = []
	if nexamples != None:
		nexamples = nexamples -1
		random.seed()
		if isTestData:
			samples = random.sample(range(1,len(array)/29),nexamples + 1)
			for i in range(0, len(samples)):
				for j in range(0,27):
					newArray.append(array[(samples[i]*29) + j])
		else:
			samples = random.sample(range(1,len(array)/27),nexamples + 1)
			for i in range(0, len(samples)):
				for j in range(0,27):
					newArray.append(array[(samples[i]*27) + j])
		array = newArray
	else:
		if isTestData:
			del array[27::29]
			del array[27::28]

	#Get rid of the labels at the top
	array = array[27:]

	if isTestData==False:
		for i in xrange(0,5):
			array.extend(posArray)
	# else:
	# 	if section == 1:
	# 		array.extend(posArray[0:int(len(posArray)/27 * .6) * 27])
	# 	elif section == 2:
	# 		array.extend(posArray[int(len(posArray)/27 * .6) * 27: int(len(posArray)/27 * .8) * 27])
	# 	else:
	# 		array.extend(posArray[int(len(posArray)/27 * .8) * 27:])


	# partition = [array[n:n+27] for n in range(0, len(array),27)]
	# np.random.shuffle(partition)
	# array = list(itertools.chain.from_iterable(partition))


	print 'grabbed random samples'
	trainingResults = [array[x] for x in xrange(0, len(array), 27)]
	trainingResults = [int(x) for x in trainingResults]

	#categories for variables
	oses = ["windows", "ios", "mac", "android", "linux", "other"]
	browsers = ["chrome", "sogou", "maxthon", "safari", "firefox", "theworld", "opera", "ie", "other"]
	# awk 'BEGIN {ORS=" "; print "[" }; {print $1, ","} END{print "]","\n"}'
	#cat trainPartial50k.txt | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > trainPartial50k2.txt
	regions = [ 0 , 1 , 2 , 3 , 15 , 27 , 40 , 55 , 65 , 79 , 80 , 94 , 106 , 124 , 134 , 146 , 164 , 183 , 201 , 216 , 238 , 253 , 275 , 276 , 298 , 308 , 325 , 333 , 344 , 359 , 368 , 374 , 393 , 394 , 395]
	cities = [ 0 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 16 , 17 , 18 , 19 , 20 , 21 , 22 , 23 , 24 , 25 , 26 , 28 , 29 , 30 , 31 , 32 , 33 , 34 , 35 , 36 , 37 , 38 , 39 , 41 , 42 , 43 , 44 , 45 , 46 , 47 , 48 , 49 , 50 , 51 , 52 , 53 , 54 , 56 , 57 , 58 , 59 , 60 , 61 , 62 , 63 , 64 , 66 , 67 , 68 , 69 , 70 , 71 , 72 , 73 , 74 , 75 , 76 , 77 , 78 , 81 , 82 , 83 , 84 , 85 , 86 , 87 , 88 , 89 , 90 , 91 , 92 , 93 , 95 , 96 , 97 , 98 , 99 , 100 , 101 , 102 , 103 , 104 , 105 , 107 , 108 , 109 , 110 , 111 , 112 , 113 , 114 , 115 , 116 , 117 , 118 , 119 , 120 , 121 , 122 , 123 , 125 , 126 , 127 , 128 , 129 , 130 , 131 , 132 , 133 , 135 , 136 , 137 , 138 , 139 , 140 , 141 , 142 , 143 , 144 , 145 , 147 , 148 , 149 , 150 , 151 , 152 , 153 , 154 , 155 , 156 , 157 , 158 , 159 , 160 , 161 , 162 , 163 , 165 , 166 , 167 , 168 , 169 , 170 , 171 , 172 , 173 , 174 , 175 , 176 , 177 , 178 , 179 , 180 , 181 , 182 , 184 , 185 , 186 , 187 , 188 , 189 , 190 , 191 , 192 , 193 , 194 , 195 , 196 , 197 , 198 , 199 , 200 , 202 , 203 , 204 , 205 , 206 , 207 , 208 , 209 , 210 , 211 , 212 , 213 , 214 , 215 , 217 , 218 , 219 , 220 , 221 , 222 , 223 , 224 , 225 , 226 , 227 , 228 , 229 , 230 , 231 , 232 , 233 , 234 , 235 , 236 , 237 , 239 , 240 , 241 , 242 , 243 , 244 , 245 , 246 , 247 , 248 , 249 , 250 , 251 , 252 , 254 , 255 , 256 , 257 , 258 , 259 , 260 , 261 , 262 , 263 , 264 , 265 , 266 , 267 , 268 , 269 , 270 , 271 , 272 , 273 , 274 , 277 , 278 , 279 , 280 , 281 , 282 , 283 , 284 , 285 , 286 , 287 , 288 , 289 , 290 , 291 , 292 , 293 , 294 , 295 , 296 , 297 , 299 , 300 , 301 , 302 , 303 , 304 , 305 , 306 , 307 , 309 , 310 , 311 , 312 , 313 , 314 , 315 , 316 , 317 , 318 , 319 , 320 , 321 , 322 , 323 , 324 , 326 , 327 , 328 , 329 , 330 , 331 , 332 , 334 , 335 , 336 , 337 , 338 , 339 , 340 , 341 , 342 , 343 , 345 , 346 , 347 , 348 , 349 , 350 , 351 , 352 , 353 , 354 , 355 , 356 , 357 , 358 , 360 , 361 , 362 , 363 , 364 , 365 , 366 , 367 , 369 , 370 , 371 , 372 , 373 , 375 , 376 , 377 , 378 , 379 , 380 , 381 , 382 , 383 , 384 , 385 , 386 , 387 , 388 , 389 , 390 , 391 , 392, 1 , 2 , 3 , 15 , 27 , 40 , 55 , 65 , 79 , 80 , 94 , 106 , 124 , 134 , 146 , 164 , 183 , 201 , 216 , 238 , 253 , 275 , 276 , 298 , 308 , 325 , 333 , 344 , 359 , 368 , 374 , 393 , 394 , 395] 
	allTags = [ 0,10006 , 10024 , 10031 , 10048 , 10052 , 10057 , 10059 , 10063 , 10067 , 10074 , 10075 , 10076 , 10077 , 10079 , 10083 , 10093 , 10102 , 10684 , 11092 , 11278 , 11379 , 11423 , 11512 , 11576 , 11632 , 11680 , 11724 , 11944 , 13042 , 13403 , 13496 , 13678 , 13776 , 13800 , 13866 , 13874 , 14273 , 16593 , 16617 , 16661 , 16706 , 16751 , 10110 , 10111] 
	slotVisibilities = ['FirstView','SecondView','ThirdView', 'FourthView', 'FifthView','OtherView', 'Na']


	#lets not recalc these more than we have to, but apparently len is O(1)
	lenOses = len(oses)
	lenBrowsers = len(browsers)
	lenRegions = len(regions)
	lenCities = len(cities)
	lenTags = len(allTags)
	lenSlotVisibilities = len(slotVisibilities)

	outArray = []
	# parse each element individually and add it to the array
	for exampleNum in xrange(0,len(array),27):
		if exampleNum % 2700000 == 0:
			print 'at example', exampleNum/27
		# print array[exampleNum: exampleNum + 27]
		example = []
		# parse timestamp first
		index = exampleNum + 4
		example.append(array[index][0:4]) #year
		example.append(array[index][4:6]) #month
		example.append(array[index][6:8]) #day
		example.append(array[index][8:10]) #hour
		example.append(array[index][10:12]) #min 
		example.append(array[index][12:14]) #sec
		example.append(array[index][14:17]) #ms

		index += 1 #logtype
		example.append(array[index]) 
		#8 features so far

		index += 2 #os/browsers

		os = array[index][0:array[index].index('_')]
		browser = array[index][array[index].index('_') + 1: len(array[index])]

		user_agent_oses = np.zeros(lenOses)
		user_agent_browsers = np.zeros(lenBrowsers)

		user_agent_oses[oses.index(os)] = 1
		user_agent_browsers[browsers.index(browser)] = 1

		example.extend(user_agent_oses)
		example.extend(user_agent_browsers)
		#23 features so far

		index += 1 #ip address

		address = array[index]
		example.append(address[0:address.index('.')])
		address = address[address.index('.') + 1: len(address)]	
		example.append(address[0:address.index('.')])
		address = address[address.index('.') + 1: len(address)]
		example.append(address[0:address.index('.')])
		#26 features so far

		index += 1 #region

		region = np.zeros(lenRegions)
		if int(array[index]) in regions:
			region[regions.index(int(array[index]))] = 1
		else:
			region[0] = 1

		example.extend(region)
		#61 features so far

		index += 1 #city, they also include regions as the regions are cities as well

		city = np.zeros(lenCities)
		if int(array[index]) in cities:
			city[cities.index(int(array[index]))] = 1
		else:
			city[0] = 1

		example.extend(city)
		#457 features so far

		index += 1 #adexchange

		example.append(array[index])

		index += 5 #slotwidth

		example.append(array[index])

		index += 1 #slotheight

		example.append(array[index])
		#460 features so far

		index += 1 #slotvisibility, this varies by advertiser apparently...
		if isinstance(array[index], (int, long)):
			example.append(array[index])
		else:
			slotvisibility = np.zeros(lenSlotVisibilities)
			if array[index] in slotVisibilities:
				slotvisibility[slotVisibilities.index(array[index])] = 1
			else:
				slotvisibility[0] = 1
			example.extend(slotvisibility)
		#467 features so far

		index += 1 #slotformat
		if isinstance(array[index], (int, long)):
			example.append(array[index])
		else:
			example.append(0)

		index += 1 #slotprice

		example.append(array[index])

		index += 2 #bidprice

		example.append(array[index])
		#470 features so far

		#dont use paying price because we dont know that, only if we think user will click, 464 features

		# index += 1

		# example.append(array[index])

		index += 4

		tag_indicators = np.zeros(lenTags)
		if array[index] == 'null':
			tag_indicators[0] = 1
		else:
			if isinstance(array[index], (int,long)):
				if array[index] in allTags:
					tag_indicators[allTags.index(array[index])] = 1
				else:
					tag_indicators[0] = 1
			else:
				tags = array[index].split(',')
				tags = [int(x) for x in tags]
				for tag in tags:
					if tag in allTags:
						tag_indicators[allTags.index(tag)] = 1
					else:
						tag_indicators[0] = 1
		example.extend(tag_indicators)
		#515 total features

		example = [int(x) for x in example]

		# example[example == 'null'] = 0

		example = np.array(example)

		example = example.astype(np.float32)

		outArray.append(example)
		# print example, len(example)

	# if isTestData == False:
	# 	f = open('data/inputMatrix'+ sys.argv[1] + '.pckl', 'w')
	# 	pickle.dump(np.array(outArray), f)
	# 	f.close()
		# return np.array(outArray)

	print 'starting mean normalization'
	def normalize(inArray):
		for i in range(0,len(inArray[0])):
			print 'mean normalizing var', i
			totalVal = 0
			minVal = inArray[0][i]
			maxVal = inArray[0][i]
			for j in range(0,len(inArray)):
				totalVal += inArray[j][i]
				if inArray[j][i] < minVal:
					minVal = inArray[j][i]
				if inArray[j][i] > maxVal:
					maxVal = inArray[j][i]
			deviation = maxVal - minVal
			average = float(totalVal)/(len(inArray))
			if deviation == 0:
				deviation = 1
			for j in range(0,len(inArray)):
				inArray[j][i] = (inArray[j][i] - minVal)/deviation
		return inArray

	outArray = normalize(outArray)  #when single threading

	# threads = 4
	# dividedVars = []
	# for i in xrange(0,threads):
	# 	varCollector = []
	# 	for example in outArray:
	# 		varCollector.append(example[i * (len(outArray[0]) / threads) : (i+1) * (len(outArray[0]) / threads)])
	# 	dividedVars.append(varCollector)

	# pool = ThreadPool(threads)

	# output = pool.map(normalize, dividedVars)
	# pool.close() 
	# pool.join() 

	# outArray = []
	# for i in xrange(0,len(output[0])):
	# 	varConcat = []
	# 	for j in xrange(0,len(output)):
	# 		varConcat.extend(output[j][i])	
	# 	outArray.append(varConcat)

	print 'mean normalization complete'

	print 'set size:', len(outArray)
	print 'click count:', trainingResults.count(1)

	#convert it into usable input arrays
	trainingResults = np.array(trainingResults)
	trainingResults = trainingResults.astype(np.int)

	outArray = np.array(outArray)

	trainingSet = (outArray, trainingResults)
	stop = timeit.default_timer();
	sec = stop - start
	print "Took ", "{:0>8}".format(datetime.timedelta(seconds=sec))
	return trainingSet

posFileDir = '/Users/karansamel/Documents/GitRepos/make-ipinyou-data/' + sys.argv[1] + '/train.pos.txt'
trainFileDir = '/Users/karansamel/Documents/GitRepos/make-ipinyou-data/' + sys.argv[1] + '/train.log.txt'
testFileDir = '/Users/karansamel/Documents/GitRepos/make-ipinyou-data/' + sys.argv[1] + '/test.log.txt'

trainfile = open(trainFileDir, 'r')
posFile = open(posFileDir, 'r')
trainData = getData(trainfile, False, False, nexamples = 15000) #3083056

testfile = open(testFileDir, 'r')
testData = getData(testfile, True, False) #3083056

testfile.close()
trainfile.close()

training_data = (trainData[0][0:int(len(trainData[0]) * .8)], trainData[1][0:int(len(trainData[1]) * .8)])
validation_data = (trainData[0][int(len(trainData[0]) * .8):], trainData[1][int(len(trainData[1]) * .8):])
test_data = testData

f = open('data/' + sys.argv[1] + 'PartialHalf.pckl', 'w')
pickle.dump([training_data, validation_data, test_data], f)
f.close()