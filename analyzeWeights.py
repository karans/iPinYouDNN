import pickle
import theano

f = open('layers/1458')
layers = pickle.load(f)
f.close()

weights = layers[0].w.get_value()

weightValues = []
for nodes in xrange(0, len(weights)):
	weightValues.append(sum(weights[nodes]))

topWeights = sorted(weightValues, reverse = True)

indicies = []
for i in xrange(0,15):
	indicies.append(weightValues.index(topWeights[i]))

features = ['year', 'month', 'day', 'hour', 'min', 'sec', 'ms', 'logtype', "windows", "ios", "mac", "android", "linux", "other", "chrome", "sogou", "maxthon", "safari", "firefox", "theworld", "opera", "ie", "other", 'IP_1', 'IP_2', 'IP_3', 
0 , 1 , 2 , 3 , 15 , 27 , 40 , 55 , 65 , 79 , 80 , 94 , 106 , 124 , 134 , 146 , 164 , 183 , 201 , 216 , 238 , 253 , 275 , 276 , 298 , 308 , 325 , 333 , 344 , 359 , 368 , 374 , 393 , 394 , 395, 

'adexchange', 'slotwidth', 'slotheight', 'slotvisibility', 'slotformat', 'slotprice', 'bidprice', 0,10006 , 10024 , 10031 , 10048 , 10052 , 10057 , 10059 , 10063 , 10067 , 10074 , 10075 , 10076 , 10077 , 10079 , 10083 , 10093 , 10102 , 10684 , 11092 , 11278 , 11379 , 11423 , 11512 , 11576 , 11632 , 11680 , 11724 , 11944 , 13042 , 13403 , 13496 , 13678 , 13776 , 13800 , 13866 , 13874 , 14273 , 16593 , 16617 , 16661 , 16706 , 16751 , 10110 , 10111]

FM = range(0,201)
FM = [str(str(float(x)/200) + ' percent') for x in FM]
features.extend(FM)

for i in xrange(0,len(indicies)):
	print features[indicies[i]]
