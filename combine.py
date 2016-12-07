import pickle, sys
import numpy as np
f = open('data/' + sys.argv[1] + '.pckl')
training_data, validation_data, test_data = pickle.load(f)
f.close()

f = open('data/' + sys.argv[1] + 'testOutput.pckl')
test_predictions,FM_test = pickle.load(f)
f.close()

f = open('data/' + sys.argv[1] + 'trainOutput.pckl')
train_predictions,FM_train = pickle.load(f)
f.close()

# def normalize(inArray):
# 	for i in range(0,len(inArray[0])):
# 		print 'mean normalizing var', i
# 		totalVal = 0
# 		minVal = inArray[0][i]
# 		maxVal = inArray[0][i]
# 		for j in range(0,len(inArray)):
# 			totalVal += inArray[j][i]
# 			if inArray[j][i] < minVal:
# 				minVal = inArray[j][i]
# 			if inArray[j][i] > maxVal:
# 				maxVal = inArray[j][i]
# 		deviation = maxVal - minVal
# 		average = float(totalVal)/(len(inArray))
# 		if deviation == 0:
# 			deviation = 1
# 		for j in range(0,len(inArray)):
# 			inArray[j][i] = (inArray[j][i] - minVal)/deviation
# 	return inArray

# FM_test = normalize(FM_test)
# FM_train = normalize(FM_train)

new_test = []
for elementNum in xrange(0,len(test_data[0])):
	new_test.append(np.append(test_data[0][elementNum],FM_test[elementNum]))
new_test = np.asarray(new_test)

new_train = []
for elementNum in xrange(0,len(training_data[0])):
	new_train.append(np.append(training_data[0][elementNum],FM_train[elementNum]))
new_train = np.asarray(new_train)

f = open('data/FM' + sys.argv[1] + '.pckl', 'wb')
pickle.dump([(new_train, training_data[1]), (new_train, validation_data[1]), (new_test, test_data[1])], f)
f.close()
