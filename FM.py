import pywFM, sys
import cPickle as pickle
import numpy as np
import itertools
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.sparse import *
from scipy import *

print 'parsing',sys.argv[1]
trainFile = open('/Users/karansamel/Documents/GitRepos/make-ipinyou-data/' + sys.argv[1] + '/train.partial.txt', 'r')
trainString = trainFile.readlines()
trainData = ''.join(str(x) for x in trainString)
trainArray = trainData.split()
labels = trainArray[1:27]
trainArray = trainArray[27:]

#shuffle

partition = [trainArray[n:n+27] for n in range(0, len(trainArray),27)]
np.random.shuffle(partition)
trainArray = list(itertools.chain.from_iterable(partition))

yTrain = [trainArray[i] for i in xrange(0, len(trainArray), 27)]
yTrain = map(int, yTrain)

split = [trainArray[i+1:i+27] for i in xrange(0, len(trainArray), 27)]

v = DictVectorizer()
dictionary = []
for i in xrange(0, len(split)):
    dictionary.append(dict(zip(labels, split[i])))

trainEncoded = v.fit_transform(dictionary)

print 'trainEncoded takes', float(sys.getsizeof(trainEncoded) * sys.getsizeof(trainEncoded[0]))/1000000, 'mb'
print 'yTrain takes', float(sys.getsizeof(yTrain))/1000000, 'mb'


#Parse testing data
testFile = open('/Users/karansamel/Documents/GitRepos/make-ipinyou-data/' + sys.argv[1] + '/test.log.txt', 'r')
testString = testFile.readlines()
testData = ''.join(str(x) for x in testString)
testArray = testData.split()
del testArray[27::29]
del testArray[27::28]

labels = testArray[1:27]
testArray = testArray[27:]

yTest = [testArray[i] for i in xrange(0, len(testArray), 27)]
yTest = map(int, yTest)

split = [testArray[i+1:i+27] for i in xrange(0, len(testArray), 27)]

v = DictVectorizer()
dictionary = []
for i in xrange(0, len(split)):
    dictionary.append(dict(zip(labels, split[i])))
testEncoded = v.fit_transform(dictionary)

print 'testEncoded takes', float(sys.getsizeof(testEncoded) * sys.getsizeof(testEncoded[0]))/1000000, 'mb'
print 'yTest takes', float(sys.getsizeof(yTest))/1000000, 'mb'

# f = open('encodedData1458.pckl', 'wb')
# pickle.dump([trainEncoded, yTrain, testEncoded, yTest], f)
# f.close()


# f = open('encodedData1458.pckl', 'rb')
# trainEncoded, yTrain, testEncoded, yTest = pickle.load(f)
# f.close()
# print 'prepared data'


bestAccuracy = 0
bestROCAUC = 0
# for iterations in xrange(0,100):
fm = pywFM.FM(task='classification', num_iter=150, init_stdev = .15, learning_method='mcmc', k2 = 10, learn_rate = .1, r0_regularization = 0,r1_regularization = 0,r2_regularization = 10, temp_path='temp_files/')
model = fm.run(trainEncoded,yTrain,trainEncoded,yTrain)
# print 'Model built with k2', k2size
print sorted(model.predictions)[-10:]

# for j in xrange(1,50,1):
#     def classify(prediction):
#         if prediction >= float(j)/50:
#             return 1
#         else:
#             return 0
#     y_pred = model.predictions
#     y_pred = map(classify, y_pred)
#     # print 'true ys', y_true
#     # print 'raw predcitons', model.predictions
#     # print 'predicted ys', y_pred
#     # print accuracy_score(yTest, y_pred), 'accuracy using threshold', float(j)/50
#     # print roc_auc_score(yTest, y_pred), 'ROC AUC using threshold', float(j)/50, '\n'
#     if accuracy_score(yTest, y_pred) > bestAccuracy:
#         bestAccuracy = accuracy_score(yTest, y_pred)
#     if roc_auc_score(yTest, y_pred) > bestROCAUC:
#         bestROCAUC = roc_auc_score(yTest, y_pred)
# print bestAccuracy, 'accuracy at'
# print bestROCAUC, 'ROC AUC at'
testOutputs = []
for i in xrange(0, len(model.predictions)):
    example = np.zeros(201)
    example[int(model.predictions[i] * 200)] = 1
    example = np.array(example, dtype='f')
    testOutputs.append(example)


f = open('data/' + sys.argv[1] + 'trainOutput.pckl', 'wb')
pickle.dump([model.predictions,testOutputs], f)
f.close()

modelTest = fm.run(trainEncoded,yTrain,testEncoded,yTest)
testOutputs = []
for i in xrange(0, len(modelTest.predictions)):
    example = np.zeros(201)
    example[int(modelTest.predictions[i] * 200)] = 1
    example = np.array(example, dtype='f')
    testOutputs.append(example)
f = open('data/' + sys.argv[1] + 'testOutput.pckl', 'wb')
pickle.dump([modelTest.predictions,testOutputs], f)
f.close()

# # you can also get the model weights
# print 'global bias', model.global_bias
# print 'linear weights', model.weights
# print 'pairwise weights', model.pairwise_interactions

