import cPickle as pickle
import numpy as np
import sys 

filename = 'inputMatrix' + sys.argv[1]
f = open('data/' + str(filename) + '.pckl')
data = pickle.load(f)
f.close()

cov = np.cov(data.T)
for i in xrange (0,len(cov)):
	cov[i][i] = 0
cov = np.absolute(cov)

inputMatrix = np.zeros((5,103))

kernelSize = 3
row = 0
col = 0
i = 0
added = [] #list of attributes already added to our input matrix
while col < 103: #for the first 3 rows

	rowIndex = np.unravel_index(np.argmax(cov > 0), cov.shape)[0]
	colIndex = np.unravel_index(np.argmax(cov > 0), cov.shape)[1]

	if rowIndex == 0 and colIndex == 0:
		break

	cov[rowIndex][colIndex] = 0
	cov[colIndex][rowIndex] = 0

	if rowIndex not in added:
		added.append(rowIndex)
		print 'added attribute', rowIndex
		inputMatrix[row][col] = rowIndex
		row += 1

	if row == 3:
		row = 0
		col += 1

	if colIndex not in added:
		print 'added attribute', colIndex
		added.append(colIndex)
		inputMatrix[row][col] = colIndex
		row += 1

	if row == 3:
		row = 0
		col += 1	
col = 0
row = 3
while col < 103: #for the first 3 rows

	rowIndex = np.unravel_index(np.argmax(cov > 0), cov.shape)[0]
	colIndex = np.unravel_index(np.argmax(cov > 0), cov.shape)[1]

	if rowIndex == 0 and colIndex == 0:
		break

	cov[rowIndex][colIndex] = 0
	cov[colIndex][rowIndex] = 0

	if rowIndex not in added:
		print 'added attribute', rowIndex
		added.append(rowIndex)
		inputMatrix[row][col] = rowIndex
		row += 1

	if row == 5:
		row = 3
		col += 1

	if colIndex not in added:
		print 'added attribute', colIndex
		added.append(colIndex)
		inputMatrix[row][col] = colIndex
		row += 1

	if row == 5:
		row = 3
		col += 1	


inputArray = inputMatrix.reshape((1,515)).flatten()

missA = range(0,515)
missA = set(missA) - set(added)
missA = list(missA)

missIndex = np.argwhere(inputArray == 0).flatten()

for i in xrange(0,len(missIndex)):
	inputArray[missIndex[i]] = missA[i]

print inputArray
