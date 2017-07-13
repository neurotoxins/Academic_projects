import sys
from random import shuffle
import numpy as np

#Defining the range of K values used to implement cross-validation
kRange = [3,4,5];
kAccuracy = [];

#testData = testFile.read().splitlines();
trainData = np.load('lr_train_148064.npz');
testData = np.load('lr_test_16451.npz');


matchesTotal = 10000
train_x = trainData['X'][0:matchesTotal]
train_y = trainData['Y'][0:matchesTotal]

trainData = np.c_[train_x, train_y]

matchesTotal = 1000
test_x = testData['X'][0:matchesTotal]
test_y = testData['Y'][0:matchesTotal]

testData = np.c_[test_x, test_y]

result = '';

#Declaring a function to calculate the NN calculated result for every testData row input
def getNNResult(kResults):
	results = []
	for i, val in enumerate(kResults):
		resultVec = val[220]
		results.append(resultVec)
	
	#print results
	return max(set(results), key=results.count)

#Declaring a function that takes in a k, training data and test data to calculate the accuracy
def getAccuracy(k, testVals, trainVals):
	hammingVals = []
	kValue = k
	accuracy = 0
	for i, val in enumerate(testVals):
		#print i, val
		testVal = val

		hammingVal = []
		for j, tVal in enumerate(trainVals):
			hammingRes = []
			trainVal = tVal
			#resultVals = train_y
			
			#distance = len([k for k, l in zip(testVal, trainVal) if k == l])
			
			distance = np.sum(np.logical_and(testVal,trainVal))
			#trainVal.append(distance)
			trainVal = np.append(trainVal, distance)
			#hammingRes.append(resultVals[j])
			#hammingRes.append(distance)
			#resultVals[j].extend(distance)
			#hammingRes = hammingRes.append(trainVal[0])
			hammingVal.append(trainVal.tolist())

		hammingVals.append(hammingVal)

	#Calculating the accuracy for the list of testVals
	accuracyVec = []
	positiveCount = 0
	for i, val in enumerate(testVals):
		hammingList = []
		result = []
		hammingList = hammingVals[i]

		#shuffling the list
		shuffle(hammingList)

		#Sorting the list of lists
		hammingList.sort(key=lambda x: x[221], reverse=True)

		results = hammingList[:kValue]

		result = getNNResult(results)
		#testRes = val.split(',')[9]
		testRes = val[220]

		if result == testRes:
			accuracyVec.append(1)
			positiveCount = positiveCount + 1
		else:
			accuracyVec.append(0)
		#print result

	accuracy = float(positiveCount)/len(accuracyVec)

	#print accuracyVec, len(accuracyVec), accuracy, positiveCount
			
	return accuracy;

def getAvgAccuracyForK(k):
	accuracyValues = []
	kValue = k
	partitionStart = 0;
	partitionEnd = 0;
	partitionSize = round(len(trainData)/10);

	for j in range(0, 3):
		if j == 2:
			partitionEnd = len(trainData);
		else:
			partitionEnd = partitionEnd + partitionSize;


		cvTestData = trainData[int(partitionStart):int(partitionEnd)];

		cvTrainData = np.append(trainData[0:int(partitionStart)], trainData[int(partitionEnd):len(trainData)], axis=0);


		accuracyValues.append(getAccuracy(kValue, cvTestData, cvTrainData));

		partitionStart = partitionEnd;
	
	avgAcc =  reduce(lambda x, y: x + y, accuracyValues) / len(accuracyValues)
	#print accuracyValues, avgAcc
	return avgAcc

def getOptimalK():
	accuracies = []
	for k in kRange:
		accuracies.append(getAvgAccuracyForK(k));

	print 'The following are the accuracies for k-fold cross validation for every k'
	print accuracies
	return accuracies.index(max(accuracies))+1

print 'Calculating the NN accuracy ...'
#result = map(lambda x:x.split(','), testData);
finalK = getOptimalK()

finalAccuracy = getAccuracy(finalK, testData, trainData);

print 'The following are the optimal K and accuracy values with the given test data:'
print finalK, (finalAccuracy*100)
#print(result);
