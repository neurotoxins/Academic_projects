from __future__ import division
import random
import numpy as np
import glob
import math
from random import shuffle

#Initalizing the training and test data
# trainData = np.load('lr_train_148064.npz');
# testData = np.load('lr_test_16451.npz');

testFile = open('./test_libsvm');
trainFile = open('./train_libsvm');

trainData = trainFile.read().splitlines();
testData = testFile.read().splitlines();

trainFile.close();
testFile.close();

matchesTotal = 20000
trainData = trainData[0:matchesTotal]

matchesTotal = 15000
testData = testData[0:matchesTotal]


learningRateInitial = 0.02;
hyperParameter = 1;
epochs = 10;
marginVal = 0.01;

#weightVector = [0.5]*11;

#Method to return a list of randomly generated values between -1 and 1 for initial weight vector
def getInitWeightVec(val):
	return [0 for _ in range(0,val)];

#Defining the function for perceptron and margin perceptron to get the final weight Vector
def computeWeightVector(data, astro, length):
	tCounter = 0;
	weightVec = getInitWeightVec(length);

	for e in range(0, epochs):
		shuffle(data);
		for i, val in enumerate(data):
			features = val.split(' ');

			sign = features.pop(0);

			if astro:
				if int(sign) == 0:
					sign = -1;

			featureVec = [0]*length;

			for j, fVal in enumerate(features):
				feature = fVal.split(':');
				if astro:
					featureVec[int(feature[0])-1] = float(feature[1]);
				else:
					featureVec[int(feature[0])] = float(feature[1]);

			result = np.dot(weightVec, featureVec);

			product = result * int(sign);
			
			if product <= 1:
				gradient = [wt-ft for wt, ft in zip(weightVec, (int(sign)*hyperParameter*np.array(featureVec)))];
			else:
				gradient = weightVec;

			learningRate = learningRateInitial/(1+(learningRateInitial*tCounter)/ hyperParameter);

			tCounter += 1;

			weightVec = [wt-ft for wt, ft in zip(weightVec, (learningRate*np.array(gradient)))];

	return {'weightVector': weightVec};

#Defining the function for testing the final weight vector against the corresponsing test data
def getAccuracy(data, astro, weightVec):
	mistakes = 0;
	print 'in get accuracy'
	for i, val in enumerate(data):
		features = val.split(' ');
		sign = features.pop(0);

		if astro:
			if int(sign) == 0:
				sign = -1;

		featureVec = [0]*len(weightVec);

		for j, fVal in enumerate(features):
			feature = fVal.split(':');
			if astro:
				featureVec[int(feature[0])-1] = float(feature[1]);
			else:
				featureVec[int(feature[0])] = float(feature[1]);

		result = np.dot(weightVec, featureVec);

		product = result * int(sign);

		if product < 0:
			mistakes += 1;

	accuracy = (float(len(data)-mistakes)/len(data))*100;
	return accuracy;


#Defining a method to compute the margin among all the test features with the given feature vector.
def computeMargin(data, astro, weightVec):
	margins = [];
	weightVecNorm = 0;

	for i, wVal in enumerate(weightVec):
		weightVecNorm = weightVecNorm + (wVal*wVal);

	weightVecNorm = math.sqrt(weightVecNorm);

	for i, val in enumerate(data):
		features = val.split(' ');
		sign = features.pop(0);

		if astro:
			if int(sign) == 0:
				sign = -1;

		featureVec = [0]*len(weightVec);

		for j, fVal in enumerate(features):
			feature = fVal.split(':');
			if astro:
				featureVec[int(feature[0])-1] = float(feature[1]);
			else:
				featureVec[int(feature[0])] = float(feature[1]);

		result = np.dot(weightVec, featureVec);

		margins.append(abs(result)/weightVecNorm);

	return min(margins);


def crossValidation(data, astro, length):
	learningRates = [0.001, 0.01, 0.1, 1];
	hyperParameters = [0.01, 0.1, 1, 10, 100];

	index = 0;
	parameters = [];
	accuracies = [];
	indexOrder = [];
	partitionSize = round(len(data)/10);

	for i, lVal in enumerate(learningRates):
		for j, hVal in enumerate(hyperParameters):
			parameters.append(str(lVal)+':'+str(hVal));
			index += 1;

	#Iterating through every combination of learning rates and hyper parameters
	for i, pVal in enumerate(parameters):
		partitionStart = 0;
		partitionEnd = 0;
		avgAccuracy = 0;

		global learningRateInitial;
		global hyperParameter;

		#10 iterations for 10-cross validation
		for j in range(0, 10):
			if j == 9:
				partitionEnd = len(data);
			else:
				partitionEnd = partitionEnd + partitionSize;

			cvTestData = data[int(partitionStart):int(partitionEnd)];
			cvTrainData = data[0:int(partitionStart)] + data[int(partitionEnd):len(data)];

			parameterVals = pVal.split(':');
			learningRateInitial = float(parameterVals[0]);
			hyperParameter = float(parameterVals[1]);

			#Computing the weight vectors for this set of parameters and finding the accuracies on the test data
			sgdResults = computeWeightVector(cvTrainData, astro, length);
			avgAccuracy = avgAccuracy + getAccuracy(cvTestData, astro, sgdResults['weightVector']);

			partitionStart = partitionEnd;

		accuracies.append(avgAccuracy);

	indexOrder = sorted(range(len(accuracies)), key=lambda x: accuracies[x], reverse=True)[0:5]

	print '5 best parameters with their corresponding avg. cross validation accuracy are:';
	for i, iVal in enumerate(indexOrder):
		params = parameters[iVal].split(':');
		print 'Learnin rate, hyper parameter, avg. accuracy :', params[0], params[1], accuracies[iVal]/10;

	maxAccuracyIndex = accuracies.index(max(accuracies));
	test = indexOrder[0:1];

	return parameters[maxAccuracyIndex];


#Defining the main method for all function calls
def main():
	maxOriginalDist = 0;

	accuracyFinal = 0;
	marginVal = 0;

	global learningRateInitial;
	global hyperParameter;
	global epochs;

	'''Uncomment the following lines to enable the cross validation'''

	#Computing weight vectors using cross validation.
	# print 'Computing cross validation on Train data ... ';
	# optimalParamsOrig = crossValidation(trainData, True, 220);
	# print 'Optimal Params for Original Train (LearningRate:HyperParameter):', optimalParamsOrig;

	# #Use the optimal values of Learning rate and Hyper Parameter for the test data.
	# #Resetting the epochs
	# epochs = 10;

	# #Computing for Original Data
	# paramVals = optimalParamsOrig.split(':');
	# learningRateInitial = float(paramVals[0]);
	# hyperParameter = float(paramVals[1]);

	sgdResults = computeWeightVector(trainData, True, 220);
	accuracyFinal = getAccuracy(trainData, True, sgdResults['weightVector']);
	marginVal = computeMargin(testData, True, sgdResults['weightVector']);

	print 'The Accuracy and Margin for Original Data:', accuracyFinal, marginVal;


main();

