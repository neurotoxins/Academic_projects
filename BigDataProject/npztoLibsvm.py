from __future__ import division
import numpy as np

testing_data = np.load('lr_test_16451.npz')
training_data = np.load('lr_train_148064.npz')

def createLibSVM(data, fileName):
	output = open( fileName, 'wb' )

	for i, match in enumerate(data['X']):
		line = ''
		line = line + str(data['Y'][i])

		for j, hero in enumerate(match):
			if int(hero) == 1:
				line = line + ' ' + "%s:%s" % ( j+1 , hero )


		if(i+1 < len(data['X'])):
			line = line  + '\n'

		output.write( line )


createLibSVM(testing_data, 'test_libsvm')
createLibSVM(training_data, 'train_libsvm')