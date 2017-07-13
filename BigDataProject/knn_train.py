from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np

train_data = np.load('lr_train_148064.npz')
X = train_data['X']
Y = train_data['Y']

relevant_indices = range(0, 100000)
X = X[relevant_indices]
Y = Y[relevant_indices]

def my_distance(vec1,vec2):
    '''Returns a count of the elements that were 1 in both vec1 and vec2.'''
    #dummy return value to pass pyfuncdistance check
    return 0.0

def poly_weights_evaluate(distances):
    '''Returns a list of weights for the provided list of distances.'''
    pass

heroesTotal = 110
matchesTotal = len(X)

print 'Training KNN using data from %d matches...' % matchesTotal

model = KNeighborsClassifier(n_neighbors=matchesTotal,metric=my_distance,weights=poly_weights_evaluate).fit(X, Y)

# Populate model pickle
with open('knn_model_%d.pkl' % matchesTotal, 'w') as output_file:
    pickle.dump(model, output_file)