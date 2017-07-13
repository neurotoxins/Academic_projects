
The training and test sample data are imported from the locan mongo db using the following python script,

importData.py - this compresses the features in the form of compressed np matrix with the extension '.npz'. This produces the following two test and train files which could be loaded in python using np.load().

lr_train_148064.npz
lr_test_16451.npz

The digits signify the number of records in the dataset. To access the feature matrix for the test, you can use np.load('./lr_test_16451.npz')['X'] and 'Y' for the corresponding results.

---------------

The following file is used to convert the current data from npz to libsvm format for using in my programs.

npztoLibsvm.py - This converts the npz to produce the following two files, 

test_libsvm
train_libsvm

----------------

Logistic Regression:-

load matlabworkspace.mat file first in matlab to get the environment ready. This will make test_X, train_X, test_labels and train_labels to load.

Next run the BigDataLogisticRegression with these 4 parameters in order defined in function definition.

This will give you y-values which are the accuracies of different sizes of training data.

To run the standard logistic regression from the skikit libraries,

lr_train.py - execute this to load the training data and produce a serialized object of the logistic regression model named lr_model.pkl

lr_test.py - execute this to load all the test data and obtain the precision, recall, F1 and support scores from the library.

plotAccuracies.py - execute this to divide the training data into number of chunks of your choice (default 100) and plot the training and test accuracies using the standard library. 

-----------------

Stochastic Gradient Descent:-

SGDLearner.py - execute this to run the stochastic gradient descent for the default config over 20000 training and 15000 test examples.

If you want to change the number of training examples, you can change the matchesTotal variable before training to set the number. (larger training set takes more time)

To enable cross validation, uncomment the code in the main() method and the program finds the optimal values of parameters and uses them to get the final accuracies of the test data. 

-------------------

K Nearest Neighbours:-

nearestNeighbours.py - execute this to run the KNN with k-fold cross validation over the default set of parameters. (over 10000 training and 1000 test examples).

You can change the number of cross validations and values of k from the code (k -[] and the range number for the for loop in getAvgAccuracyForK() method).


To run the standard skikit knn method,

knn_train.py - execute this to load the training data and store the knn model for the number of relevant records (by changing the range of relevant_indices variable).

knn_test.py - execute this to load all the test data and run it over all the training data to obtain the accuracies. You can change the training data by replacing the 'knn_model_*.pkl' with the one produced from knn_train.py.

knn_cross_validation - execute this to perform the cross validation on the test and train data.

The above three programs take quite a bit of time to execute. 
