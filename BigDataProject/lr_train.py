from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

def train(X, Y, samples):
    print 'Training using data from %d matches...' % samples
    return LogisticRegression().fit(X[0:samples], Y[0:samples])

def main():
    dataImported = np.load('lr_train_148064.npz')
    X_train = dataImported['X']
    Y_train = dataImported['Y']

    model = train(X_train, Y_train, len(X_train))

    print 'Model: ', model

    with open('lr_model.pkl', 'w') as output_file:
        pickle.dump(model, output_file)

if __name__ == "__main__":
    main()