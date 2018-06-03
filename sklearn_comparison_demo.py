import numpy as np
import pandas as pd
import scipy.linalg
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.grid_search import GridSearchCV
import linear_svm_square_loss

def compare(x,y):
	"""Compare the results of Sklearn svm with my custom implementation

    Parameters
    ----------
    x: features
    y: labels

    Returns
    -------
    error_train: miscalssification error on training set (with my implementation)
    error_test: miscalssification error on test set (with my implementation)
    error_train_svm: miscalssification error of sklearn's function on training set
    error_test_svm: miscalssification error of sklearn's function on testing set
    """
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
	scaler = sklearn.preprocessing.StandardScaler()
	scaler.fit(x_train)
	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)
	n = np.size(x_train, 1)
	eta_init = 1/(scipy.linalg.eigh(1/len(y_train)*x_train.T.dot(x_train), eigvals=(n-1, n-1), eigvals_only=True)[0]+lambduh)
	optimal_lambduh2 = linear_svm_square_loss.find_optimal_lambduh(x_test,y_test,eta_init,-10,10,0.1)
	betas_opt = linear_svm_square_loss.mylinearsvm(optimal_lambduh2, eta_init, 100, x_train, y_train)
	error_train = linear_svm_square_loss.compute_misclassification_error(betas_opt[-1,:],x_train,y_train)
	error_test = linear_svm_square_loss.compute_misclassification_error(betas_opt[-1,:],x_test,y_test)
	print('misclassification error for the training set using my implementation is: ', error_train_lambda_opt)
	print('misclassification error for the testing set using my implementation is: ', error_test_lambda_opt)

	svm_l2 = svm.LinearSVC(penalty='l2', loss='squared_hinge')
	parameters = {'C':[10**i for i in range(-2, 2)]}
	clf_svm = GridSearchCV(svm_l2, parameters)
	clf_svm.fit(x_train, y_train)
	error_train_svm = 1 - clf_svm.score(x_train, y_train)
	error_test_svm = 1 - clf_svm.score(x_test, y_test)
	print('misclassification error for the training set using sklearn is: ', error_train_svm)
	print('misclassification error for the testing set using sklearn is: ', error_test_svm)

	return error_train, error_test, error_train_svm, error_test_svm