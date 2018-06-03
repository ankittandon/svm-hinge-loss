import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import sklearn.preprocessing
import sklearn
import linear_svm_square_loss

# Use real-world spam dataset
spam = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data', sep=' ', header=None)
test_indicator = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.traintest', sep=' ',header=None)
features = np.asarray(spam)[:, 0:-1]
labels = np.asarray(spam)[:, -1]*2 - 1 

X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=0)

# Standardize the data
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# try with lambda = 1
lambduh = 1
n = np.size(X_train, 1)
t_init = 1/(scipy.linalg.eigh(1/len(y_train)*X_train.T.dot(X_train), eigvals=(n-1, n-1), eigvals_only=True)[0]+lambduh)
graddescent_betas = linear_svm_square_loss.mylinearsvm(lambduh, t_init, 100, X_train, y_train)

# Calculate misclassifcation error on training set and testing set
error_train = linear_svm_square_loss.compute_misclassification_error(betas[-1,:],X_train,y_train)
error_test = linear_svm_square_loss.compute_misclassification_error(betas[-1,:],X_test,y_test)
print('Misclassification for training set (lambda = 1): ', error_train)
print('Misclassification error test set (lambda = 1): ', error_test)

# Plot miscalssification error and objective value
linear_svm_square_loss.plot_misclassification_error(betas, X_train, y_train, title='Training set misclassification error when lambda = 1', file_name = 'misclass_plot_train_lambda1.png')
linear_svm_square_loss.plot_misclassification_error(betas, X_test, y_test, title='Test set misclassification error when lambda = 1', file_name = 'misclass_plot_test_lambda1.png')
linear_svm_square_loss.plot_objective(betas, lambduh, X_train, y_train, file_name = 'objective_plot_train_lambda1.png')
 
# Find optimal value of lambda through cross-validation
optimal_lambduh2 = linear_svm_square_loss.find_optimal_lambduh(X_test,y_test,t_init,-100,100,0.1)
print('Optimal value of lambda is: ', optimal_lambduh2)

# Calculate misclassifcation error on training set and testing set
betas_opt = linear_svm_square_loss.mylinearsvm(optimal_lambduh2, t_init, 100, X_train, y_train)
error_train_lambda_opt = linear_svm_square_loss.compute_misclassification_error(betas_opt[-1,:],X_train,y_train)
error_test_lambda_opt = linear_svm_square_loss.compute_misclassification_error(betas_opt[-1,:],X_test,y_test)
print('Misclassification error for the optimal lambda value on the training set is: ', error_train_lambda_opt)
print('Misclassification error for the optimal lambda value on the testing set is: ', error_test_lambda_opt)

# Plot miscalssification error and objective value
linear_svm_square_loss.plot_misclassification_error(betas_opt, X_train, y_train, title='Training set misclassification error for the optimal lambda value.', file_name = 'misclass_plot_train_lambda_opt.png')
linear_svm_square_loss.plot_misclassification_error(betas_opt, X_test, y_test, title='Test set misclassification error for the optimal lambda value.', file_name = 'misclass_plot_test_lambda_opt.png')
linear_svm_square_loss.plot_objective(betas_opt, optimal_lambduh2, X_train, y_train, file_name = 'objective_plot_train_lambda_opt.png')