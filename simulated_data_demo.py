import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import linear_svm_squared_hinge_loss

# Simulate a dataset with  100 obervations and 50 features
features = np.zeros((100, 50))
features[0:50, :] = np.random.normal(scale=1, size=(50, 50))
features[50:100, :] = np.random.normal(loc=1, scale=5, size=(50, 50))
labels = np.asarray([1]*50 + [-1]*50)

X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=0)

# Standardize the data
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

#try with lambda = 1
lambduh = 1
n = np.size(X_train, 1)
t_init = 1/(scipy.linalg.eigh(1/len(y_train)*X_train.T.dot(X_train), eigvals=(n-1, n-1), eigvals_only=True)[0]+lambduh)
graddescent_betas = linear_svm_square_loss.mylinearsvm(lambduh, t_init, 100, X_train, y_train)

# Calculate misclassifcation error on training set and testing set
error_train_lambda1 = linear_svm_square_loss.compute_misclassification_error(betas[-1,:],X_train,y_train)
error_test_lambda1 = linear_svm_square_loss.compute_misclassification_error(betas[-1,:],X_test,y_test)
print('Misclassification error for the lambda value 1 on the training set is: ', error_train_lambda1)
print('Misclassification error for the lambda value 1 on the testing set is: ', error_test_lambda1)

# Plot miscalssification error and objective value
linear_svm_square_loss.plot_misclassification_error(betas, X_train, y_train,
                             title='Training set misclassification error when lambda = 1',
                             file_name = 'misclass_plot_train_lambda1.png')
linear_svm_square_loss.plot_misclassification_error(betas, X_test, y_test,
                             title='Test set misclassification error when lambda = 1',
                             file_name = 'misclass_plot_test_lambda1.png')
linear_svm_square_loss.plot_objective(betas, lambduh, X_train, y_train, file_name = 'objective_plot_train_lambda1.png' )
 

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
linear_svm_square_loss.plot_misclassification_error(betas_opt, X_train, y_train,
                             title='Training set misclassification error for the optimal lambda value.',
                             file_name = 'misclass_plot_train_lambda_opt.png')
linear_svm_square_loss.plot_misclassification_error(betas_opt, X_test, y_test,
                             title='Test set misclassification error for the optimal lambda value.',
                             file_name = 'misclass_plot_test_lambda_opt.png')
linear_svm_square_loss.plot_objective(betas_opt, optimal_lambduh2, X_train, y_train, file_name = 'objective_plot_train_lambda_opt.png')