import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import sklearn.preprocessing


def computegrad(beta, l, x=X_train, y=y_train):
    """Computes gradient.

    Parameters
    ----------
    beta: beta parameter

    l: lambda parameter

    x: features

    y: labels

    Returns
    -------
    gradient of objective function
    """
    xybeta = (y[:,np.newaxis]*x).dot(beta)
    a = -2/len(y)*np.maximum(0,1-xybeta).dot(y[:,np.newaxis]*x)
    b = 2*l*beta
    return a+b


def obj(beta, l, x=X_train, y=y_train):
    """Computes objective value.

    Parameters
    ----------
    beta: beta parameter

    l: lambda parameter

    x: features

    y: labels

    Returns
    -------
    objective value
    """
    a = y[:,np.newaxis]*x
    b = a.dot(beta)
    c = 1/len(y)*(np.sum(np.maximum(0,1-b)**2))
    d = l * np.linalg.norm(beta)**2
    return (c+d)


def backtracking(beta, l, alpha=0.5, b=0.8, maxiteration=100, t = 1, x=X_train, y=y_train):
    """Updates the step size using backtracking algorithm.

    Parameters
    ----------
    beta: beta parameter

    l: lambda parameter

    x: features

    y: labels

    Returns
    -------
    updated step size
    """
    F_gradient = computegrad(beta, l)
    F_gradient_norm_squared = np.linalg.norm(F_gradient)**2
    iteration = 0
    found_t = False
    while iteration < maxiteration and (not found_t):
        if iteration==maxiteration:
            print("max iterations reached")
        if obj(beta - t * F_gradient, l) < (obj(beta,l) - alpha * t * F_gradient_norm_squared):
            found_t = True
        else:
            #print("changing eta")
            t *= b
            iteration+=1
    return t

def mylinearsvm(x,y,l,maxiter,beta_init,theta_init,t_init):
    """ Linear Support Vector Machine with fast gradient method
    Parameters
    ----------
    l: lambda parameter
    t_init: initial step size
    maxiter: maximum number of iterations
    x: features
    y: labels
    beta_init, theta_init: initial betas array - can be 0's or random
    Returns
    -------
    betavals,thetavals: beta and theta values of all iterations
    """    
    beta = beta_init
    theta = theta_init
    t_new = t_init
    F_gradient_theta = computegrad(theta,l)
    iteration = 0
    maxiteration = 100
    betavals = list()
    thetavals = list()
    while iteration < maxiteration:
        t_new = backtracking(beta = beta, l=l, t=t_new, alpha = 0.5, b = 0.8, maxiteration = maxiteration)
        beta_new = theta - t_new * F_gradient_theta
        theta = beta_new + iteration/(iteration+3)*(beta_new-beta)
        betavals.append(beta_new)
        thetavals.append(theta)
        F_gradient = computegrad(theta,l)
        beta = beta_new
        iteration+=1
        #if iteration%10 == 0:
         #   print(iteration)
    return betavals, thetavals


def compute_misclassification_error(beta_opt, x, y):
    """ Compute misclassication error
    Parameters
    ----------
    beta_opt: beta values

    x: features

    y: labels

    Returns
    -------
    Fraction of misclassication
    """
    y_pred = x.dot(beta_opt) > 0.5
    y_pred = y_pred*2 - 1  # Convert to +/- 1, same as logistic regression
    return np.mean(y_pred != y)