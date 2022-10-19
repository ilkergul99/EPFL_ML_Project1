import numpy as np
from helper_functions_project1 import *

def least_squares(y, tx):
    """
    Least squares regression using normal equations
    Args:
        y: Given labels of data = (N,)
        tx: Features of the data = (N,D)
    Returns:
        w: optimized weight vector for the model
        loss: optimized final loss based on mean squared error
    """
    Gram_matrix = tx.T.dot(tx) #shape = (D,D)
    second_matrix = tx.T.dot(y) #shape = (D,1)
    w = np.linalg.solve(Gram_matrix, second_matrix) #shape = (D,1)
    loss = compute_loss_mse(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    Args:
        y: Given labels of data = (N,)
        tx: Features of the data = (N,D)
        lambda_: regularization parameter
    Returns:
        w: optimized weight vector for the model
        loss: optimized final loss based on mean squared error
    """
    ridge_matrix = tx.T.dot(tx) + 2 * y.shape[0] * lambda_ * np.identity(tx.shape[1]) #shape = (D,D)
    second_matrix = tx.T.dot(y) #shape = (D,1)
    w = np.linalg.solve(ridge_matrix, second_matrix) #shape = (D,1)
    loss = compute_loss_mse(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic Regression using Gradient Descent
    Args:
        y: labels
        tx: features
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: number of steps to run
        gamma: a scalar denoting the total number of iterations 
    Returns:
        w: optimized weight vector for the model
        loss: optimized final loss based on logistic loss
    """
    w = initial_w.copy()
    
    for _ in range(max_iters):
        # compute the gradient
        grad = negative_likelihood_grad(y, tx, w)
        # update weights
        w = w - gamma * grad
        
    return w, negative_likelihood_loss(y, tx, w)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized Logistic Regression using Gradient Descent 
    Args:
        y: Given labels of data = (N,)
        tx: Features of the data = (N,D)
        lambda_: regularization parameter
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: number of steps to run
        gamma: a scalar denoting the total number of iterations 
    Returns:
        w: optimized weight vector for the model
        loss: optimized final loss based on logistic loss
    """
    w = initial_w.copy()
    
    for _ in range(max_iters):
        # compute the gradient
        grad = reg_negative_likelihood_grad(y, tx, w, lambda_)
        # update weights
        w = w - gamma * grad
        
    return w, negative_likelihood_loss(y, tx, w)
    