import numpy as np
from helper_functions_project1 import *
"""""""""""""""""""""""""""load data start"""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#height, weight, gender = load_data()
#x, mean_x, std_x = standardize(height)
#y, tx = build_model_data(x, weight)
x,y = load_data("train.csv")
x = standardize(x)[0]
tx = np.c_[np.ones(len(y)), x]

"""""""""""""""""""""""""""load data end"""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""performs gradient descent"""
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm. 
    Args:
        y: Given labels of data = (N,)
        tx: Features of the data = (N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        loss[-1]: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws[-1]: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of GD 
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        #compute loss and gradient
        grad = compute_gradient(y,tx,w)
        loss = compute_loss_mse(y,tx,w)
        
        #update the weights
        w = w - gamma*grad
        ws.append(w)
        losses.append(loss)
        print("GD iter. {bi}/{ti}: loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))  
    return losses, ws

"""performs stochastic gradient descent"""
def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for yn,txn in batch_iter(y,tx,batch_size,1):
            grad, e = compute_stoch_gradient(yn, txn, w)
            w = w - gamma*grad
            loss = compute_mse(e)
            ws.append(w)
            losses.append(loss)
            
        print("SGD iter. {bi}/{ti}: loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss))        
    return losses, ws

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
        y: Given labels of data = (N,)
        tx: Features of the data = (N,D)
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
    