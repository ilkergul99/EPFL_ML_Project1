import numpy as np

def sigmoid(t):
    """apply the sigmoid function on t"""
    return 1 / (1 + np.exp(-t))

def negative_likelihood_loss(y, tx, w):
    """returns loss"""
    t = np.dot(tx, w)
    return np.sum(np.log(1 + np.exp(t)) - y * t) 

def negative_likelihood_grad(y, tx, w):
    """returns grad"""
    return np.dot(tx.T, sigmoid(np.dot(tx, w)) - y)

def reg_negative_likelihood_grad(y, tx, w, lambda_):
    """returns grad with regularization term for l2"""
    return np.dot(tx.T, sigmoid(np.dot(tx, w)) - y) + 2 * lambda_ * w

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w.copy()
    
    for _ in range(max_iters):
        # compute the gradient
        grad = negative_likelihood_grad(y, tx, w)
        # update weights
        w = w - gamma * grad
        
    return w, negative_likelihood_loss(y, tx, w)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w.copy()
    
    for _ in range(max_iters):
        # compute the gradient
        grad = reg_negative_likelihood_grad(y, tx, w, lambda_)
        # update weights
        w = w - gamma * grad
        
    return w, negative_likelihood_loss(y, tx, w)
    