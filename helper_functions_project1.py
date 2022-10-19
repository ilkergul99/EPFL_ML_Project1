import numpy as np

def compute_loss_mse(y, tx, w):

    """Calculate the loss using either MSE or MAE.

    Args:
        y: Given labels of data = (N,)
        tx: Features of the data = (N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    error = y - tx.dot(w)
    return np.mean(error ** 2)/2

def compute_loss_mae(y, tx, w):

    """Calculate the loss using either MSE or MAE.

    Args:
        y: Given labels of data = (N,)
        tx: Features of the data = (N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    error = y - tx.dot(w)
    return np.mean(np.absolute(error))/2

def compute_gradient(y, tx, w):
    """Computes the gradient at w.
        
    Args:
        y: Given labels of data = (N,)
        tx: Features of the data = (N,D)
        w: numpy array of shape=(D,). The vector of model parameters.
        
    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    error = y - tx.dot(w)
    return -1/y.shape[0]*tx.T.dot(error)/error.size

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.
        
    Args:
        y: Given labels of data = (N,)
        tx: Features of the data = (N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of GD 
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    convergence = 1e-10
    w = initial_w
    for useless in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss_mse(y, tx, w)
        w = w - gamma*gradient 
        # store w and loss
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < convergence:
            break
    return ws[-1], losses[-1]

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