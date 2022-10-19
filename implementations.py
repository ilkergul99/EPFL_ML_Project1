import numpy as np

"""""""""""""""""""""""""""load data start"""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def load_data(path):
    #alternative read at once and then split into x and y
    X = np.genfromtxt(
    path, delimiter=",", skip_header=1, usecols=range(2,32))
    Y = np.genfromtxt(
    path, delimiter=",", skip_header=1, converters={1: lambda x: 0 if b's' in x else 1}, usecols=[1])
    return X,Y

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

#height, weight, gender = load_data()
#x, mean_x, std_x = standardize(height)
#y, tx = build_model_data(x, weight)
x,y = load_data("train.csv")
x = standardize(x)[0]
tx = np.c_[np.ones(len(y)), x]

"""""""""""""""""""""""""""load data end"""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""returns the mean squared error"""
def compute_mse(e):
    sq = np.square(e)
    error = 0.5*np.mean(sq)
    return error

"""returns the gradient"""
def compute_gradient(y, tx, w):
    e = y - np.dot(tx,w)
    g = -np.dot(tx.T,e)/(len(y))
    return g,e

"""returns the gradient for sgd, same as the previous function"""
def compute_stoch_gradient(y, tx, w):
    e = y - np.dot(tx,w)
    g = -np.dot(tx.T,e)/(len(y))
    return g,e

"""performs gradient descent"""
def gradient_descent(y, tx, w_init, max_iters, gamma):
    # Define parameters to store w and loss
    ws = [w_init]
    losses = []
    w = w_init
    
    for n_iter in range(max_iters):
        #compute loss and gradient
        grad,e = compute_gradient(y,tx,w)
        loss = compute_mse(e)
        
        #update the weights
        w = w - gamma*grad
        ws.append(w)
        losses.append(loss)
        print("GD iter. {bi}/{ti}: loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))  
    return losses, ws

"""the copy of the function in helpers.py, should we change this?"""
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

"""performs stochastic gradient descent"""
def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
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
    