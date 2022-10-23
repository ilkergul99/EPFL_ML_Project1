import numpy as np

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

"""returns the mean squared error"""
def compute_mse(e):
    sq = np.square(e)
    error = 0.5*np.mean(sq)
    return error

"""returns the gradient for sgd, same as the previous function"""
def compute_stoch_gradient(y, tx, w):
    e = y - np.dot(tx,w)
    g = -np.dot(tx.T,e)/(len(y))
    return g,e

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
    return -1*tx.T.dot(error)/error.size

def load_data(path):
    #alternative read at once and then split into x and y
    X = np.genfromtxt(
    path, delimiter=",", skip_header=1, usecols=range(2,32))
    Y = np.genfromtxt(
    path, delimiter=",", skip_header=1, converters={1: lambda x: 0 if b's' in x else 1}, usecols=[1])
    return X,Y

def negative_likelihood_grad(y, tx, w):
    """returns grad"""
    return np.dot(tx.T, sigmoid(np.dot(tx, w)) - y)

def negative_likelihood_loss(y, tx, w):
    """returns loss"""
    t = np.dot(tx, w)
    return np.sum(np.log(1 + np.exp(t)) - y * t) 

def reg_negative_likelihood_grad(y, tx, w, lambda_):
    """returns grad with regularization term for l2"""
    return np.dot(tx.T, sigmoid(np.dot(tx, w)) - y) + 2 * lambda_ * w

def sigmoid(t):
    """apply the sigmoid function on t"""
    return 1 / (1 + np.exp(-t))

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})