import numpy as np
from helper_functions_project1 import *

"""performs gradient descent"""


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
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
    w = initial_w
    losses = [compute_loss_mse(y, tx, w)]

    for n_iter in range(max_iters):
        # compute loss and gradient
        grad = compute_gradient(y, tx, w)
        # update the weights
        w = w - gamma * grad
        loss = compute_loss_mse(y, tx, w)
        ws.append(w)
        losses.append(loss)
        # print("GD iter. {bi}/{ti}: loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
    return ws[-1], losses[-1]


"""performs stochastic gradient descent"""


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
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
    w = initial_w
    losses = [compute_loss_mse(y, tx, w)]
    for n_iter in range(max_iters):
        for yn, txn in batch_iter(y, tx, 1, 1):
            grad, e = compute_stoch_gradient(yn, txn, w)
            w = w - gamma * grad
            loss = compute_loss_mse(yn, txn, w)
            ws.append(w)
            losses.append(loss)

        # print("SGD iter. {bi}/{ti}: loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
    return ws[-1], losses[-1]


"""performs least squares using normal equations"""


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
    Gram_matrix = tx.T.dot(tx)  # shape = (D,D)
    second_matrix = tx.T.dot(y)  # shape = (D,1)
    w = np.linalg.solve(Gram_matrix, second_matrix)  # shape = (D,1)
    loss = compute_loss_mse(y, tx, w)
    return w, loss


"""performs ridge regression using normal equations"""


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
    ridge_matrix = tx.T.dot(tx) + 2 * y.shape[0] * lambda_ * np.identity(
        tx.shape[1]
    )  # shape = (D,D)
    second_matrix = tx.T.dot(y)  # shape = (D,1)
    w = np.linalg.solve(ridge_matrix, second_matrix)  # shape = (D,1)
    loss = compute_loss_mse(y, tx, w)
    return w, loss


"""performs Logistic Regression using Gradient Descent"""


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

    for i in range(max_iters):
        # compute the gradient
        grad = negative_likelihood_grad(y, tx, w)
        # update weights
        w = w - gamma * grad

    return w, negative_likelihood_loss(y, tx, w)


"""performs Regularized Logistic Regression using Gradient Descent"""


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
