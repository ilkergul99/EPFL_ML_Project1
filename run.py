import numpy as np

def build_k_indices(y, k_fold, seed): # TODO: not tested these they probably not work yet, adding as a guide
    """build k indices for k-fold.
    
    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    """

    te_x, te_y = x[k_indices[k]], y[k_indices[k]]
    tr_x, tr_y = x[k_indices[(np.arange(len(k_indices))!=k).reshape(-1)]], y[k_indices[(np.arange(len(k_indices))!=k).reshape(-1)]]
    
    tr_x = build_poly(tr_x, degree)
    te_x = build_poly(te_x, degree)
    
    w = ridge_regression(tr_y.reshape(-1), tr_x, lambda_) # TODO: use cross validation for all implementations
    
    loss_tr = np.sqrt(2 * compute_mse(tr_y.reshape(-1), tr_x, w))
    loss_te = np.sqrt(2 * compute_mse(te_y.reshape(-1), te_x, w))
    return loss_tr, loss_te
    
def cross_validation_demo(degree, k_fold, lambdas):
    """cross validation over regularisation parameter lambda.
    
    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """
    
    seed = 12
    degree = degree
    k_fold = k_fold
    lambdas = lambdas
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    # INSERT YOUR CODE HERE
    # cross validation over lambdas: TODO
    # ***************************************************
    for lambda_ in lambdas:
        trl, tel = 0, 0
        for k in range(k_fold):
            trlu, telu = cross_validation(y, x, k_indices, k, lambda_, degree)
            trl += trlu
            tel += telu
        rmse_tr.append(trl/k_fold)
        rmse_te.append(tel/k_fold)

    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    best_lambda = lambdas[np.argmin(rmse_te)]
    best_rmse = rmse_te[np.argmin(rmse_te)]
    print("For polynomial expansion up to degree %.f, the choice of lambda which leads to the best test rmse is %.5f with a test rmse of %.3f" % (degree, best_lambda, best_rmse))
    
    return best_lambda, best_rmse


best_lambda, best_rmse = cross_validation_demo(7, 4, np.logspace(-4, 0, 30))