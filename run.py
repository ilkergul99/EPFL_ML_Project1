import numpy as np
import time
from data_preprocessing import *
from helper_functions_project1 import *
from implementations import *

def build_k_indices(y, k_fold, seed): 
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
        train and test mean accuracies(not categorical overall)

    """

    te_x, te_y = x[k_indices[k]], y[k_indices[k]]
    tr_x, tr_y = x[k_indices[(np.arange(len(k_indices))!=k)].reshape(-1)], y[k_indices[(np.arange(len(k_indices))!=k)].reshape(-1)]
    tr_x, te_x = apply_preprocessing(tr_x, te_x, degree=degree)
    
    w, _ = ridge_regression(tr_y.reshape(-1), tr_x, lambda_)
    #w, _ = logistic_regression(tr_y.reshape(-1), tr_x, np.zeros([tr_x.shape[1],1]), 10000, 0.001) 
    #w, _ = least_squares_SGD(tr_y.reshape(-1), tr_x, np.zeros([tr_x.shape[1],1]), 1, 10, 0.0001)
    #w, _ = least_squares(tr_y.reshape(-1), tr_x)
    preds_tr = tr_x.dot(w)
    preds_tr[preds_tr < 0] = -1
    preds_tr[preds_tr >= 0] = 1
    loss_tr = np.sqrt(2 * compute_mse(tr_y.reshape(-1) - preds_tr))
    acc_tr = np.sum(preds_tr == tr_y) / len(tr_y)
    
    preds_te = te_x.dot(w)
    preds_te[preds_te < 0] = -1
    preds_te[preds_te >= 0] = 1
    loss_te = np.sqrt(2 * compute_mse(te_y.reshape(-1) - preds_te))
    acc_te = np.sum(preds_te == te_y) / len(te_y)
    return loss_tr, loss_te, acc_tr, acc_te
    
def best_degree_selection(train_x, train_y, degrees, k_fold, lambdas, seed = 1):
    """cross validation over regularisation parameter lambda and degree.
    
    Args:
        degrees: shape = (d,), where d is the number of degrees to test 
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_params : parameters that result in the best rmse
        best_rmse : value of the rmse for the couple (best_degree, best_lambda)
        
    >>> best_degree_selection(np.arange(2,11), 4, np.logspace(-4, 0, 30))
    (7, 0.004520353656360241, 0.28957280566456634)
    """
    
    # split data in k fold
    k_indices = build_k_indices(train_y, k_fold, seed)
    starting_time = time.time()
    rmse_tr = []
    rmse_te = []
    acc_tr = []
    acc_te = []
    pairs = []
    for degree in degrees:
        for lambda_ in lambdas:
            trl, tel, atr, ate = 0, 0, 0, 0
            for k in range(k_fold):
                trlu, telu, atru, ateu = cross_validation(train_y, train_x, k_indices, k, lambda_, degree)
                trl += trlu
                tel += telu
                atr += atru
                ate += ateu
            rmse_tr.append(trl/k_fold)
            rmse_te.append(tel/k_fold)
            acc_tr.append(atr/k_fold)
            acc_te.append(ate/k_fold)
            pairs.append((lambda_, degree))
            print("For lambda: {} and degree:{}, rmse_tr:{}, rmse_te:{}, acc_tr:{}, acc_te:{}".format(lambda_, degree, rmse_tr[-1], rmse_te[-1], acc_tr[-1], acc_te[-1]))

    best_params = pairs[np.argmin(rmse_te)]
    best_rmse = rmse_te[np.argmin(rmse_te)]
    acc_for_best_rmse = acc_te[np.argmin(rmse_te)]
    print("overall time passed: {}".format(time.time()-starting_time))
    return best_params, best_rmse, acc_for_best_rmse


