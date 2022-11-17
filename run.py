import numpy as np
from data_preprocessing import *
from helper_functions_project1 import *
from implementations import *

JETS_LOG_INDICES = [
    [0, 2, 3, 8, 9, 10, 11, 13, 16, 19, 21],
    [0, 1, 2, 3, 8, 9, 10, 13, 16, 19, 21, 23, 29],
    [0, 1, 2, 3, 5, 8, 9, 10, 13, 16, 19, 21, 23, 26, 29],
    [0, 1, 2, 3, 5, 8, 9, 10, 13, 16, 19, 21, 23, 26, 29],
]


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
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree, index):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()
        index:      index of the jet (jet number)

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)
        train and test mean accuracies(not categorical overall)

    """

    te_x, te_y = x[k_indices[k]], y[k_indices[k]]
    tr_x, tr_y = (
        x[k_indices[(np.arange(len(k_indices)) != k)].reshape(-1)],
        y[k_indices[(np.arange(len(k_indices)) != k)].reshape(-1)],
    )
    tr_x, te_x = apply_preprocessing(
        tr_x.copy(), te_x.copy(), degree=degree, log_cols=JETS_LOG_INDICES[index]
    )

    w, _ = ridge_regression(tr_y.reshape(-1), tr_x, lambda_)
    # w, _ = logistic_regression(tr_y.reshape(-1), tr_x, np.zeros([tr_x.shape[1],1]), 10000, 0.001)
    # w, _ = least_squares_SGD(tr_y.reshape(-1), tr_x, np.zeros([tr_x.shape[1],1]), 1, 10, 0.0001)
    # w, _ = least_squares(tr_y.reshape(-1), tr_x)
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


def best_degree_selection(
    train_x, train_y, degrees, k_fold, lambdas, index, seed=1, verbose=False
):
    """cross validation over regularisation parameter lambda and degree.

    Args:
        train_x: shape = (N, d), train features, N is the number of samples, d is the dimension
        train_y: shape = (N,), train ground truth, N is the number of samples
        degrees: shape = (d,), where d is the number of degrees to test
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
        index: integer, index of the jet (jet number)
        seed: the random seed
        verbose: increases the printed info volume

    Returns:
        best_params : parameters that result in the best rmse
        best_rmse : value of the rmse for the couple (best_degree, best_lambda)
        acc_for_best_rmse : accuracy for the degrees which results in the lowest loss
    """

    # split data in k fold
    k_indices = build_k_indices(train_y, k_fold, seed)
    rmse_tr = np.zeros((len(degrees) * len(lambdas), 2))
    rmse_te = np.zeros((len(degrees) * len(lambdas), 2))
    acc_tr = np.zeros((len(degrees) * len(lambdas), 2))
    acc_te = np.zeros((len(degrees) * len(lambdas), 2))
    pairs = []
    for i, degree in enumerate(degrees):
        for j, lambda_ in enumerate(lambdas):
            trl, tel, atr, ate = (
                np.zeros(k_fold),
                np.zeros(k_fold),
                np.zeros(k_fold),
                np.zeros(k_fold),
            )
            for k in range(k_fold):
                trlu, telu, atru, ateu = cross_validation(
                    train_y, train_x, k_indices, k, lambda_, degree, index
                )
                trl[k] = trlu
                tel[k] = telu
                atr[k] = atru
                ate[k] = ateu
            rmse_tr[i * len(lambdas) + j] = np.array([np.mean(trl), np.std(trl)])
            rmse_te[i * len(lambdas) + j] = np.array([np.mean(tel), np.std(tel)])
            acc_tr[i * len(lambdas) + j] = np.array([np.mean(atr), np.std(atr)])
            acc_te[i * len(lambdas) + j] = np.array([np.mean(ate), np.std(ate)])
            pairs.append((lambda_, degree))
            if verbose:
                print(
                    "For lambda: {} and degree:{}, rmse_tr:{}, rmse_te:{}, acc_tr:{}, acc_te:{}".format(
                        lambda_,
                        degree,
                        rmse_tr[i * len(lambdas) + j],
                        rmse_te[i * len(lambdas) + j],
                        acc_tr[i * len(lambdas) + j],
                        acc_te[i * len(lambdas) + j],
                    )
                )

    best_params = pairs[np.argmin(rmse_te[:, 0])]
    best_rmse = rmse_te[np.argmin(rmse_te[:, 0])]
    acc_for_best_rmse = acc_te[np.argmin(rmse_te[:, 0])]
    return best_params, best_rmse, acc_for_best_rmse


def cross_validation_on_jets(
    data_path, degrees, k_fold, lambdas, seed=1, verbose=False
):
    """cross validation over regularisation parameter lambda and degree.

    Args:
        data_path: path to the folder containing the data to be used for cross validation
        degrees: shape = (d,), where d is the number of degrees to test
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
        seed: the random seed
        verbose: increases the printed info volume
    Returns:
        results : a list of length 4 containing for each jet (best_params, best_rmse, acc_for_best_rmse)
    """
    train_y, train_x, train_ids = load_csv_data(data_path + "/train.csv")
    results = []
    cumulative_acc = 0
    cumulative_acc_std = 0
    for i in range(4):
        # compute jet data
        tr_x_jet = train_x[np.where(train_x[:, 22] == i)]
        tr_y_jet = train_y[np.where(train_x[:, 22] == i)]

        res = best_degree_selection(
            tr_x_jet, tr_y_jet, degrees, k_fold, lambdas, i, seed, verbose
        )
        results.append(res)
        if verbose:
            print(res)
        cumulative_acc += tr_y_jet.shape[0] * res[2][0]
        cumulative_acc_std += res[2][1]
    print("Average accuracy: ", cumulative_acc / train_y.shape)
    print("Average accuracy std: ", cumulative_acc_std / 4)
    print("Average loss: ", sum([res[1] for res in results]) / 4)
    for res in results:
        print(res[0])
    return results


def main(data_path, submission_name):
    """
    Runs the training and saves the file for our best results.
    Args:
        data_path: string, folder that contains the train and test data
        submission_name: string, name of the string to be used for submission
    """
    # read the datasets
    np.random.seed(353)
    train_y, train_x, train_ids = load_csv_data(data_path + "/train.csv")
    test_y, test_x, test_ids = load_csv_data(data_path + "/test.csv")
    pred_final = np.zeros(test_y.shape)

    degrees = [6, 8, 8, 7]
    lambdas = [1e-3, 1e-3, 1e-7, 1e-7]

    for i in range(4):
        # compute jet data
        tr_x_jet = train_x[np.where(train_x[:, 22] == i)]
        tr_y_jet = train_y[np.where(train_x[:, 22] == i)]
        te_x_jet = test_x[np.where(test_x[:, 22] == i)]
        # preprocessing
        tr_x_jet, te_x_jet = apply_preprocessing(
            tr_x_jet, te_x_jet, degree=degrees[i], log_cols=JETS_LOG_INDICES[i]
        )
        # training
        w, _ = ridge_regression(tr_y_jet.reshape(-1), tr_x_jet, lambdas[i])
        # prediction
        preds_jet = te_x_jet.dot(w)
        preds_jet[preds_jet < 0] = -1
        preds_jet[preds_jet >= 0] = 1
        pred_final[np.where(test_x[:, 22] == i)] = preds_jet

    create_csv_submission(test_ids, pred_final, submission_name)


if __name__ == "__main__":
    main("ml_project_dataset", "submission.csv")
