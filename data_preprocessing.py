import numpy as np

def column_standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    
    x = x - mean_x
    std_x = np.std(x, axis=0)
    
    x = x / std_x
    return x, mean_x, std_x

def handle_missing_and_outliers(x, outlier_coef):
    """
    Handles missing values and outliers, 
    Args:
        x: Features of the data = (N,D)
        outlier_coef: coefficient that will be used to determine upper and lower bounds
        
    Returns:
        res_x: the resulting data features
    """
    res_x = x.copy()
    mean_of_non_null = np.zeros((x.shape[1]))
    for i in range(x.shape[1]):
        mean_of_non_null[i] = np.mean(res_x[np.where(res_x[:, i] != -999), i])
        res_x[np.where(x[:, i] == -999), i] = mean_of_non_null[i]
        
    if outlier_coef != -1:
        for i in range(x.shape[1]):
            std = np.std(res_x[:, i])
            lower, upper = mean_of_non_null[i] - outlier_coef * std, mean_of_non_null[i] + outlier_coef * std
            res_x[np.where(res_x[:, i] < lower), i] = lower
            res_x[np.where(res_x[:, i] > upper), i] = upper
    
    return res_x
    
def remove_correlated_cols(tr_x, te_x, tol):
    """
    Removes one of the columns that has correlation higher than 1-tol
    Args:
        tr_x: Features of the training data = (N1,D)
        tr_x: Features of the training data = (N2,D)
        tol: tolerance that chooses correltion threshold
        
    Returns:
        res_tr_x: the resulting trainging data features
        res_te_x: the resulting testing data features
    """
    cor_matrix = np.corrcoef(tr_x, rowvar=False)
    highs = np.where(cor_matrix > 1-tol)
    remove_columns = highs[0][np.where(highs[0] > highs[1])]
    res_tr_x = tr_x.copy()
    res_tr_x = np.delete(res_tr_x, remove_columns, 1)
    if te_x is not None:
        res_te_x = te_x.copy()
        res_te_x = np.delete(res_te_x, remove_columns, 1)
    return res_tr_x, res_te_x

def build_poly(x, degree, columns=[]):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.
        
    Returns:
        poly: numpy array of shape (N,d+1)
    """
    if columns == []:
        columns = np.arange(x.shape[1])
    res = [x]
    p=np.arange(2, degree+1)
    for cind in columns:
        res.append(np.power(np.tile(x.T[cind, :].reshape((-1,1)), degree -1), p))
    return np.concatenate(res, axis=1)
    
def apply_preprocessing(tr_x, te_x, corr_tol=0.01, outlier_coef=2.5, degree=1):
    """
    applies the preproceesing functions in this file 
    Args:
        tr_x: Features of the training data = (N1,D)
        te_x: Features of the testing data = (N2,D)
        corr_tol: tolerance that chooses correltion threshold
        outlier_coef: coefficient that will be used to determine upper and lower bounds
        
    Returns:
        res_tr_x: the resulting trainging data features
        res_te_x: the resulting testing data features
    """
    unnecessary_columns = []
    for i in range(tr_x.shape[1]):
        if len(np.unique(tr_x[:,i])) == 1:
            unnecessary_columns.append(i)
    print("unnecessary columns are: ", unnecessary_columns)
    tr_x = np.delete(tr_x, unnecessary_columns, axis=1)
    te_x = np.delete(te_x, unnecessary_columns, axis=1)
    tr_x = handle_missing_and_outliers(tr_x, outlier_coef)
    te_x = handle_missing_and_outliers(te_x, outlier_coef)
    
    if corr_tol > 0:
        tr_x, te_x = remove_correlated_cols(tr_x, te_x, corr_tol)
    
    tr_x = column_standardize(tr_x)[0]
    te_x = column_standardize(te_x)[0]
    
    if degree > 1:
        tr_x = build_poly(tr_x, degree)
        te_x = build_poly(te_x, degree)
    tr_x = np.c_[np.ones(tr_x.shape[0]), tr_x]
    te_x = np.c_[np.ones(te_x.shape[0]), te_x]
    return tr_x, te_x