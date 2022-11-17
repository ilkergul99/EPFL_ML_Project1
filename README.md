# EPFL Machine Learning Higgs Boson Challange Project


In this project, we implemented 6 machine learning methods to be able to make predictions for the https://www.aicrowd.com/challenges/epfl-machine-learning-higgs

## Running the model

In order to get the results that we submitted here are the steps:

- Create a folder called `ml_project_dataset` under this directory
- Place the `test.csv` and `train.csv` files under the created `ml_project_dataset` folder in this directory
- Run
```
python run.py
```
After finishing the `run.py` the predictions for the test dataset will be created as `submission.csv` file in this directory

## To run cross validation on jet categories

Our cross validation code can be imported in any other `.py` and `.ipynb` file. To do so, here are the steps:
- Import the cross validation code using
```
from run import cross_validation_on_jets
```
- Then run
```
cross_validation_on_jets(<data_path>, <degrees>, <k_fold>, <lambdas>, <seed = 1>, <verbose=False>)
```
data_path should point to the path of the data, degrees should be a list containing the polynomial basis expansion degree options, k_fold should be an integer, lambdas should be a list containing the regularazation parameter options, seed and verbose is optional. When set to true, verbose increases the output volume

## To do data pre-processing
In order to replicate our data pre-processing steps in any other `.py` and `.ipynb` file, follow these steps:
- Import the apply_preprocessing code using
```
from data_preprocessing import apply_preprocessing
```
- Then run
```
res_tr_x, res_te_x = apply_preprocessing(tr_x, te_x, corr_tol=0.01, outlier_coef=2.0, degree=1, log_cols=[])
```
tr_x should be the features of the training set, te_x should be the features of the test set, it returns resulting training features and resulting test features.

## To read the report regarding the project
In order to read the explanation regarding the implementation for this project you can check `EPFL_Machine_Learning_Higgs_Boson_Challange_Project.pdf` in this directory







