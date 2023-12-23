"""
Module Name: my_module
Description: This file provides a tool to implement a machine learning workflow:
1.Import a dataset
2. Clean data, perform pre-processing
3. Split the dataset
4. Train the model (including feature selection)
5. Validate the model

Author: Jérôme Sioc'han de Kersabiec, Pierre Monot, and Romain Pépin
Contact: jerome.de-kersabiec@imt-atlantique.net ; pierre.monot@imt-atlantique.net ; romain.pepin@imt-atlantique.net
Date: 07/12/2023
"""
import numpy as np
import optuna
import pandas as pd
import re

from sklearn.model_selection import train_test_split, ShuffleSplit

from optuna.integration import OptunaSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, cross_val_predict, GridSearchCV
from typing import Tuple
from typing import Union
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report as report
from scikitplot.metrics import plot_confusion_matrix, plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Disable all warnings
warnings.filterwarnings("ignore")


class BinaryClassificationModel(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassificationModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def read_file_header_attribute(
        path_to_file: str,
        index_column: Union[int, None] = None,
        delimiter: str = ','
) -> pd.DataFrame:
    """
    Give the dataframe stored in a file.

    :param path_to_file: path of the file containing the dataframe
    :param index_column: If the file contains a column for the index, it specifies the index of that column
    :param delimiter: the delimiter used in the file
    :return: dataframe
    """
    # Read the first line of the file
    with open(path_to_file, 'r') as file:
        first_line = file.readline().strip()

    # Check if the first line contains at least one numerical value
    if any(char.isdigit() for char in first_line):
        # If yes, set header=None
        dataframe = pd.read_csv(path_to_file, delimiter=delimiter, header=None, index_col=index_column)
    else:
        # If no, read the file normally
        dataframe = pd.read_csv(path_to_file, delimiter=delimiter, index_col=index_column)
    return dataframe


def preprocess_data(
        path_to_file: str,
        index_column: Union[int, None] = None,
        categorical_max_different_values: int = 10,
        drop_first: bool = True
) -> pd.DataFrame:
    """
    Preprocess data from a given file. First, read it as a dataframe, parse columns types to their according types,
    fill missing values and do one-hot-encoding for categorical columns.

    :param path_to_file: path of the file containing the dataframe
    :param index_column: If the file contains a column for the index, it specifies the index of that column
    :param categorical_max_different_values: Maximum number of different values in a column from which it can be set as
                                             categorical
    :param drop_first: Whether during one-hot-encoding first column should be dropped
    :return: preprocess dataframe
    """
    dataframe = read_file_header_attribute(path_to_file, index_column)

    dataframe = dataframe.map(lambda x_: x_.strip() if isinstance(x_, str) else x_)
    dataframe.replace(['?'], value='', inplace=True)

    regex_integer = re.compile(r'^(-?\d+|-?\d+\.0*|nan|)$')  # Regex expression for any integer (-4;3.0000;-2.;nan...)
    regex_float = re.compile(r'^(-?\d+\.\d+|-?\d+|-?\d+\.|nan|)$')

    for col in dataframe.columns:

        if dataframe[col].nunique() <= categorical_max_different_values:
            dataframe[col] = dataframe[col].astype('category')
            # @Future to do
            # COULD BE MODIFIED TO SEE WHETHER IT CAN BE CAST TO INTEGERS/FLOATS AND THEN TAKE THE MEDIAN
            # MAY IMPROVE RESULTS OR (NOT SURE)
            # Replace empty strings and NaN values with the most frequent value
            dataframe[col] = dataframe[col].replace('', dataframe[col].mode()[0])
            dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])

        else:
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
            if dataframe[col].apply(lambda x_: bool(regex_integer.match(str(x_)))).all():
                dataframe[col] = dataframe[col].astype('Int64')
            elif dataframe[col].apply(lambda x_: bool(regex_float.match(str(x_)))).all():
                dataframe[col] = dataframe[col].astype('Float64')
            # Fill NaN values with the median
            dataframe[col] = dataframe[col].fillna(dataframe[col].median())
            # Center and normalize the data
            scaler = StandardScaler()
            scaler.fit(dataframe[col].values.reshape(-1, 1))
            dataframe[col] = scaler.transform(dataframe[col].values.reshape(-1, 1))

    dataframe = pd.get_dummies(dataframe, drop_first=drop_first)
    boolean_columns = dataframe.select_dtypes(include='bool').columns
    dataframe[boolean_columns] = dataframe[boolean_columns].astype(int)
    return dataframe


def prepare_dataset_for_training(
        dataframe: pd.DataFrame,
        target_column_name: str,
        test_size: float = 0.3,
        random_state: int = 42,
        n_splits: int = 5
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    ShuffleSplit
]:
    """
    Prepare a dataset for training. Split the dataset into random train and test subsets with a cross validation object.

    :param dataframe: data
    :param target_column_name: Name of the target variable
    :param test_size: Size of the test dataset (should be between 0.0 and 1.0)
    :param random_state: Controls the shuffling applied to the data before applying the split. Pass an int for
                         reproducible output across multiple function calls.
    :param n_splits: Number of splits for cross validation if it is used
    :return: x and y data for training and testing and a cross validation object
    """

    x = dataframe.drop(target_column_name, axis=1)
    y = dataframe[target_column_name]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    if n_splits is None:
        n_splits = int(len(dataframe) / 100)
    cvp = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    return x_train, y_train, x_test, y_test, cvp


def training(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        cvp: ShuffleSplit,
        random_state: int = 42,
        verbose: int = 0,
        test_size: float = 0.3,
        n_trials: int = 20
) -> list[
    Union[
        BinaryClassificationModel,
        SVC,
        LogisticRegression,
        DecisionTreeClassifier,
        RandomForestClassifier,
        AdaBoostClassifier,
        KNeighborsClassifier
    ]
]:
    """
    Train different models : SVC, logistic regression, decision tree, random forest, AdaBoost, K nearest neighbors and
    deep learning.

    :param x_train: Features data to train the model on (without target variable)
    :param y_train: Target variable data to train the model on
    :param cvp: Cross validation object to use
    :param random_state: A random seed for result reproduction purpose
    :param verbose: Controls the verbosity: the higher, the more messages.
                    >1 : the computation time for each fold and parameter candidate is displayed;
                    >2 : the score is also displayed;
                    >3 : the fold and candidate parameter indexes are also displayed together with the starting time of
                    the computation.
    :param test_size: Size of the test dataset (should be between 0.0 and 1.0)
    :param x_test: Feature data to test the model  (without target variable)
    :param y_test: Target variable data to test the model
    :param n_trials: Number of trials to test different parameters on ML algorithms
    :return: A list of all trained models
    """
    output_models = []

    x_train_array = x_train.values
    y_train_array = y_train.values.reshape(-1, 1)
    x_test_array = x_test.values
    y_test_array = y_test.values.reshape(-1, 1)
    # Convert to PyTorch tensors
    x_train_tensor = torch.from_numpy(x_train_array).float()
    y_train_tensor = torch.from_numpy(y_train_array).float()
    x_test_tensor = torch.from_numpy(x_test_array).float()
    y_test_tensor = torch.from_numpy(y_test_array).float()

    input_size = x_train.shape[1]  # Number of features
    # Create a study object and optimize hyperparameters
    study = optuna.create_study(direction='maximize')

    study.optimize(
        lambda trial: objective(trial, cvp, input_size, x_train_tensor, y_train_tensor),
        n_trials=n_trials
    )

    # Print the best hyperparameters
    best_params = study.best_params
    best_lr = best_params['lr']
    best_num_epochs = best_params['num_epochs']
    print(f"Best Learning Rate: {best_lr}, Best Number of Epochs: {best_num_epochs}")

    # Use the best hyperparameters to train the final model
    final_model = BinaryClassificationModel(input_size)
    criterion = nn.BCELoss()
    final_optimizer = optim.Adam(final_model.parameters(), lr=best_lr)

    for epoch in range(best_num_epochs):
        # Forward pass
        y_pred = final_model(x_train_tensor)

        # Compute the loss
        loss = criterion(y_pred, y_train_tensor)

        # Backward pass and optimization
        final_optimizer.zero_grad()
        loss.backward()
        final_optimizer.step()

    # Evaluate the final model on the test set (unseen data for the model)
    with torch.no_grad():
        final_model.eval()
        y_pred_test = final_model(x_test_tensor)
    y_pred_test = (y_pred_test > 0.5).float()
    conf_matrix = confusion_matrix(y_test_tensor.detach().numpy(), y_pred_test.detach().numpy())

    output_models.append(final_model)

    # All the models that will be applied
    models = [
        SVC(random_state=random_state),
        LogisticRegression(random_state=random_state),
        DecisionTreeClassifier(random_state=random_state),
        RandomForestClassifier(random_state=random_state),
        AdaBoostClassifier(random_state=random_state),
        KNeighborsClassifier()
    ]

    # List of parameters to test for all models
    param_grids = [
        {
            'C': [0.1, 1, 10, 100],
            'kernel': ['sigmoid', 'rbf']
        },  # SVM
        {
            'penalty': ['l2', None],
            'C': [0.1, 1, 10, 100],
            'solver': ['lbfgs', 'newton-cholesky', 'newton-cg', 'sag', 'saga']
        },  # Logistic Regression
        {
            'max_depth': [1, 2, 3, 4, 6, 8, 10, 15, 20, None]
        },  # Decision Tree Classifier
        {
            'n_estimators': [100, 150, 200],
            'max_depth': [1, 2, 3, 4, 6, 8, 10, 15, 20, None]
        },  # Random Forest Classifier
        {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.1, 0.5, 1, 2]
        },  # AdaBoost Classifier
        {
            'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10],
            'weights': ['uniform', 'distance']
        }  # KNN
    ]

    # Train all models
    for model, param_grid in zip(models, param_grids):
        # Find the best parameters with cross-validation and train the model with those best parameters
        grid_search = GridSearchCV(model, param_grid, cv=cvp, verbose=verbose)
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(x_test)
        conf_matrix = confusion_matrix(y_test, y_pred)

        output_models.append(best_model)

    return output_models


# The following function is not tested yet
def objective(trial, cvp, input_size, x_train_tensor, y_train_tensor):
    # Sample hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    num_epochs = trial.suggest_int('num_epochs', 100, 5000, step=1000)

    # Training loop
    f1_scores = []

    for train_index, test_index in cvp.split(x_train_tensor):
        # Define the model
        model = BinaryClassificationModel(input_size)

        # Define the loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            # Forward pass
            y_pred = model(x_train_tensor[train_index])

            # Compute the loss
            loss = criterion(y_pred, y_train_tensor[train_index])

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate the model on the test set
        with torch.no_grad():
            model.eval()
            y_pred_test = model(x_train_tensor[test_index])
        y_pred_test = (y_pred_test > 0.5).float()
        # Calculate accuracy on the test set
        f1_scores.append(f1_score(y_train_tensor[test_index].detach().numpy(), y_pred_test.detach().numpy()))
    f1 = sum(f1_scores)/len(f1_scores)

    return f1

def visualize_results(
    ground_truth: pd.Series,
    result: pd.Series
):
    """
    Prints different metrics and plots to visualize the classification results

    :param ground_truth: pandas.Series of the true Labels
    :param result: pandas.Series of the predicted Labels
    """

    #First prints the predictions scores
    print(report(ground_truth,result))

    #Plots the confusion matrix

    plot_confusion_matrix(ground_truth,result,normalize = True)
    plt.show()





df = preprocess_data('kidney_disease.csv', index_column=0)

xtrain, ytrain, xtest, ytest, cv = prepare_dataset_for_training(
    df,
    'classification_notckd',
    n_splits=8,
)

print(training(x_train=xtrain, y_train=ytrain, x_test=xtest, y_test=ytest, cvp=cv, n_trials=10))

visualize_results(y_train,y_train)