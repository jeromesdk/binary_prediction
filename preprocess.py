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

import pandas as pd
import re
from sklearn.model_selection import train_test_split, ShuffleSplit
from typing import Tuple
from typing import Union
from sklearn.preprocessing import StandardScaler


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

    dataframe = dataframe.map(lambda x: x.strip() if isinstance(x, str) else x)
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
    return dataframe


def prepare_dataset_for_training(
        dataframe: pd.DataFrame,
        target_column_name: str,
        test_size: float = 0.3,
        random_state: int = 42,
        cross_validation: bool = True,
        n_splits: Union[int, None] = None
) -> Union[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series, ShuffleSplit]]:
    """
    Prepare a dataset for training. Split the dataset into random train and test subsets.

    :param dataframe: Controls the shuffling applied to the data before applying the split. Pass an int for reproducible
                      output across multiple function calls
    :param target_column_name: Name of the target variable
    :param test_size: Precise the size of the test dataset (should be between 0.0 and 1.0)
    :param random_state: Controls the shuffling applied to the data before applying the split. Pass an int for
                         reproducible output across multiple function calls.
    :param cross_validation: Choose true to use cross validation
    :param n_splits: Number of splits for cross validation if it is used
    :return: List containing train-test split of inputs.
    """
    if not cross_validation:
        x_ = dataframe.drop(target_column_name, axis=1)
        y_ = dataframe[target_column_name]
        x__train, x__test, y__train, y__test = train_test_split(x_, y_, test_size=test_size, random_state=random_state)
        return x__train, y__train, x__test, y__test
    else:
        if n_splits is None:
            n_splits = int(len(dataframe)/100)
        x_ = dataframe.drop(target_column_name, axis=1)
        y_ = dataframe[target_column_name]
        cvp_ = ShuffleSplit(n_splits=n_splits, test_size=test_size)
        return x_, y_, cvp_


df = preprocess_data('kidney_disease.csv', index_column=0)
print(df)
print(df.info())

x_train, y_train, x_test, y_test = prepare_dataset_for_training(
    df,
    'classification_notckd',
    cross_validation=False
)

x, y, cvp = prepare_dataset_for_training(
    df,
    'classification_notckd',
    n_splits=8
)
