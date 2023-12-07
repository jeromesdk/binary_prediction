import pandas as pd
import numpy as np
import re


def read_file_header_attribute(path_to_file, index_column=None, delimiter=','):
    """
    Read the dataframe stored in a file

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


def preprocess_data(path_to_file, index_column=None, categorical_max_different_values=10, drop_first=True):
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

        elif dataframe[col].apply(lambda x: bool(regex_integer.match(str(x)))).all():
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
            dataframe[col] = dataframe[col].astype('Int64')
            # Fill NaN values with the median
            dataframe[col] = dataframe[col].fillna(dataframe[col].median())

        elif dataframe[col].apply(lambda x: bool(regex_float.match(str(x)))).all():
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
            dataframe[col] = dataframe[col].astype('Float64')
            # Fill NaN values with the median
            dataframe[col] = dataframe[col].fillna(dataframe[col].median())

    dataframe = pd.get_dummies(dataframe, drop_first=drop_first)

    return dataframe


df = preprocess_data('kidney_disease.csv', index_column=0)
print(df)
print(df.info())
