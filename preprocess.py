import pandas as pd
import numpy as np
import re


def read_file_header_attribute(path_to_file, index_column):
    # Read the first line of the file
    with open(path_to_file, 'r') as file:
        first_line = file.readline().strip()

    # Check if the first line contains at least one numerical value
    if any(char.isdigit() for char in first_line):
        # If yes, set header=None
        dataframe = pd.read_csv(path_to_file, delimiter=',', header=None, index_col=index_column)
    else:
        # If no, read the file normally
        dataframe = pd.read_csv(path_to_file, delimiter=',', index_col=index_column)
    return dataframe


def replace_with_binary(dataframe):
    for column in dataframe.columns:
        unique_values = dataframe[column].replace('', np.nan).dropna().unique()

        # Check if there are only two unique non-null values in the column
        if len(unique_values) == 2:
            dataframe[column] = \
                (dataframe[column].apply(
                    lambda x: 1 if pd.notna(x) and x == unique_values[0] else 0 if pd.notna(x) else x))

    return dataframe


def preprocess_data(path_to_file, index_column=None, categorical_max_different_values=10, drop_first=True):
    dataframe = read_file_header_attribute(path_to_file, index_column)

    dataframe = dataframe.map(lambda x: x.strip() if isinstance(x, str) else x)
    dataframe.replace(['?'], value='', inplace=True)

    regex_integer = re.compile(r'^(-?\d+|-?\d+\.0*|nan|)$')  # Regex expression for any integer (-4;3.0000;-2.;nan...)
    regex_float = re.compile(r'^(-?\d+\.\d+|-?\d+|-?\d+\.|nan|)$')

    for col in dataframe.columns:

        if dataframe[col].nunique() <= categorical_max_different_values:
            dataframe[col] = dataframe[col].astype('category')
            # @TODO
            # COULD BE MODIFIED TO SEE WHETHER OR NOT IT CAN BE CAST TO INTEGERS/FLOATS AND THEN TAKE THE MEDIAN
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
