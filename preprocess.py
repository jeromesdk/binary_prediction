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


def preprocess_data(path_to_file, index_column=None):
    dataframe = read_file_header_attribute(path_to_file, index_column)
    dataframe = dataframe.map(lambda x: x.strip() if isinstance(x, str) else x)

    dataframe.replace(['?'], value='', inplace=True)

    regex_integer = re.compile(r'^(-?\d+|-?\d+\.0*|nan|)$')  # Regex expression for any integer (-4;3.0000;-2.;nan...)
    regex_float = re.compile(r'^(-?\d+\.\d+|-?\d+|-?\d+\.|nan|)$')

    for col in dataframe.columns:

        if dataframe[col].nunique() < 15:
            dataframe[col] = dataframe[col].astype('category')

        elif dataframe[col].apply(lambda x: bool(regex_integer.match(str(x)))).all():
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
            dataframe[col] = dataframe[col].astype('Int64')

        elif dataframe[col].apply(lambda x: bool(regex_float.match(str(x)))).all():
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
            dataframe[col] = dataframe[col].astype('Float64')

    replace_with_binary(dataframe)
    numeric_columns = dataframe.select_dtypes(include=['int', 'float']).columns
    # Calculate the median for each numeric column
    medians = dataframe[numeric_columns].median()
    # Fill NaN values in each column with its respective median
    dataframe[numeric_columns] = dataframe[numeric_columns].fillna(medians)

    for col in dataframe.columns:
        # If the column is categorical
        if dataframe[col].dtype == 'category':
            # Check if the column has exactly 2 unique non-NaN values
            if dataframe[col].nunique(dropna=True) == 2:
                # Find the most frequent value
                most_frequent_value = dataframe[col].mode().iloc[0]

                # Replace NaN values with the most frequent value
                dataframe[col].fillna(most_frequent_value, inplace=True)
            else:
                # Create a temporary variable to store numeric values
                temp_col = pd.to_numeric(dataframe[col].cat.codes, errors='coerce')
                # Set NaN values to the median of the numeric values
                median_value = temp_col.median()
                temp_col.fillna(median_value, inplace=True)
                # Replace the original categorical column with the temporary numeric values
                dataframe[col] = temp_col.astype('category')

    return dataframe


df = preprocess_data('kidney_disease.csv', index_column=0)
