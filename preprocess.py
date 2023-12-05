
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import re


def read_file_header_attribute(path_to_file, index_column):
    # Read the first line of the file
    with open(path_to_file, 'r') as file:
        first_line = file.readline().strip()

    # Check if the first line contains at least one numerical value
    if any(char.isdigit() for char in first_line):
        # If yes, set header=None
        df = pd.read_csv(path_to_file, delimiter=',', header=None, index_col=index_column)
    else:
        # If no, read the file normally
        df = pd.read_csv(path_to_file, delimiter=',', index_col=index_column)
    return df


def preprocess_data2(path_to_file, index_column=None):
    dataframe = read_file_header_attribute(path_to_file, index_column)
    dataframe.replace(['?'], value='', inplace=True)

    category_columns = []
    regex_integer = re.compile(r'^(-?\d+|-?\d+\.0*|nan)$')

    for col in dataframe.columns:

        if dataframe[col].apply(lambda x: bool(regex_integer.match(str(x)))).all():
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
            dataframe[col] = dataframe[col].astype('Int64')

        if dataframe[col].nunique() < 15:
            category_columns.append(col)
            dataframe[col] = dataframe[col].astype('category')

    print(dataframe.dtypes)
    print(dataframe)


def preprocess_data():
    data_bn = pd.read_csv('data_banknote_authentication.txt', sep = ',', header = None, names=['var', 'skew', 'curt', 'entropy', 'class'])
    ds_bn = data_bn.drop(columns=['class'])
    cl_bn = data_bn['class']

    data_ckd = pd.read_csv('kidney_disease.csv')
    data_ckd.dropna(inplace = True)
    data_ckd.reset_index(inplace=True, drop=True)
    data_ckd['age'] = data_ckd['age'].astype('int64')
    data_ckd['bp'] = data_ckd['bp'].astype('int64')
    data_ckd['sg'] = data_ckd['sg'].astype("category")
    data_ckd['al'] = data_ckd['al'].astype("category")
    data_ckd['su'] = data_ckd['su'].astype("category")
    data_ckd['rbc'] = data_ckd['rbc'].astype("category")
    data_ckd['pc'] = data_ckd['pc'].astype("category")
    data_ckd['pcc'] = data_ckd['pcc'].astype("category")
    data_ckd['ba'] = data_ckd['ba'].astype("category")
    data_ckd['bgr'] = data_ckd['bgr'].astype("int64")
    data_ckd['bu'] = data_ckd['bu'].astype("int64")
    data_ckd['sod'] = data_ckd['sod'].astype("int64")
    data_ckd['pcv'] = data_ckd['pcv'].astype("int64")
    data_ckd['wc'] = data_ckd['wc'].astype("int64")
    data_ckd['rc'] = data_ckd['rc'].astype("float64")
    data_ckd['htn'] = data_ckd['htn'].astype("category")
    data_ckd['dm'] = data_ckd['dm'].astype("category")
    data_ckd['cad'] = data_ckd['cad'].astype("category")
    data_ckd['appet'] = data_ckd['appet'].astype("category")
    data_ckd['pe'] = data_ckd['pe'].astype("category")
    data_ckd['ane'] = data_ckd['ane'].astype("category")
    data_ckd['classification'] = data_ckd['classification'].astype("category")

    gt = pd.get_dummies(data_ckd['classification'])
    ckd = data_ckd.drop(columns='classification')

    for column in ckd.columns:
        if ckd[column].dtype == 'category':
            ckd=pd.get_dummies(ckd,columns=[column])
        else:
            ckd[column] = (ckd[column]-ckd[column].min())/(ckd[column].max()-ckd[column].min())
    return(ds_bn,cl_bn,ckd,gt)


# print(pd.read_csv('kidney_disease.csv', delimiter=',').dtypes)
preprocess_data2('kidney_disease.csv', index_column=0)

# # Créer un exemple de DataFrame (assurez-vous d'utiliser votre propre DataFrame)
# data = {'colonne': ['-7', '-9.0000000', '6.000', '61', '3.', 'NaN', '6.2', '-4.126']}
# df = pd.DataFrame(data)
#
# # Définir le motif regex pour les entiers valides
# pattern = re.compile(r'^(-?\d+|-?\d+\.0*|NaN)$')
#
# # Appliquer la vérification sur la colonne
# result = df['colonne'].apply(lambda x: bool(pattern.match(str(x))))
# print(result)
# # Vérifier si toutes les valeurs sont des entiers
# if result.all():
#     print("La colonne contient uniquement des entiers conformes aux critères.")
# else:
#     print("La colonne contient au moins une valeur qui ne correspond pas aux critères.")
