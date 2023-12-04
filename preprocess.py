
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

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
    