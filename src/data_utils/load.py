# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 09:59:53 2016

@author: vitaliyradchenko
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np

def mape(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_data():
    train = pd.read_csv('../data/train.csv', sep=';')
    test = pd.read_csv('../data/test.csv', sep=';')
    
    #train = train.drop(['id'], axis=1)
    #test = test.drop(['id'], axis=1)
    
    train, val = train_test_split(train, test_size=0.1, random_state=42)
    
    val = val.drop(['prime_tot_ttc'], axis=1)
    test =  pd.concat([val, test], axis=0)
    
    return train, test
    
def get_score(pred,save = True,name = 'submission1'):
    train = pd.read_csv('../data/train.csv', sep=';')
    
    train, val = train_test_split(train, test_size=0.1, random_state=42)
    score = mape(val.prime_tot_ttc.values,pred[:30000]) 
    if save:
        test = pd.read_csv('../data/test.csv', sep=';')
        answers = pd.DataFrame({'COTIS': pred[30000:], 'ID': test.id.values})
        answers[['ID', 'COTIS']].to_csv('%s.csv' %(name),sep=';',index=False)
    return score
    
def beautiful_head(df, size_chunk_columns=6, n_rows=5):
    from IPython.core.display import display, HTML

    if type(size_chunk_columns) is list:
        current_column = 0
        for i in size_chunk_columns:
            display(HTML('<style> .df thead tr { background-color: #B0B0B0; } </style>' +
                         df.iloc[:,current_column:current_column+i].head(n_rows).to_html(classes='df')))
            current_column += i
    elif type(size_chunk_columns) is int:
        for i in range(0, df.shape[1], size_chunk_columns):
            display(HTML('<style> .df thead tr { background-color: #B0B0B0; } </style>' +
                         df.iloc[:,i:i+size_chunk_columns].head(n_rows).to_html(classes='df')))