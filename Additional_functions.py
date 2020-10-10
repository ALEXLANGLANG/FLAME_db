# In[1]:

import numpy as np
import pandas as pd
import pyodbc
import time
import pickle
import operator
from operator import itemgetter
from joblib import Parallel, delayed

from sklearn import linear_model

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

from sqlalchemy import create_engine

import matplotlib.pyplot as plt

import mysql.connector
import psycopg2
import sqlite3

def insert_data_to_db(table_name,data,treatment_column_name,outcome_column_name,
                      select_db,database_name,host,user,password,port = None,driver = None):
    
    table = table_name
    cur,conn = connect_db(select_db = select_db, database_name=database_name, host = host, 
                       port = port, user=user, password= password,driver = driver)


    data['matched'] = [0]*data.shape[0]
    colnames = data.columns
    cur.execute('drop table if exists {}'.format(table))


    col_setup =""
    for i in range(len(colnames)):
        v = colnames[i]
        if v == outcome_column_name:
            col_setup += v + ' float(53)'
        else :
            col_setup += v + ' int'
        if i != len(colnames) - 1:
            col_setup += ','
    cur.execute('CREATE TABLE ' + table +  '('+ col_setup+');')

    for i in range(data.shape[0]):
        col = ','.join(['{0}'.format(v) for v in colnames])
        values = ','.join(['{0}'.format(v) for v in data.iloc[i]])
        cur.execute('INSERT INTO '+  table +'('+ col +') VALUES (' + values + ')')
        
    conn.commit()
    print('Insert {} rows successfully to Database'.format(data.shape[0]))
    
    
def ATE_db(res_post):
    MG_weight = 0
    MG_weight_total = 0
    ATE_total = 0
    for i in range(len(res_post[1])):
        if type((res_post[1][i])) != type(None):  
            MGs = res_post[1][i]
            for j in range(MGs.shape[0]):
                MG = MGs.iloc[j]
                MG_weight = MG['count_c'] + MG['count_t']
                ATE_total += MG_weight* (MG['effect_t'] - MG['effect_c'])
                MG_weight_total += MG_weight

    ATE = ATE_total/MG_weight_total           
    return ATE

def gen_data_db(n = 250,p = 5, TE = 1):
    if p <= 2:
        print("p should be larger than 2")
        return None

    covs = np.random.binomial(1,0.5,size=(n,p))
    treated = np.random.binomial(1, 0.5, size = n)
    outcome = 15 * covs[:, 1] - 10 * covs[:, 2] + 5 * covs[:, 3] + TE * treated + np.random.normal(size = n)

    data = np.append(covs, treated.reshape(-1,1), axis=1)
    data = np.append(data, outcome.reshape(-1,1), axis=1)
    col_names = ['cov' + str(i+1) for i in range(p)] + ['Treated', 'outcome123']
    data = pd.DataFrame(data)
    data.columns = col_names 
    return data

def gen_data_neg(n = 250,p = 5, TE = 1,verbose = False):
    n = int(n)
    p = int(p)
    if p <= 2:
        print("p should be larger than 2")
        return None
    covs = np.random.binomial(1,0.5,size=(n,p))
    treated = np.random.binomial(1, 0.5, size = n)
    outcome = np.random.normal(size = n) + TE *treated
    for i in range(p):
        if i % 3 == 0:
            outcome += (-1)*covs[:,i]
        else:
            outcome += covs[:,i]

            
    data = np.append(covs, treated.reshape(-1,1), axis=1)
    data = np.append(data, outcome.reshape(-1,1), axis=1)
    col_names = ['cov' + str(i+1) for i in range(p)] + ['Treated', 'outcome123']
    data = pd.DataFrame(data)
    data.columns = col_names 
    return data    

#Generate dataset with exponential and Power  Decay
def gen_data_db_exp(n = 250,p = 5,TE = 1, verbose = True):
    n = int(n)
    p = int(p)
    if p <= 2:
        print("p should be larger than 2")
        return None
    covs = np.random.binomial(1,0.5,size=(n,p))
    treated = np.random.binomial(1, 0.5, size = n)
    outcome = np.random.normal(size = n) + TE *treated
    for i in range(p):
        if verbose:
            outcome += 5*(1/2)**(i+1)*covs[:,i]
        else:
            outcome += 5*(1/(i+1))*covs[:,i]
            
    data = np.append(covs, treated.reshape(-1,1), axis=1)
    data = np.append(data, outcome.reshape(-1,1), axis=1)
    col_names = ['cov' + str(i+1) for i in range(p)] + ['Treated', 'outcome123']
    data = pd.DataFrame(data)
    data.columns = col_names 
    return data


def gen_data_db_intersection(n = 250,p = 5,TE = 1, verbose = True):
    n = int(n)
    p = int(p)
    if p <= 2:
        print("p should be larger than 2")
        return None
    covs = np.random.binomial(1,0.5,size=(n,p))
    treated = np.random.binomial(1, 0.5, size = n)
    outcome = np.random.normal(size = n) + TE *treated
    for i in range(p):
        if i !=0:
            outcome += covs[:,i]*covs[:,i-1]
            
    data = np.append(covs, treated.reshape(-1,1), axis=1)
    data = np.append(data, outcome.reshape(-1,1), axis=1)
    col_names = ['cov' + str(i+1) for i in range(p)] + ['Treated', 'outcome123']
    data = pd.DataFrame(data)
    data.columns = col_names 
    return data

