import numpy as np
import pandas as pd
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
from tqdm import tqdm
import matplotlib.pyplot as plt
import mysql.connector
import psycopg2
import sqlite3
def connect_db(select_db, database_name, host, port, user, password = None, driver= None):
    conn = None;
    if select_db == "MySQL":
        conn = mysql.connector.connect(host=host,
                                        port=port,
                                        user=user,
                                        password=password,
                                        database=database_name)
    elif select_db == "postgreSQL":
        conn = psycopg2.connect(host=host,
                                port=port,
                                user=user,
                                password=password,
                                database=database_name)
        
    elif select_db == "Microsoft SQL server":
        conn = pyodbc.connect('DRIVER='+driver+
                              '; SERVER='+host+
                              ';DATABASE='+database_name+
                              ';UID='+user+
                              ';PWD='+ password)
    else:
        raise Exception("please select the database you are using ")

    cur = conn.cursor()
    return cur,conn

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




def gen_data_db(n = 250,p = 5, TE = 1):
    if p <= 2:
        print("p should be larger than 2")
        return None
    weights = [0]*p
    covs = np.random.binomial(1,0.5,size=(n,p))
        
    treated = np.random.binomial(1, 0.5, size = n)
    outcome =  TE * treated + np.random.normal(size = n)
    
    for i in range(1,p):
        coeff = 2/i
        outcome  = outcome  +  coeff*covs[:,i]
        
        weights[i-1] = coeff

    weights = np.array(weights)
    weights = weights/(sum(weights))


    data = np.append(covs, treated.reshape(-1,1), axis=1)
    data = np.append(data, outcome.reshape(-1,1), axis=1)
    
    col_names =['cov' + str(i+1) for i in range(p)] + ['Treated', 'outcome123']
    data = pd.DataFrame(data)
    data.columns = col_names 
    return data,weights



