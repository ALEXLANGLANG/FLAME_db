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




def construct_sec_order(arr):
    # data generation function helper.
    second_order_feature = []
    num_cov_sec = len(arr[0])
    for a in arr:
        tmp = []
        for i in range(num_cov_sec):
            for j in range(i+1, num_cov_sec):
                tmp.append( a[i] * a[j] )
        second_order_feature.append(tmp)
        
    return np.array(second_order_feature)

def data_generation_dense_2(num_control, num_treated, num_cov_dense, num_covs_unimportant, 
                            control_m = 0.1, treated_m = 0.9):
    # the data generation function that I'll use.
    xc = np.random.binomial(1, 0.5, size=(num_control, num_cov_dense))   # data for conum_treatedrol group
    xt = np.random.binomial(1, 0.5, size=(num_treated, num_cov_dense))   # data for treatmenum_treated group
        
    errors1 = np.random.normal(0, 0.05, size=num_control)    # some noise
    errors2 = np.random.normal(0, 0.05, size=num_treated)    # some noise
    
    dense_bs_sign = np.random.choice([-1,1], num_cov_dense)
    dense_bs = [ np.random.normal(s * 10, 1) for s in dense_bs_sign ]

    yc = np.dot(xc, np.array(dense_bs)) + errors1     # y for conum_treatedrol group 
    
    treatment_eff_coef = np.random.normal( 1.5, 0.15, size=num_cov_dense)
    treatment_effect = np.dot(xt, treatment_eff_coef) 
    
    second = construct_sec_order(xt[:,:5])
    treatment_eff_sec = np.sum(second, axis=1)
    
    yt = np.dot(xt, np.array(dense_bs)) + treatment_effect + treatment_eff_sec + errors2    # y for treated group 

    xc2 = np.random.binomial(1, control_m, size=(num_control, num_covs_unimportant))   #
    xt2 = np.random.binomial(1, treated_m, size=(num_treated, num_covs_unimportant))   #
        
    df1 = pd.DataFrame(np.hstack([xc, xc2]), 
                       columns=['{0}'.format(i) for i in range(num_cov_dense + num_covs_unimportant)])
    df1['outcome'] = yc
    df1['treated'] = 0

    df2 = pd.DataFrame(np.hstack([xt, xt2]), 
                       columns=['{0}'.format(i) for i in range(num_cov_dense + num_covs_unimportant )] ) 
    df2['outcome'] = yt
    df2['treated'] = 1

    df = pd.concat([df1,df2])
    df['matched'] = 0
  
    return df, dense_bs, treatment_eff_coef


# In[4]:

# this function takes the current covariate list, the covariate we consider dropping, name of the data table, 
# name of the holdout table, the threshold (below which we consider as no match), and balancing regularization
# as input; and outputs the matching quality
def score_tentative_drop_c(cov_l, c, db_name, holdout_df, thres, tradeoff,cur, conn,treatment_column_name,outcome_column_name):
    
    covs_to_match_on = set(cov_l) - {c} # the covariates to match on
    
    # the flowing query fetches the matched results (the variates, the outcome, the treatment indicator)
    s = time.time()

    cur.execute('''with temp AS 
        (SELECT 
        {0}
        FROM {3}
        where matched=0
        group by {0}
        Having sum({4})> 0 and sum({4})<count(*) 
        )
        (SELECT {1}, {4}, {5}
        FROM {3}
        WHERE matched=0 AND EXISTS 
        (SELECT 1
        FROM temp 
        WHERE {2}
        )
        )
        '''.format(','.join(['{0}'.format(v) for v in covs_to_match_on ]),
                   ','.join(['{1}.{0}'.format(v, db_name) for v in covs_to_match_on ]),
                   ' AND '.join([ '{1}.{0}=temp.{0}'.format(v, db_name) for v in covs_to_match_on ]),
                   db_name,treatment_column_name,outcome_column_name
                  ) )
    res = np.array(cur.fetchall())
#     print(len(res))
    time_match = time.time() - s
    
    s = time.time()
    # the number of unmatched treated units
    cur.execute('''select count(*) from {0} where matched = 0 and {1} = 0'''.format(db_name,treatment_column_name))
    num_control = cur.fetchall()
    # the number of unmatched control units
    cur.execute('''select count(*) from {0} where matched = 0 and {1} = 1'''.format(db_name,treatment_column_name))
    num_treated = cur.fetchall()
    time_BF = time.time() - s
    
    
    
    s = time.time() # the time for fetching data into memory is not counted if use this
    
    # below is the regression part for PE
    tree_c = DecisionTreeRegressor(max_depth=8, random_state=0)
    tree_t = DecisionTreeRegressor(max_depth=8, random_state=0)
    
    holdout = holdout_df.copy()
    holdout = holdout[ ["{}".format(c) for c in covs_to_match_on] + [treatment_column_name, outcome_column_name]]
    
    mse_t = np.mean(cross_val_score(tree_t, holdout[holdout[treatment_column_name] == 1].iloc[:,:-2], 
                                holdout[holdout[treatment_column_name] == 1][outcome_column_name] , scoring = 'neg_mean_squared_error' ) )
        
    mse_c = np.mean(cross_val_score(tree_c, holdout[holdout[treatment_column_name] == 0].iloc[:,:-2], 
                                holdout[holdout[treatment_column_name] == 0][outcome_column_name], scoring = 'neg_mean_squared_error' ) )
#     print(mse_t+mse_c)
#     print(tradeoff * (float(len(res[res[:,-2]==0]))/num_control[0][0] + float(len(res[res[:,-2]==1]))/num_treated[0][0]) + ( mse_t + mse_c ))
    # above is the regression part for BF
    
    time_PE = time.time() - s
    
    if len(res) == 0:
        return (( mse_t + mse_c ), time_match, time_PE, time_BF)
    else:  
#         print(tradeoff * (float(len(res[res[:,-2]==0]))/num_control[0][0]))

        return (tradeoff * (float(len(res[res[:,-2]==0]))/num_control[0][0] + 
                            float(len(res[res[:,-2]==1]))/num_treated[0][0]) + ( mse_t + mse_c ),
                time_match, 
                time_PE, 
                time_BF)


# In[5]:

# update matched units
# this function takes the currcent set of covariates and the name of the database; and update the "matched"
# column of the newly mathced units to be "1"

def update_matched(covs_matched_on, db_name, level,cur, conn,treatment_column_name,outcome_column_name): 
   
  
    cur.execute('''with temp AS 
    (SELECT 
    {0}
    FROM {3}
    where matched=0
    group by {0}
    Having sum({5})>0 and sum({5})<count(*) 
    )
    update {3} set matched={4}
    WHERE EXISTS
    (SELECT {0}
    FROM temp
    WHERE {2} and {3}.matched = 0
    )
    '''.format(','.join(['{0}'.format(v) for v in covs_matched_on]),
               ','.join(['{1}.{0}'.format(v, db_name) for v in covs_matched_on]),
               ' AND '.join([ '{1}.{0}=temp.{0}'.format(v, db_name) for v in covs_matched_on ]),
               db_name,
               level,
               treatment_column_name,
               outcome_column_name
              ) )
    conn.commit()
    
    return


# In[6]:

# get CATEs 
# this function takes a list of covariates and the name of the data table as input and outputs a dataframe 
# containing the combination of covariate values and the corresponding CATE 
# and the corresponding effect (and the count and variance) as values

def get_CATE_db(cov_l, db_name, level,cur, conn,treatment_column_name,outcome_column_name):

    cur.execute(''' select {0}, avg({4} * 1.0), count(*)
                    from {1}
                    where matched = {2} and {3} = 0
                    group by {0}
                    '''.format(','.join(['{0}'.format(v) for v in cov_l]), 
                              db_name, level, treatment_column_name,outcome_column_name) )
    res_c = cur.fetchall()
    
    cur.execute(''' select {0}, avg({4} * 1.0), count(*)
                    from {1}
                    where matched = {2} and {3} = 1
                    group by {0}
                    '''.format(','.join(['{0}'.format(v) for v in cov_l]), 
                              db_name, level, treatment_column_name,outcome_column_name) )
    res_t = cur.fetchall()
            
    if (len(res_c) == 0) | (len(res_t) == 0):
        return None
    
    cov_l = list(cov_l)
    
    result = pd.merge(pd.DataFrame(np.array(res_c), columns=['{}'.format(i) for i in cov_l]+['effect_c', 'count_c']), 
                  pd.DataFrame(np.array(res_t), columns=['{}'.format(i) for i in cov_l]+['effect_t', 'count_t']), 
                  on = ['{}'.format(i) for i in cov_l], how = 'inner') 
    
    result_df = result[['{}'.format(i) for i in cov_l] + ['effect_c', 'effect_t', 'count_c', 'count_t']]
        
    # -- the following section are moved to after getting the result
    # -- the above section are moved to after getting the result
    
    return result_df


# In[7]:

def run_db(db_name, holdout_df, num_covs,treatment_column_name,outcome_column_name, cur, conn, reg_param = 0.1):

    cur.execute('update {0} set matched = 0'.format(db_name)) # reset the matched indicator to 0
    conn.commit()

    covs_dropped = [] # covariate dropped
    list_drop=[]
    ds = []
    
    level = 1

    timings = [0]*5 # first entry - match (groupby and join), 
                    # second entry - regression (compute PE), 
                    # third entry - compute BF, 
                    # fourth entry - keep track of CATE, 
                    # fifth entry - update database table (mark matched units). 
    
    # initialize the current covariates to be all covariates
    cols_all = holdout_df.columns.drop([treatment_column_name, outcome_column_name])
    if 'matched' in cols_all: 
        cols_all = cols_all.drop('matched')
    cur_covs= cols_all
    
    # make predictions and save to disk
    s = time.time()
    update_matched(cur_covs, db_name, level,cur, conn,treatment_column_name,outcome_column_name) # match without dropping anything
    timings[4] = timings[4] + time.time() - s
        
    s = time.time()
    d = get_CATE_db(cur_covs, db_name, level,cur, conn,treatment_column_name,outcome_column_name) # get CATE without dropping anything
    timings[3] = timings[3] + time.time() - s
    
    ds.append(d)

    
    
    while len(cur_covs)>1:        
        level += 1

        # the early stopping conditions
        cur.execute('''select count(*) from {0} where matched=0 and {1}=0'''.format(db_name,treatment_column_name))
        if cur.fetchall()[0][0] == 0:
            print(" Early stopping: All control units matched")
            break
        cur.execute('''select count(*) from {0} where matched=0 and {1}=1'''.format(db_name,treatment_column_name))
        if cur.fetchall()[0][0] == 0:
            print("Early stopping: All treated units matched")
            break
        
        # drop the the most unimportant cov
        best_score = -np.inf
        cov_to_drop = None
        cur_covs = list(cur_covs)
        for c in cur_covs: 
            score,time_match,time_PE,time_BF = score_tentative_drop_c(cur_covs, c, db_name, 
                                                                      holdout_df, 0, reg_param,cur, conn,
                                                                      treatment_column_name,outcome_column_name)
            timings[0] = timings[0] + time_match
            timings[1] = timings[1] + time_PE
            timings[2] = timings[2] + time_BF
            if score > best_score:
                best_score = score
                cov_to_drop = c
        list_drop.append(cov_to_drop)
        cur_covs = set(cur_covs) - {cov_to_drop} # remove the dropped covariate from the current covariate set
#         print(" Matched Group with Level " + str(level) + " covs_matched_on: ")
#         print(list(cur_covs))
        
        s = time.time()
        update_matched(cur_covs, db_name, level,cur, conn,treatment_column_name,outcome_column_name)
        timings[4] = timings[4] + time.time() - s
        
        s = time.time()
        d = get_CATE_db(cur_covs, db_name, level,cur, conn,treatment_column_name,outcome_column_name)
        timings[3] = timings[3] + time.time() - s
        
        ds.append(d)
        
        covs_dropped.append(cov_to_drop) # append the removed covariate at the end of the covariate
        
    return timings, ds, list_drop


def read_files(input_data, holdout_data):
    """Both options can be either df or csv files and are parsed here.
    
    Input:
        input_data: The matching data as string filename or df
        holdout_data: The holdout data as string filename, df, fraction, bool
        
    Return:
        input_data: dataframe
        holdout_data: dataframe
    """
        
    # Check the type of the input data
    if type(input_data) != str:
        raise Exception("Need to specify the name of the table that contains the dataset in your database "\
                        "frame in parameter 'input_data'")

            
    # Now read the holdout data
    if (type(holdout_data) == pd.core.frame.DataFrame):
        df_holdout = holdout_data
    elif (type(holdout_data) == str()):
        df_holdout = pd.read_csv(holdout_data)
        
    else:
        print("Please input your holdout_data")
    
    df_holdout.columns = map(str, df_holdout.columns)
    
    return df_holdout

def connect_db(select_db, database_name, host, port, user, password, driver):
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
        raise Exception("please select the database you are using "\
                        "frame in parameter 'input_data'")

    cur = conn.cursor()
    return cur,conn

# def FLAME_db(input_data, holdout_data, treatment_column_name, outcome_column_name, reg_param,
#              select_db, database_name, host, port, user, password):
    
#     cur,conn = connect_db(select_db = select_db, database_name=database_name, host = host, 
#                    port = port, user=user, password= password)
    
#     holdout_data = read_files(input_data, holdout_data)
    
#     res = run_db(input_data, holdout_data, len(holdout_data.columns)-3,
#                  treatment_column_name,outcome_column_name, cur, conn,reg_param)
#     print("Done")
#     return res

def FLAME_db(input_data, holdout_data, treatment_column_name, outcome_column_name, reg_param,
             select_db, database_name, host, user, password, port = None, driver = None):
    
    cur,conn = connect_db(select_db = select_db, database_name=database_name, host = host, 
                   port = port, user=user, password= password, driver = driver)
    
    holdout_data = read_files(input_data, holdout_data)
    
    res = run_db(input_data, holdout_data,
                 treatment_column_name,outcome_column_name, cur, conn,reg_param = reg_param)
    print("Done Matching")
    return res