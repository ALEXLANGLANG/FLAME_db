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
from gen_insert_data import *

#Read in data 
def read_files(input_data, holdout_data):
    """Both options can be either df or csv files and are parsed here.
    
    Input:
        input_data: string, name of table in database
        holdout_data: The holdout data as string filename, df
        
    Return:
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


def get_CATE_db(cov_l, db_name, level,cur, conn,treatment_column_name,outcome_column_name):
    #Get the matched groups for this level
    #Get average outcome for control and treated units respectively
    #Get the number of control and treated units respectively in each matched groups
    #Combine these above as result_df
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
    result = pd.merge(pd.DataFrame(np.array(res_c), 
                                   columns=['{}'.format(i) for i in cov_l]+['avg_outcome_control', 'num_control']), 
                  pd.DataFrame(np.array(res_t), columns=['{}'.format(i) for i in cov_l]+['avg_outcome_treated', 'num_treated']), 
                  on = ['{}'.format(i) for i in cov_l], how = 'inner') 
    
    result_df = result[['{}'.format(i) for i in cov_l] + 
                       ['avg_outcome_control', 'avg_outcome_treated', 'num_control', 'num_treated']]
    return result_df


# Calculating Balancing factor
def get_BF(cov_l, c, db_name, holdout_df, thres, tradeoff,cur, conn,treatment_column_name,outcome_column_name): 
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

    time_match = time.time() - s
    
    s = time.time()
    # the number of unmatched treated units
    cur.execute('''select count(*) from {0} where matched = 0 and {1} = 0'''.format(db_name,treatment_column_name))
    num_control = cur.fetchall()
    # the number of unmatched control units
    cur.execute('''select count(*) from {0} where matched = 0 and {1} = 1'''.format(db_name,treatment_column_name))
    num_treated = cur.fetchall()
    time_BF = time.time() - s
    
    if len(res) == 0:
        return (time_match, time_BF, 0)
    else:  
        return (time_match, time_BF, tradeoff * (float(len(res[res[:,-2]==0]))/num_control[0][0] + 
                            float(len(res[res[:,-2]==1]))/num_treated[0][0]))
    
    
    
# this function takes the current covariate set, the covariate we consider dropping, name of the data table, 
# name of the holdout table, the threshold (below which we consider as no match), and balancing regularization
# as input; and outputs the matching quality
def score_tentative_drop_c(cov_l, c, db_name, holdout_df, thres, tradeoff,cur, conn,treatment_column_name,outcome_column_name):
    '''
    cov_l: current covariates to be matched
    c: the covariate tend to be dropped
    db_name: the name of input data in the database
    '''
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

    tree_c = DecisionTreeRegressor(max_depth=8, random_state=0)
    tree_t = DecisionTreeRegressor(max_depth=8, random_state=0)
    
    holdout = holdout_df.copy()
    holdout = holdout[ ["{}".format(c) for c in covs_to_match_on] + [treatment_column_name, outcome_column_name]]

   
    mse_t = np.mean(cross_val_score(tree_t, holdout[holdout[treatment_column_name] == 1].iloc[:,:-2], 
                                    holdout[holdout[treatment_column_name] == 1][outcome_column_name]  
                                     ,scoring  = 'neg_mean_squared_error') )#scoring
        
    mse_c = np.mean(cross_val_score(tree_c, holdout[holdout[treatment_column_name] == 0].iloc[:,:-2], 
                                    holdout[holdout[treatment_column_name] == 0][outcome_column_name]
                                     ,scoring = 'neg_mean_squared_error') )# 

    
    time_PE = time.time() - s
    
    if len(res) == 0:
        return (( mse_t + mse_c )/2, time_match, time_PE, time_BF)
    else:  
        return (tradeoff * (float(len(res[res[:,-2]==0]))/num_control[0][0] + 
                            float(len(res[res[:,-2]==1]))/num_treated[0][0]) + ( mse_t + mse_c )/2,
                time_match, 
                time_PE, 
                time_BF)



def run_main(db_name, holdout_df,treatment_column_name,outcome_column_name, cur, conn, tradeoff, ratio, option):
    '''
    db_name: the name of your table containing the dataset to be matched
    holdout_df: # holdout set
    tradeoff:  Tradeoff between BF and PE
    ratio: a hyperparameter to decide if we should do fast dropping without matching
    option: 'adaptive_weight' or a list of fixed weights if you want to use your own weights 
    '''
    
    cur.execute('update {0} set matched = 0'.format(db_name)) # reset the matched indicator to 0
    conn.commit()
    
    weights_option = option # contain array of weights of fixed weights
    fixed_weights = False
    covs_dropped = [] # covariate dropped
    ds = []
    level = 1

    # initialize the current covariates to be all covariates
    cols_all = holdout_df.columns.drop([treatment_column_name, outcome_column_name])
    cur_covs= cols_all
    
    
    #Check if fixed weights are input
    if type(weights_option) != str and  len(weights_option) != 0:
        fixed_weights = True



    # Do mathcing without dropping any covs at first time
    update_matched(cur_covs, db_name, level,cur, conn,treatment_column_name,outcome_column_name)
    print("Level"+ str(level) +":  Do matching without dropping any covs")
    d = get_CATE_db(cur_covs, db_name, level,cur, conn,treatment_column_name,outcome_column_name)
    if type(d) == pd.DataFrame: # ignore the None type
        ds.append(d)

        
    # do varible importance selection to get the dictionary of covs weights 
    order_cov = dict()
    for i in range(len(cur_covs)):
        c = cur_covs[i]
        score = 0
        if fixed_weights == False:
            score,time_match,time_PE,time_BF = score_tentative_drop_c(cur_covs, c, db_name, 
                                                                      holdout_df, 0, tradeoff,cur, conn,
                                                                      treatment_column_name,outcome_column_name)
        else:
            score = -weights_option[i] #convert into negative socre, we are tring to drop the biggest score
        order_cov[c] = score
        
    sorted_x = sorted(order_cov.items(), key=operator.itemgetter(1),reverse=True)  # sorted weights 
    cur_covs = dict(sorted_x) 
    
    
    
    #Get the maximum of PE when we do not drop any covariate
    max_PE = score_tentative_drop_c(cur_covs.keys(), None, db_name, holdout_df, 0,tradeoff,cur, 
                                    conn,treatment_column_name,outcome_column_name)[0]
    
    is_unimportant = True # Flag to decide if we just drop without matching
    
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
            
        best_score = -np.inf
        cov_to_drop = None
        
        # if fixed_weights do dropping without cross-validtion
        if fixed_weights: 
            if tradeoff != 0:
                for c, PE in cur_covs.items():
                    time_match,time_BF,BF = get_BF(cur_covs.keys() , c, db_name, 
                                                    holdout_df, 0, tradeoff,cur, conn,
                                                   treatment_column_name,outcome_column_name)

                    score = BF + PE
                    if score > best_score:
                        best_score = score
                        cov_to_drop = c
            else:
                cov_to_drop = sorted_x[level-2][0] 
            print("Level"+ str(level) +":  Do matching after drop " + 
                  cov_to_drop + "  with fixed negative weights: " + str(cur_covs[cov_to_drop]))
        else:

            # if adaptive weights:
            # 1. check if it is unimportant,  all unimportant covs directly based on a bound
            # 2. drop the important covs with cross-validation
            
            if is_unimportant: 
                if  max(cur_covs.values()) >= (1 + ratio)*max_PE: 
                    c = sorted_x[level-2][0]
                    cov_to_drop = c
                    score,time_match,time_PE,time_BF = score_tentative_drop_c(cur_covs, c, db_name, 
                                                                              holdout_df, 0, tradeoff,cur,
                                                                              conn,treatment_column_name,outcome_column_name)
                    if score < (1 + ratio)*max_PE:
                        is_unimportant = False
                else:
                    is_unimportant = False
                    
            if not is_unimportant:
                for c, PE in cur_covs.items():   
                    score = score_tentative_drop_c(cur_covs, c, db_name, 
                                                   holdout_df, 0, tradeoff,cur, 
                                                   conn,treatment_column_name,outcome_column_name)[0]
                    if score > best_score:
                        best_score = score
                        cov_to_drop = c
                print("Level"+ str(level) +":Do matching after slow drop " + 
                      cov_to_drop + " with negative mse: " + str(cur_covs[cov_to_drop]))

            
        # If the cov is unimpoartant, just drop it without matching
        if not fixed_weights and is_unimportant:
            print("Level"+ str(level) +": No matching after fast drop " + 
                  cov_to_drop + " with negative mse: " + str(cur_covs[cov_to_drop]))
            del cur_covs[cov_to_drop]
            continue
            
        del cur_covs[cov_to_drop]
        
        #Update the database and get CATE and matched groups
        update_matched(cur_covs.keys(), db_name, level,cur, conn,treatment_column_name,outcome_column_name)
        d = get_CATE_db(cur_covs.keys(), db_name, level,cur, conn,treatment_column_name,outcome_column_name)

        
        if type(d) == pd.DataFrame:
            ds.append(d)
        covs_dropped.append(cov_to_drop) # append the removed covariate at the end of the covariate
        
    return ds

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

def FLAME_db(input_data, holdout_data, treatment_column_name, 
                 outcome_column_name, reg_param, 
                 select_db, database_name, host, user, password,
                 port = None, driver = None,ratio = 0.01, weights_option = 'adaptive_weight'):
    '''
    input_data: the name of your table containing the dataset to be matched
    holdout_data: # holdout set
    treatment_column_name
    outcome_column_name
    reg_param:  Tradeoff between BF and PE
    select_db: the name of database you want to use: "MySQL", "postgreSQL","Microsoft SQL server"
    database_name:
    host: 
    user: 
    password:
    port = None
    driver = None
    ratio = 0.01: a hyperparameter to decide if we should do fast dropping without matching
    weights_option: 'adaptive_weight' or a list of fixed weights if you want to use your own weights 
    '''
    
    cur,conn = connect_db(select_db = select_db, database_name=database_name, host = host, 
                   port = port, user=user, password= password, driver = driver)

    holdout_data = read_files(input_data, holdout_data)
    
    res = run_main(input_data, holdout_data,treatment_column_name,outcome_column_name, 
                   cur, conn, reg_param, ratio,weights_option)
    print("Done Matching")
    return res




