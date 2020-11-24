from helpers import *


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




