import numpy as np
import pandas as pd
from flame_db.gen_insert_data import *
from flame_db.FLAME_db_algorithm import *
from flame_db.matching_helpers import *
from flame_db.utils import *
import unittest
import pandas as pd
import os
import sys




def check_statistics(res_post_new):
    ATE_ = ATE_db(res_post_new)
    ATT_ = ATT_db(res_post_new)
    if type(ATE_) == np.nan:
        print("ATE: " + str(ATE_))
        return True
    if type(ATT_) == np.nan:
        print("ATT:" + str(ATT_))
        return True
    return False


p = 20
TE = 5
data,weight_array = gen_data_db(n = 5000,p = p, TE = TE)
holdout,weight_array = gen_data_db(n = 500,p = p, TE = TE)
#Connect to the database
select_db = "postgreSQL"  # Select the database you are using
database_name='tmp' # database name
host ='vcm-17819.vm.duke.edu' # "127.0.0.1"
port = "5432"
user="newuser"
password= "sunxian123"
conn = connect_db(database_name, user, password, host, port)

#Insert the data into database
insert_data_to_db("test_df", # The name of your table containing the dataset to be matched
                    data,
                    treatment_column_name= "Treated",
                    outcome_column_name= 'outcome123',conn = conn)

is_corrct = 1


res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
            holdout_data = holdout, # holdout set
            treatment_column_name= "Treated",
            outcome_column_name= 'outcome123',
            C = 0.1,
            conn = conn,
            matching_option = 0,
            verbose = 3,
            k = 0
            )
check_statistics(res_post_new)


res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
            holdout_data = holdout, # holdout set
            treatment_column_name= "Treated",
            outcome_column_name= 'outcome123',
            C = 0.1,
            conn = conn,
            matching_option = 2,
            adaptive_weights = False,
            weight_array = weight_array,
            verbose = 3,
            k = 0
            )
check_statistics(res_post_new)


#Insert the data into database
insert_data_to_db("test_df", # The name of your table containing the dataset to be matched
                    data,
                    treatment_column_name= "Treated",
                    outcome_column_name= 'outcome123',conn = conn,add_missing = True)
holdout_miss = holdout.copy()
m,n = holdout_miss.shape
for i in range(int(m/100)):
    for j in [0,int(n/2)]:
        holdout_miss.iloc[i,j] = np.nan
res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                        holdout_data = holdout_miss, # holdout set
                        treatment_column_name= "Treated",
                        outcome_column_name= 'outcome123',
                        C = 0,
                        conn = conn,
                        matching_option = 1,
                        adaptive_weights = 'decisiontree',
                        verbose = 2,
                        missing_data_replace = 2,
                        missing_holdout_replace = 1)
check_statistics(res_post_new)
#for verbose in [0,1,2,3]:
#    print(verbose)
#    for matching_option in [0,1,2,3]:
#        print(matching_option)
#        #Test fixed weights
#        for adaptive_weights in ['ridge', 'decisiontree',False]:
#            print(adaptive_weights)
#            res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
#                        holdout_data = holdout, # holdout set
#                        treatment_column_name= "Treated",
#                        outcome_column_name= 'outcome123',
#                        C = 0.1,
#                        conn = conn,
#                        matching_option = matching_option,
#                        adaptive_weights = adaptive_weights,
#                        weight_array = weight_array,
#                        verbose = verbose,
#                        k = 0
#                        )
#            if check_statistics(res_post_new):
#                is_corrct = 0
##





#class TestFlame_db(unittest.TestCase):
#
#    def test_weights(self):
#        #Generate toy dataset
#        p = 20
#        TE = 5
#        data,weight_array = gen_data_db(n = 5000,p = p, TE = TE)
#        holdout,weight_array = gen_data_db(n = 500,p = p, TE = TE)
#        #Connect to the database
#        select_db = "postgreSQL"  # Select the database you are using
#        database_name='tmp' # database name
#        host ='vcm-17819.vm.duke.edu' # "127.0.0.1"
#        port = "5432"
#        user="newuser"
#        password= "sunxian123"
#        conn = connect_db(database_name, user, password, host, port)
#
#        #Insert the data into database
#        insert_data_to_db("test_df", # The name of your table containing the dataset to be matched
#                            data,
#                            treatment_column_name= "Treated",
#                            outcome_column_name= 'outcome123',conn = conn)
#
#        is_corrct = 1
#        try:
#
#            for verbose in [0,1,2,3]:
#                print(verbose)
#                for matching_option in [0,1,2,3]:
#                    print(matching_option)
#                    #Test fixed weights
#                    for adaptive_weights in ['ridge', 'decisiontree',False]:
#                        print(adaptive_weights)
#                        res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
#                                    holdout_data = holdout, # holdout set
#                                    treatment_column_name= "Treated",
#                                    outcome_column_name= 'outcome123',
#                                    C = 0.1,
#                                    conn = conn,
#                                    matching_option = matching_option,
#                                    adaptive_weights = adaptive_weights,
#                                    weight_array = weight_array,
#                                    verbose = verbose,
#                                    k = 0
#                                    )
#                        if check_statistics(res_post_new):
#                            is_corrct = 0
#
#        except (KeyError, ValueError):
#                is_corrct = 0
#
#        self.assertEqual(1, is_corrct,
#                             msg='Error when test weights')
            
#    def test_stop_iterations(self):
#        is_corrct = 1
#        try:
#            for early_stop_iterations in [2,3]:
#                res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
#                            holdout_data = holdout, # holdout set
#                            treatment_column_name= "Treated",
#                            outcome_column_name= 'outcome123',
#                            C = 0.1,
#                            conn = conn,
#                            matching_option = 0,
#                            adaptive_weights = 'decisiontree',
#                            verbose = 1,
#                            k = 0,
#                            early_stop_iterations = early_stop_iterations
#                            )
#                if check_statistics(res_post_new):
#                    is_corrct = 0
#            
#        except (KeyError, ValueError):
#                is_corrct = 0
#
#        self.assertEqual(1, is_corrct,
#                             msg='Error when test stop_iterations')
#
#    def test_missing_datasets(self):
#            is_corrct = 1
#            try:
#                for missing_data_replace in [0,1,2]:
#                    for missing_holdout_replace in [0,1]:
#                        holdout_miss = holdout.copy()
#                        m,n = holdout_miss.shape
#                        for i in range(int(m/100)):
#                            for j in [0,int(n/2)]:
#                                holdout_miss.iloc[i,j] = np.nan
#                        res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
#                                                holdout_data = holdout_miss, # holdout set
#                                                treatment_column_name= "Treated",
#                                                outcome_column_name= 'outcome123',
#                                                C = 0,
#                                                conn = conn,
#                                                matching_option = 1,
#                                                adaptive_weights = 'decisiontree',
#                                                verbose = 1,
#                                                missing_data_replace = missing_data_replace,
#                                                missing_holdout_replace = missing_holdout_replace)
#                        if check_statistics(res_post_new):
#                            is_corrct = 0
#
#            except (KeyError, ValueError):
#                    is_corrct = 0
#
#            self.assertEqual(1, is_corrct,
#                                 msg='Error when test missing datasets')
#
#        #     print(ATE_db(res_post_new))
#        #     print(ATT_db(res_post_new))        
