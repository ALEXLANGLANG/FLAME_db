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
class TestFlame(unittest.TestCase):
    
    def test_large_C_repeats_F(self):
        p = 4
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
        try:
            for verbose in [0,1,2,3]:
                for matching_option in [0,1,2,3]:
                    #Test fixed weights
                    res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    C = 0,
                                    conn = conn,
                                    matching_option = 3,
                                    adaptive_weights = False,
                                    weight_array = weight_array,        
                                    verbose = verbose,
                                    k = 2
                                    )
                    print(ATE_db(res_post_new))
                    print(ATT_db(res_post_new))
        except (KeyError, ValueError):
                # We would hit this block if theres a key error, so df columns
                # are not equal or have different units, or weird entry in df, (string)
                is_corrct = 0

        self.assertEqual(1, is_corrct,
                             msg='Error when test fixed weights')
# #Generate toy dataset
# p = 4
# TE = 5
# data,weight_array = gen_data_db(n = 5000,p = p, TE = TE)
# holdout,weight_array = gen_data_db(n = 500,p = p, TE = TE)


# #Connect to the database
# select_db = "postgreSQL"  # Select the database you are using
# database_name='tmp' # database name
# host ="vcm-17819.vm.duke.edu" #"127.0.0.1"
# port = "5432"
# user="newuser"
# password= "sunxian123"
# conn = connect_db(database_name, user, password, host, port)


# #Insert the data into database
# insert_data_to_db("test_df", # The name of your table containing the dataset to be matched
#                     data, 
#                     treatment_column_name= "Treated",
#                     outcome_column_name= 'outcome123',conn = conn)


# #**********************************Test**********************************#

# #Test fixed weights
# res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
#                 holdout_data = holdout, # holdout set
#                 treatment_column_name= "Treated",
#                 outcome_column_name= 'outcome123',
#                 C = 0,
#                 conn = conn,
#                 matching_option = 3,
#                 adaptive_weights = False,
#                 weight_array = weight_array,        
#                 verbose = 2,
#                 k = 0
#                 )
# print(ATE_db(res_post_new))
# print(ATT_db(res_post_new))
# for verbose in [0,1,2,3]:
#     res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
#                     holdout_data = holdout, # holdout set
#                     treatment_column_name= "Treated",
#                     outcome_column_name= 'outcome123',
#                     C = 0,
#                     conn = conn,
#                     matching_option = 3,
#                     adaptive_weights = 'ridge',
#                     verbose = verbose,
#                     k = 0
#                     )
#     print(ATE_db(res_post_new))
#     print(ATT_db(res_post_new))
    
# for C in [0.0, 0.2,0.6]:
#     res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
#                     holdout_data = holdout, # holdout set
#                     treatment_column_name= "Treated",
#                     outcome_column_name= 'outcome123',
#                     C = C,
#                     conn = conn,
#                     matching_option = 1,
#                     adaptive_weights = 'ridge',
#                     verbose = verbose,
#                     k = 0
#                     )
#     print(ATE_db(res_post_new))
#     print(ATT_db(res_post_new))
# for matching_option in [0,1,2,3]: 
#     res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
#                     holdout_data = holdout, # holdout set
#                     treatment_column_name= "Treated",
#                     outcome_column_name= 'outcome123',
#                     C = 0.0,
#                     conn = conn,
#                     matching_option = matching_option,
#                     adaptive_weights = 'ridge',
#                     verbose = 1,
#                     k = 0
#                     )
#     print(ATE_db(res_post_new))
#     print(ATT_db(res_post_new))
# for adaptive_weights in ['ridge', 'decisiontree']:
#     res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
#                 holdout_data = holdout, # holdout set
#                 treatment_column_name= "Treated",
#                 outcome_column_name= 'outcome123',
#                 C = 0,
#                 conn = conn,
#                 matching_option = 1,
#                 adaptive_weights = adaptive_weights,
#                 verbose = verbose,
#                 k = 0
#                 )
#     print(ATE_db(res_post_new))
#     print(ATT_db(res_post_new))
# for alpha in [0.1,0.8]:
#     res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
#                 holdout_data = holdout, # holdout set
#                 treatment_column_name= "Treated",
#                 outcome_column_name= 'outcome123',
#                 C = 0,
#                 conn = conn,
#                 matching_option = 1,
#                 adaptive_weights = 'decisiontree',
#                 verbose = 1,
#                 k = 0,
#                 alpha = alpha,
#                 early_stop_iterations = 2            
#                 )
#     print(ATE_db(res_post_new))
#     print(ATT_db(res_post_new))
# for max_depth in [8,9]:
#     res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
#                 holdout_data = holdout, # holdout set
#                 treatment_column_name= "Treated",
#                 outcome_column_name= 'outcome123',
#                 C = 0,
#                 conn = conn,
#                 matching_option = 1,
#                 adaptive_weights = 'decisiontree',
#                 verbose = 1,
#                 k = 0,
#                 max_depth = max_depth,
#                 early_stop_iterations = 2            
#                 )
#     print(ATE_db(res_post_new))
#     print(ATT_db(res_post_new))
    
# for early_stop_iterations in [2,3]:
#     res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
#                 holdout_data = holdout, # holdout set
#                 treatment_column_name= "Treated",
#                 outcome_column_name= 'outcome123',
#                 C = 0,
#                 conn = conn,
#                 matching_option = 1,
#                 adaptive_weights = 'decisiontree',
#                 verbose = 1,
#                 k = 0,
#                 early_stop_iterations = early_stop_iterations            
#                 )
#     print(ATE_db(res_post_new))
#     print(ATT_db(res_post_new))
# for k in [0,2,4]:
#     res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
#                 holdout_data = holdout, # holdout set
#                 treatment_column_name= "Treated",
#                 outcome_column_name= 'outcome123',
#                 C = 0,
#                 conn = conn,
#                 matching_option = 1,
#                 adaptive_weights = 'decisiontree',
#                 verbose = 1,
#                 k = k
#                 )
#     print(ATE_db(res_post_new))
#     print(ATT_db(res_post_new))
# for ratio in [0.01,0.1]:
#     res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
#                 holdout_data = holdout, # holdout set
#                 treatment_column_name= "Treated",
#                 outcome_column_name= 'outcome123',
#                 C = 0,
#                 conn = conn,
#                 matching_option = 1,
#                 adaptive_weights = 'decisiontree',
#                 verbose = 1,
#                 ratio = ratio
#                 )
#     print(ATE_db(res_post_new))
#     print(ATT_db(res_post_new))
# for early_stop_un_c_frac in [0.2,0.5]:
#     res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
#             holdout_data = holdout, # holdout set
#             treatment_column_name= "Treated",
#             outcome_column_name= 'outcome123',
#             C = 0,
#             conn = conn,
#             matching_option = 1,
#             adaptive_weights = 'decisiontree',
#             verbose = 1,
#             early_stop_un_c_frac = early_stop_un_c_frac                
#             )
#     print(ATE_db(res_post_new))
#     print(ATT_db(res_post_new))
    
# for early_stop_un_t_frac in [0.2,0.5]:
#     res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
#             holdout_data = holdout, # holdout set
#             treatment_column_name= "Treated",
#             outcome_column_name= 'outcome123',
#             C = 0,
#             conn = conn,
#             matching_option = 1,
#             adaptive_weights = 'decisiontree',
#             verbose = 1,
#             early_stop_un_t_frac = early_stop_un_t_frac                
#             )
#     print(ATE_db(res_post_new))
#     print(ATT_db(res_post_new))
# for early_stop_pe in [3,5]:
#     res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
#             holdout_data = holdout, # holdout set
#             treatment_column_name= "Treated",
#             outcome_column_name= 'outcome123',
#             C = 0,
#             conn = conn,
#             matching_option = 1,
#             adaptive_weights = 'decisiontree',
#             verbose = 1,
#             early_stop_pe = early_stop_pe                
#             )
#     print(ATE_db(res_post_new))
#     print(ATT_db(res_post_new))
# for early_stop_pe_frac in [0.5,1]:    
#     res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
#             holdout_data = holdout, # holdout set
#             treatment_column_name= "Treated",
#             outcome_column_name= 'outcome123',
#             C = 0,
#             conn = conn,
#             matching_option = 1,
#             adaptive_weights = 'decisiontree',
#             verbose = 1,
#             early_stop_pe_frac = early_stop_pe_frac                
#             )   
#     print(ATE_db(res_post_new))
#     print(ATT_db(res_post_new))
# for missing_data_replace in [0,1,2]:
#     res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
#             holdout_data = holdout, # holdout set
#             treatment_column_name= "Treated",
#             outcome_column_name= 'outcome123',
#             C = 0,
#             conn = conn,
#             matching_option = 1,
#             adaptive_weights = 'decisiontree',
#             verbose = 1,
#             missing_data_replace = missing_data_replace                
#             ) 
#     print(ATE_db(res_post_new))
#     print(ATT_db(res_post_new))
# for missing_holdout_replace in [0,1]:    
#     res_post_new = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
#             holdout_data = holdout, # holdout set
#             treatment_column_name= "Treated",
#             outcome_column_name= 'outcome123',
#             C = 0,
#             conn = conn,
#             matching_option = 1,
#             adaptive_weights = 'decisiontree',
#             verbose = 1,
#             missing_holdout_replace = missing_holdout_replace                
#             )
#     print(ATE_db(res_post_new))
#     print(ATT_db(res_post_new))
