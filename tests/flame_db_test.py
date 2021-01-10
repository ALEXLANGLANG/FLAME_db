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
data,weight_array = gen_data_db(n = 1000,p = p, TE = TE)
holdout,weight_array = gen_data_db(n = 500,p = p, TE = TE)
#Connect to the database
select_db = "postgreSQL"  # Select the database you are using
database_name='tmp' # database name
host ='vcm-17819.vm.duke.edu' # "127.0.0.1"
port = "5432"
user="newuser"
password= "sunxian123"
conn = connect_db(database_name, user, password, host, port)

class TestFlame_db(unittest.TestCase):
              
    def test_weights(self):
        is_corrct = 1
        try:
            #Insert the data into database
            insert_data_to_db("test_df", # The name of your table containing the dataset to be matched
                                data,
                                treatment_column_name= "Treated",
                                outcome_column_name= 'outcome123',conn = conn)
            res_post_new1 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    C = 0.1,
                                    conn = conn,
                                    matching_option = 0,
                                    verbose = 3,
                                    k = 0
                                    )
            res_post_new2 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
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
            if check_statistics(res_post_new1) or check_statistics(res_post_new2):
                is_corrct = 0
            
        except (KeyError, ValueError):
                is_corrct = 0

        self.assertEqual(1, is_corrct,
                             msg='Error when test weights')


    def test_missing_datasets(self):
        is_corrct = 1
        try:
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
                                    matching_option = 2,
                                    adaptive_weights = 'decisiontree',
                                    verbose = 1,
                                    missing_data_replace = 0,
                                    missing_holdout_replace = 0)
            if check_statistics(res_post_new):
                is_corrct = 0

        except (KeyError, ValueError):
                is_corrct = 0

        self.assertEqual(1, is_corrct,
                             msg='Error when test missing datasets')  








class Test_exceptions(unittest.TestCase):
    
    def test_false_dataset(self):
        def broken_false_dataset():
            res_post_new1 = FLAME_db(input_data = data, # The name of your table containing the dataset to be matched
                                                holdout_data = holdout, # holdout set
                                                treatment_column_name= "Treated",
                                                outcome_column_name= 'outcome123',
                                                C = 0.1,
                                                conn = conn,
                                                matching_option = 0,
                                                verbose = 3,
                                                k = 0
                                                )
        with self.assertRaises(Exception) as false_dataset:
            broken_false_dataset()
            
        self.assertTrue("Need to specify the name of the table that contains the dataset in your database "\
                        "frame in parameter 'input_data'" in str(false_dataset.exception))
        
    def test_false_holdout(self):
        def broken_false_holdout():
            res_post_new1 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                                holdout_data = 0, # holdout set
                                                treatment_column_name= "Treated",
                                                outcome_column_name= 'outcome123',
                                                C = 0.1,
                                                conn = conn,
                                                matching_option = 0,
                                                verbose = 3,
                                                k = 0
                                                )
        with self.assertRaises(Exception) as holdout:
            broken_false_holdout
            
        self.assertTrue("Holdout_data shoule be a dataframe or a directory" in str(holdout.exception))

        
    def test_false_treatment_column_name(self):
        def broken_treatment_column_name():
            res_post_new1 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                                holdout_data = holdout, # holdout set
                                                treatment_column_name= "sadfdag",
                                                outcome_column_name= 'outcome123',
                                                C = 0.1,
                                                conn = conn,
                                                matching_option = 0,
                                                verbose = 3,
                                                k = 0
                                                )
        with self.assertRaises(Exception) as treatment_column_name:
            broken_treatment_column_name()
            
        self.assertTrue('Invalid input error. Treatment column name does not'\
                        ' exist' in str(treatment_column_name.exception))

    def test_false_outcome_column_name(self):
        def broken_outcome_column_name():
            res_post_new1 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= '1232114',
                                    C = 0.1,
                                    conn = conn,
                                    matching_option = 0,
                                    verbose = 3,
                                    k = 0
                                    )

        with self.assertRaises(Exception) as outcome_column_name:
            broken_outcome_column_name()
            
        self.assertTrue('Invalid input error. Outcome column name does not'\
                        ' exist' in str(outcome_column_name.exception))
        
    def test_false_treatment_column_name_value(self):
        def broken_treatment_column_name_value():
            df = holdout.copy()
            df.loc[0,'Treated'] = 4
            res_post_new1 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = df, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    C = 0.1,
                                    conn = conn,
                                    matching_option = 0,
                                    verbose = 3,
                                    k = 0
                                    )

        with self.assertRaises(Exception) as treatment_column_name_value:
            broken_treatment_column_name_value()
        self.assertTrue('Invalid input error. All rows in the treatment '\
                        'column must have either a 0 or a 1 value.' in str(treatment_column_name_value.exception))
        
   
        
    def test_false_early_stop_un_t_frac(self):
        def broken_early_stop_un_t_frac():
            res_post_new1 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    C = 0.1,
                                    conn = conn,
                                    matching_option = 0,
                                    verbose = 3,
                                    early_stop_un_t_frac = -1,
                                    k = 0
                                    )

        with self.assertRaises(Exception) as early_stop_un_t_frac:
            broken_early_stop_un_t_frac()
            
        self.assertTrue('The value provided for the early stopping critera '\
                        'of proportion of unmatched treatment units needs to '\
                        'be between 0.0 and 1.0' in str(early_stop_un_t_frac.exception))
    
    def test_false_early_stop_un_c_frac(self):
        def broken_early_stop_un_c_frac():
            res_post_new1 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    C = 0.1,
                                    conn = conn,
                                    matching_option = 0,
                                    verbose = 3,
                                    early_stop_un_c_frac=-1,
                                    k = 0
                                    )

        with self.assertRaises(Exception) as early_stop_un_c_frac:
            broken_early_stop_un_c_frac()
            
        self.assertTrue('The value provided for the early stopping critera '\
                        'of proportion of unmatched control units needs to '\
                        'be between 0.0 and 1.0' in str(early_stop_un_c_frac.exception))
    

    def test_false_early_stop_pe(self):
        def broken_early_stop_pe():
            res_post_new1 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    C = 0.1,
                                    conn = conn,
                                    matching_option = 0,
                                    verbose = 3,
                                    early_stop_pe = -10,
                                    k = 0
                                    )

        with self.assertRaises(Exception) as early_stop_pe:
            broken_early_stop_pe()
            
        self.assertTrue('The value provided for the early stopping critera '\
                        'of PE needs to be non-negative ' in str(early_stop_pe.exception))

    def test_false_early_stop_pe_frac(self):
        def broken_early_stop_pe_frac():
            res_post_new1 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    C = 0.1,
                                    conn = conn,
                                    matching_option = 0,
                                    verbose = 3,
                                    early_stop_pe_frac=-1,
                                    k = 0
                                    )

        with self.assertRaises(Exception) as early_stop_pe_frac:
            broken_early_stop_pe_frac()
            
        self.assertTrue('The value provided for the early stopping critera of'\
                        ' PE needs to be between 0.0 and 1.0' in str(early_stop_pe_frac.exception))
        
        
    def test_false_early_stop_iterations(self):
        def broken_early_stop_iterations():
            res_post_new1 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                        holdout_data = holdout, # holdout set
                        treatment_column_name= "Treated",
                        outcome_column_name= 'outcome123',
                        C = 0.1,
                        conn = conn,
                        matching_option = 0,
                        verbose = 3,
                        early_stop_iterations = True,
                        k = 0)
                        

        with self.assertRaises(Exception) as early_stop_iterations:
            broken_early_stop_iterations()
            
        self.assertTrue('The value provided for early_stop_iteration needs '\
                        'to be an integer number of iterations' in str(early_stop_iterations.exception))
        
        

        
    def test_false_weights_type(self):
        def broken_weights_type():
            res_post_new2 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    C = 0.1,
                                    conn = conn,
                                    matching_option = 2,
                                    adaptive_weights = 'safdsaf',
                                    verbose = 3,
                                    k = 0
                                    )
        with self.assertRaises(Exception) as _weights_type:
            broken_weights_type()
            
        self.assertTrue("Invalid input error. The acceptable values for "\
                            "the adaptive_weights parameter are 'ridge', "\
                            "'decisiontree'. Additionally, "\
                            "adaptive-weights may be 'False' along "\
                            "with a weight array" in str(_weights_type.exception))

    def test_false_weight_array_len(self):
        def broken_weight_array_len():
            res_post_new2 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    C = 0.1,
                                    conn = conn,
                                    adaptive_weights = False,
                                    matching_option = 2,
                                    weight_array = [1],
                                    verbose = 3,
                                    k = 0
                                    )
        with self.assertRaises(Exception) as weight_array_len:
            broken_weight_array_len()
            
        self.assertTrue('Invalid input error. Weight array size not equal'\
                            ' to number of columns in dataframe' in str(weight_array_len.exception))
        
        
    def test_false_weight_array_sum(self):
        def broken_weight_array_sum():
            df, true_TE = generate_uniform_given_importance(num_control=100, num_treated=100)
            model = matching.FLAME(adaptive_weights = False)
            model.fit(holdout_data=df, weight_array = )
            output = model.predict(df)
            res_post_new2 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    C = 0.1,
                                    conn = conn,
                                    adaptive_weights = False,
                                    matching_option = 2,
                                    weight_array = [1,1,1,1],
                                    verbose = 3,
                                    k = 0
                                    )
        with self.assertRaises(Exception) as weight_array_sum:
            broken_weight_array_sum()
            
        self.assertTrue('Invalid input error. Weight array values must '\
                            'sum to 1.0' in str(weight_array_sum.exception))
        
        
    def test_false_alpha(self):
        def broken_alpha():
            res_post_new2 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    C = 0.1,
                                    conn = conn,
                                    alpha = -10,
                                    verbose = 3,
                                    k = 0
                                    )
        with self.assertRaises(Exception) as alpha:
            broken_alpha()
            
        self.asertTrue('Invalid input error. The alpha needs to be '\
                            'positive for ridge regressions.' in str(alpha.exception))
        
        
    def test_false_C(self):
        def broken_C():
            res_post_new2 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    C = -10,
                                    conn = conn,
                                    verbose = 3,
                                    k = 0
                                    )
        with self.assertRaises(Exception) as C:
            broken_C()
            
        self.assertTrue('The C, or the hyperparameter to trade-off between'\
                           ' balancing factor and predictive error must be '\
                           ' nonnegative. 'in str(C.exception))

    def test_false_k(self):
        def broken_k():
            res_post_new2 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    conn = conn,
                                    verbose = 3,
                                    k = -10
                                    )
        with self.assertRaises(Exception) as k:
            broken_k()
            
        self.assertTrue('Invalid input error. The k must be'\
            'a postive integer.'in str(k.exception))

        
    def test_false_ratio(self):
        def broken_ratio():
            res_post_new2 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    conn = conn,
                                    verbose = 3,
                                    ratio = -10
                                    )
        with self.assertRaises(Exception) as ratio:
            broken_ratio()
            
        self.assertTrue('Invalid input error. ratio value must '\
                            'be positive and smaller than 1.0 \n'\
                        'Recommended 0.01 and please do not adjust it unless necessary 'in str(ratio.exception))
        
        
    def test_false_matching_option(self):
        def broken_matching_option():
            res_post_new2 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    conn = conn,
                                    verbose = 3,
                                    matching_option = -10
                                    )
        with self.assertRaises(Exception) as matching_option:
            broken_matching_option()
            
        self.assertTrue('Invalid input error. matching_option value must '\
            'be 0, 1, 2 or 3'in str(matching_option.exception))
        
    def test_false_verbose(self):
        def broken_verbose():
            res_post_new2 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    conn = conn,
                                    verbose = 10
                                    )
        with self.assertRaises(Exception) as verbose:
            broken_verbose()
            
        self.assertTrue('Invalid input error. The verbose option must be'\
                        'the integer 0,1,2 or 3.'in str(verbose.exception))
                
    def test_false_max_depth(self):
        def broken_max_depth():
            res_post_new2 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    conn = conn,
                                    max_depth = -10
                                    )
        with self.assertRaises(Exception) as max_depth:
            broken_max_depth()
            
        self.assertTrue('Invalid input error. The max_depth must be'\
                'a postive integer.'in str(max_depth.exception))
        
    def test_false_random_state(self):
        def broken_random_state():
            res_post_new2 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    conn = conn,
                                    random_state =None
                                    )
        with self.assertRaises(Exception) as random_state:
            broken_random_state()
            
        self.assertTrue('Invalid input error. The random_state  must be'\
                'a postive integer or None.'in str(random_state.exception))

        
        
    def test_false_missing_data_replace(self):
        def broken_missing_data_replace():
            res_post_new2 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    conn = conn,
                                    missing_data_replace =4
                                    )
        with self.assertRaises(Exception) as missing_data_replace:
            broken_missing_data_replace()
            
        self.assertTrue('Invalid input error. missing_data_replace value must '\
            'be 0, 1 or 2'in str(missing_data_replace.exception))
        
    def test_false_missing_holdout_replace(self):
        def broken_missing_holdout_replace():
            res_post_new2 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    conn = conn,
                                    missing_holdout_replace =4
                                    )
        with self.assertRaises(Exception) as missing_holdout_replace:
            broken_missing_holdout_replace()
            
        self.assertTrue('Invalid input error. missing_holdout_replace value must '\
            'be 0, or 1'in str(missing_holdout_replace.exception))
        
        
    def test_false_input_treatment_value(self):
        def broken_input_treatment_value():
            df = data.copy()
            df.loc[0,'Treated'] = 4
            insert_data_to_db("test_df", # The name of your table containing the dataset to be matched
                    df,
                    treatment_column_name= "Treated",
                    outcome_column_name= 'outcome123',conn = conn)
            
            res_post_new2 = FLAME_db(input_data = "test_df", # The name of your table containing the dataset to be matched
                                    holdout_data = holdout, # holdout set
                                    treatment_column_name= "Treated",
                                    outcome_column_name= 'outcome123',
                                    conn = conn
                                    )
        with self.assertRaises(Exception) as input_treatment_value:
            broken_input_treatment_value()
            
        self.assertTrue('Invalid input error. All rows in the treatment '\
                        'column must have either a 0 or a 1 value.'in str(input_treatment_value.exception))
        
        
        
        
        
        
        
        
        
#     def test_false_data_len(self):
#         def broken_data_len():
#             df, true_TE = generate_uniform_given_importance(num_control=1000, num_treated=1000,
#                                                   num_cov=7, min_val=0,
#                                                   max_val=3, covar_importance=[4,3,2,1,0,0,0])
#             holdout, true_TE = generate_uniform_given_importance()
#             model = matching.FLAME()
#             model.fit(holdout_data=holdout)
#             output = model.predict(df)

#         with self.assertRaises(Exception) as data_len:
#             broken_data_len()
            
#         self.assertTrue('Invalid input error. The holdout and main '\
#                             'dataset must have the same number of columns' in str(data_len.exception))
    
#     def test_false_column_match(self):
#         def broken_column_match():
#             df, true_TE = generate_uniform_given_importance()
#             holdout, true_TE = generate_uniform_given_importance()
#             set_ = holdout.columns
#             set_ = list(set_)
#             set_[0] = 'dasfadf'
#             holdout.columns  = set_
#             model = matching.FLAME()
#             model.fit(holdout_data=holdout)
#             output = model.predict(df)

#         with self.assertRaises(Exception) as column_match:
#             broken_column_match()
            
#         self.assertTrue('Invalid input error. The holdout and main '\
#                             'dataset must have the same columns' in str(column_match.exception))

#     def test_false_missing_data_replace(self):
#         def broken_missing_data_replace():
#                 df, true_TE = generate_uniform_given_importance(num_control=100, num_treated=100,
#                                                               num_cov=7, min_val=0,
#                                                               max_val=3, covar_importance=[4,3,2,1,0,0,0])
#                 holdout, true_TE = generate_uniform_given_importance(num_control=100, num_treated=100,
#                                                       num_cov=7, min_val=0,
#                                                           max_val=3, covar_importance=[4,3,2,1,0,0,0])
#                 covar_importance = np.array([4,3,2,1,0,0,0])
#                 weight_array = covar_importance/covar_importance.sum()
#                 model = matching.FLAME(missing_data_replace = 2, adaptive_weights =False)
#                 model.fit(holdout_data=holdout,weight_array = list(weight_array))
#                 output = model.predict(df)

#         with self.assertRaises(Exception) as missing_data_replace:
#             broken_missing_data_replace()
            
#         self.assertTrue('Invalid input error. We do not support missing data '\
#                         'handing in the fixed weights version of algorithms'in str(missing_data_replace.exception))
        

        
#     def test_false_data_type(self):
#         def broken_data_type():
#             df, true_TE = generate_uniform_given_importance(num_control=100, num_treated=100)
#             holdout = df.copy()
#             df.iloc[0,0] = 's'
#             model = matching.FLAME()
#             model.fit(holdout_data=holdout)
#             output = model.predict(df)

#         with self.assertRaises(Exception) as _data_type:
#             broken_data_type()

#         self.assertTrue('Invalid input error on matching dataset. Ensure all inputs asides from '\
#                         'the outcome column are integers, and if missing' \
#                         ' values exist, ensure they are handled.' in str(_data_type.exception))
#     def test_false_holdout_type(self):
#         def broken_holdout_type():
#             df, true_TE = generate_uniform_given_importance(num_control=100, num_treated=100)
#             holdout = df.copy()
#             holdout.iloc[0,0] = 's'
#             model = matching.FLAME()
#             model.fit(holdout_data=holdout)
#             output = model.predict(df)

#         with self.assertRaises(Exception) as holdout_type:
#             broken_holdout_type()

#         self.assertTrue('Invalid input error on holdout dataset. Ensure all inputs asides from '\
#                                 'the outcome column are integers, and if missing' \
#                                 ' values exist, ensure they are handled.' in str(holdout_type.exception))

#     def test_false_ATE_input(self):
#         def broken_ATE_input():
#             ATE(1)

#         with self.assertRaises(Exception) as ATE_input:
#             broken_ATE_input()
#         self.assertTrue("The matching_object input parameter needs to be "\
#                             "of type DAME or FLAME" in str(ATE_input.exception))

#     def test_false_ATE_input_model(self):
#         def broken_ATE_input_model():
#             model = matching.FLAME()
#             ATE(model)
#         with self.assertRaises(Exception) as ATE_input_model:
#             broken_ATE_input_model()
#         self.assertTrue("This function can be only called after a match has "\
#                            "been formed using the .fit() and .predict() functions" in str(ATE_input_model.exception))
