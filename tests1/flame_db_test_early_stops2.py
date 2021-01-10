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


TE = 5

p=10
data,weight_array = gen_data_db(n = 100,p = p, TE = TE)
holdout,weight_array = gen_data_db(n = 50,p = p, TE = TE)
insert_data_to_db("test_df4", # The name of your table containing the dataset to be matched
                    data,
                    treatment_column_name= "treated",
                    outcome_column_name= 'outcome',conn = conn)
res_post_new1 = FLAME_db(input_data = "test_df4", # The name of your table containing the dataset to be matched
                        holdout_data = holdout, # holdout set
                        conn = conn,
                        matching_option = 3,
                        verbose = 3,
                        k = 0,
                        early_stop_iterations = 1
                        )

res_post_new1 = FLAME_db(input_data = "test_df4", # The name of your table containing the dataset to be matched
                        holdout_data = holdout, # holdout set
                        conn = conn,
                        matching_option = 3,
                        verbose = 3,
                        k = 0,
                        early_stop_un_c_frac = 1
                        )
res_post_new1 = FLAME_db(input_data = "test_df4", # The name of your table containing the dataset to be matched
                        holdout_data = holdout, # holdout set
                        conn = conn,
                        matching_option = 3,
                        verbose = 3,
                        k = 0,
                        early_stop_un_t_frac = 1
                        )
