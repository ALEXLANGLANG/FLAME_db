[![Build Status](https://travis-ci.org/almost-matching-exactly/DAME-FLAME-Python-Package.svg?branch=master)](https://travis-ci.org/almost-matching-exactly/DAME-FLAME-Python-Package)
[![Coverage Status](https://coveralls.io/repos/github/almost-matching-exactly/DAME-FLAME-Python-Package/badge.svg)](https://coveralls.io/github/almost-matching-exactly/DAME-FLAME-Python-Package)

# FLAME_db
FLAME database verion will work well and fast on large scale dataset
--------------------------------------------------
```
import pandas as pd
import dame_flame
train_df = pd.DataFrame([[0,1,1,1,0,5], [0,1,1,0,0,6], [1,0,1,1,1,7], [1,1,1,1,1,7]], 
                  columns=["x1", "x2", "x3", "x4", "treated", "outcome"])
test_df = pd.DataFrame([[0,1,1,1,0,5], [0,1,1,0,0,6], [1,0,1,1,1,7], [1,1,1,1,1,7]], 
                  columns=["x1", "x2", "x3", "x4", "treated", "outcome"])                 
```

#### Connect to the database
```
select_db = "postgreSQL"  # Select the database you are using
database_name='tmp' # database name you use 
host = 'localhost' 
port = "5432"
user="postgres"
password= ""

conn = connect_db(database_name, user, password, host, port)
```



#### Insert the data to be matched into database

If you already have the dataset in the database, please ignore this step. Insert the test_df (data to be matched) into the database you are using.
```
insert_data_to_db("datasetToBeMatched", # The name of your table containing the dataset to be matched
                  test_df,
                  treatment_column_name= "treated",
                  outcome_column_name= 'outcome',conn = conn)
```
#### Run FLAME_db

```
res = FLAME_db(input_data = "datasetToBeMatched", # The name of your table containing the dataset to be matched
              holdout_data = train_df, # holdout set. We will use holdout set to train our model
              conn = conn # connector object that connects to your database. This is the output from function connect_db.
              )
```

#### Analysis results
```
res[0]:
            df of units with the column values of their main matched
            group. Each row represent one matched groups.
            res[0]['avg_outcome_control']: 
                average of control units' outcomes in each matched group   
            res[0]['avg_outcome_treated']: 
                average of treated units' outcomes in each matched group   
            res[0]['num_control']:
                the number of control units in each matched group
            res[0]['num_treated']:
                the number of treated units in each matched group
            res[0]['is_matched']:
                the level each matched group belongs to
        res[1]:
            a list of level numbers where we have matched groups
        res[2]:
            a list of covariate names that we dropped
```
#### Post Analysis
```
ATE_db(res)
ATT_db(res)
```
