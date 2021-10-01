[![Build Status](https://travis-ci.org/almost-matching-exactly/DAME-FLAME-Python-Package.svg?branch=master)](https://travis-ci.org/almost-matching-exactly/DAME-FLAME-Python-Package)
[![Coverage Status](https://coveralls.io/repos/github/almost-matching-exactly/DAME-FLAME-Python-Package/badge.svg)](https://coveralls.io/github/almost-matching-exactly/DAME-FLAME-Python-Package)

# 

# FLAME_db
A Python package for performing matching for observational causal inference on datasets containing discrete covariates via Database
--------------------------------------------------

## Documentation [here](https://almost-matching-exactly.github.io/DAME-FLAME-Python-Package/)

FLAME_db is a Python package for performing matching for observational causal inference on datasets containing discrete covariates. 
It implements the Fast, Large-Scale Almost Matching Exactly (FLAME) algorithms using SQL quries, which match treatment and control units on subsets of the covariates. FLAME_db scales to huge datasets with millions of observations where existing state-of-the-art methods fail, and that it achieves significantly better performance than other matching methods. The resulting matched groups are interpretable, because the matches are made on covariates, and high-quality, because machine learning is used to determine which covariates are important to match on.

--------------------------------------------------
#### Make toy dataset 
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
select_db = "postgreSQL"  # Select the database you are using: "MySQL", "postgreSQL","Microsoft SQL server"
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
    data frame of matched groups. Each row represent one matched groups.
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
#### Postprocessing

```
ATE_db(res) # Get ATE for the whole dataset
ATT_db(res) # Get ATT for the whole dataset
```
