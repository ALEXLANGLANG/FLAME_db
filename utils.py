
#Calculate ATT for the whole dataset
def ATT_db(res_post_new):
    df_matched = res_post_new
    weight_sum = 0
    weight_TT_sum = 0
    for i in range(len(df_matched)):
        for j in range(df_matched[i].shape[0]):
            MG = df_matched[i].iloc[j]
            MG_weight = MG['num_control']
            num_MG_treated =  MG['num_treated']
            mean_Y1 = MG['avg_outcome_treated']
            mean_Y0 = MG['avg_outcome_control']
            weight_sum = weight_sum + MG_weight
            weight_TT_sum += MG_weight*(mean_Y1-mean_Y0)
            
    ATT = weight_TT_sum/weight_sum
    return ATT



#Calculate ATE for the whole dataset
def ATE_db(res_post_new):
    
    df_matched= res_post_new
    weight_sum = 0; 
    weight_CATE_sum = 0
    for i in range(len(df_matched)):
        for j in range(df_matched[i].shape[0]):
            MG = df_matched[i].iloc[j]
            CATE = MG['avg_outcome_treated'] - MG['avg_outcome_control']
            MG_weight = MG['num_control'] + MG['num_treated']


            weight_sum +=  MG_weight
            weight_CATE_sum += MG_weight*CATE
            
    ATE = weight_CATE_sum/weight_sum

    return ATE