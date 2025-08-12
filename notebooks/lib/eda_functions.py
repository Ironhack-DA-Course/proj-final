import pandas as pd
import numpy as np

def detect_iqr_outliers(df: pd.DataFrame, column, floor, ceiling):
    '''
    floor: percent
    ceiling; percent
    '''
    
    df[column] = pd.to_numeric(df[column], errors='coerce')
    Q1 = df[column].quantile(floor)
    Q3 = df[column].quantile(ceiling)
    IQR = Q3 - Q1
    
    if df[column].dtype.name in ["int64", "int32"]:
        lower_bound = np.floor(Q1 - 1.5 * IQR)
        upper_bound = np.ceil(Q3 + 1.5 * IQR)
    else:
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

    if lower_bound < 0: lower_bound = 0

    lower_out_indexes = df[ df[column] <  lower_bound].index
    upper_out_indexes = df[ df[column] >  upper_bound].index

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    # out=[]
    # for x in df[column]:
    #     if x > upper_bound or x < lower_bound:
    #         out.append(x)
        
    return outliers, lower_bound, lower_out_indexes, upper_bound, upper_out_indexes



def detect_winsor_outliers(df: pd.DataFrame, column, floor, ceiling):
    '''
    floor: percentage *100
    ceiling: percentage *100
    '''
    q1 = np.percentile(df , floor)
    q3 = np.percentile(df , ceiling)

    lower_out_indexes = df[ df[column] <  q1].index
    upper_out_indexes = df[ df[column] >  q3].index

    outliers = df[(df[column] < q1) | (df[column] > q3)]

    return outliers, q1, lower_out_indexes, q3, upper_out_indexes


