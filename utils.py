import numpy as np 
import pandas as pd 
from copy import deepcopy
import parse
from metrics import ScaledError


def remove_null_labels(X_raw,y_raw,tseries_raw):
    X = deepcopy(X_raw)
    y = deepcopy(y_raw)
    tseries = deepcopy(tseries_raw)

    # remove missing labels
    idx = ~y.isnull()
    X = X.loc[idx].reset_index(drop=True)
    y = y.loc[idx].values
    tseries = tseries.loc[idx].values
    
    return X,y,tseries

    
    
def scale_data(data,scale_path):
    
    avg_plus_one = pd.read_pickle(scale_path)
    scaled_data = deepcopy(data)

    for pattern in avg_plus_one.columns:
        cols = [c for c in scaled_data.columns if c[:len(pattern)] == pattern]
        scales = avg_plus_one.loc[scaled_data.ts_id,pattern].values.reshape(-1,1)
        scaled_data[cols] /= scales

    return scaled_data