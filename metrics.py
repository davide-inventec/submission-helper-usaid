from abc import ABC
import pandas as pd 
import numpy as np

class Metric(ABC):

    def __call__(self,label,pred,ts_ids):
        pass



class ScaledError(Metric):
    """NB: remove 0 scales"""
    def __init__(self,scales, output_shape="valid"):
        self.scales = scales
        self.output_shape = output_shape

    def __call__(self,label,pred,ts_ids):
        """ts_ids is ignored"""
        scaling = self.scales[ts_ids]
        if self.output_shape == "valid":
            # Return the valid errors, this is useful for direct aggregation
            return np.abs(label - pred)[scaling > 0] / scaling[scaling > 0]
        else:
            # Return the same number of samples, mainly used for structuring outputs etc
            # Simply set the error to -1 to flag the error is not available
            error = np.abs(label - pred)/scaling
            error[scaling <= 0] = np.nan
            return error
        
class ScaledErrorByTS(Metric):
    """NB: remove 0 scales"""
    def __init__(self,scales):
        self.scales = scales

    def __call__(self,label,pred,ts_ids):

        scaled_errors = []
        for ts_id in np.unique(ts_ids):

            scale = self.scales[ts_id]  # This is expected to be a scalar
            idx = np.where(ts_ids == ts_id)[0]  # Get the labels and predictions for the specific ts_id
            abs_error = np.abs(label[idx] - pred[idx])
            
            scaled_errors.append((abs_error / scale).mean() if scale > 0 else np.nan)
            
        return pd.Series(scaled_errors, index = np.unique(ts_ids))

      
class SMAPE(Metric):
    def __call__(self,label,pred,ts_ids):
        """ts_ids is ignored"""
        err = np.abs(label - pred)
        scale = (np.abs(label) + np.abs(pred))
        scale[scale==0] = 1 # it means that if both prediction and labels are 0, smape is 0
        return  err / scale 


class MAE(Metric):
    def __call__(self,label,pred,ts_ids):
        """ts_ids is ignored"""
        return np.abs(label - pred)


class MLAE(Metric):
    def __call__(self,label,pred,ts_ids):
        """ts_ids is ignored"""
        return np.log(np.abs(label - pred)  + 1)