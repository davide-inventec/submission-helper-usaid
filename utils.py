import numpy as np 
import pandas as pd 
from copy import deepcopy
import parse
from metrics import ScaledError

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print ("Toc: start time not set")

def recursive_update(model,X,horizon,engineer=None):
    """
    recursively shift all time-varying columns and update label_minus_0 field.
    engineer: None or callable
    """
    # save patterns identifying time-varying predictors
    all_cols = X.columns.values
    col_patterns = np.unique(([v[:-2] for v in all_cols if "_minus_" in v or "_plus_" in v]))
    cols_to_shift = [[v for v in all_cols if pattern in v] for pattern in col_patterns]
    
    # recursively shift all time-varying columns and update label_minus_0 field 
    for _ in range(horizon - 1):
        
        pred = model.predict(X if engineer is None else engineer(X))
        for cols in cols_to_shift:
            X[cols] = X[cols].shift(periods=1,axis=1)
        X["label_minus_0"] = pred
    return X


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


class HorizontalSplit:
    def __init__(self,tseries,n_splits=5,seed=14):
        
        self.tseries = tseries
        np.random.seed(seed)
        ts_ids = tseries.unique()
        # set number of time-series in each fold
        splits_size = ts_ids.shape[0] // n_splits
        mod = ts_ids.shape[0] % n_splits
        # create a list of lists of unique ts_id. 
        # each list correspond to a fold
        self.ts_folds = []
        for _ in range(n_splits-1):
            size = splits_size
            if mod > 0:
                size += 1
                mod -= 1
            sel = np.random.choice(ts_ids,size=size,replace=False)
            self.ts_folds.append(sel)
            ts_ids = list(set(ts_ids) - set(sel))
        self.ts_folds.append(np.array(ts_ids))   
        
        
    def split(self,X,y):
        for fold_ts in self.ts_folds:
            idx_train = np.where(~self.tseries.isin(fold_ts))[0]
            idx_valid = np.where(self.tseries.isin(fold_ts))[0]
            yield (idx_train,idx_valid)
            
            

def print_scores(scores,stat = "mean"):
    
    assert stat in {"mean","std"}
    
    if stat == "mean":
        stat_computer = np.mean
    elif stat == "std":
        stat_computer = np.std

    print("[" + stat.upper() + "]", end = "")
    if stat == "std": print(" ", end = "")
    for score in scores.keys():
        print(" " * (15 - len(score)),score.upper(), end = "")
    print()

    for mode in scores[score].keys():

        print(mode + " ", end = "")
        for score in scores.keys():
            m = stat_computer(scores[score][mode])
            print(" "*(15 - len(str(round(m,3)))),round(m,3), end = "")
        print()
    print()
    
    
def scale_data(data,scale_path):
    
    avg_plus_one = pd.read_pickle(scale_path)
    scaled_data = deepcopy(data)

    for pattern in avg_plus_one.columns:
        cols = [c for c in scaled_data.columns if c[:len(pattern)] == pattern]
        scales = avg_plus_one.loc[scaled_data.ts_id,pattern].values.reshape(-1,1)
        scaled_data[cols] /= scales

    return scaled_data
    
    
    
def summarize(train_data, predictions, loader):
    # Get all the correct dates and ts_id
    correct_dates = loader.main_data[["ts_id","date"]].drop_duplicates().reset_index()

    # Get the reduced DataFrame
    X, y, tseries = train_data
    summary = []
    metrics["same_mase"] = ScaledError(pd.read_pickle(scale_path), output_shape="same")
    for k in range(5): 
        fold_df = []
        for mode in ["train","valid"]:
            pred = predictions[k][mode]["pred"]
            label = predictions[k][mode]["label"]
            idx = predictions[k][mode]["idx"]
            
            # We are using multiple data sources, make sure that they are synced
            assert (label == y[idx]).all()
            error = metrics["MAE"](label, pred, tseries[idx])
            
            df = pd.DataFrame({
                "ts_id": tseries[idx],
                "date": X["date"].values[idx],
                "fold": int(k),
                "mode": mode,
                "label": label,
                "pred": pred, 
            })
            
            # Compute errors based on the selected metrics
            for m in ["same_mase","modmase","MAE"]:
                error = metrics[m](label, pred, tseries[idx])
                
                # The metric might return a pd.Series or numpy array
                try: df["error_{}".format(m)] = error
                except: df["error_{}".format(m)] = error.to_numpy()
            
            fold_df.append(df)
            
        fold_df = pd.concat(fold_df, axis=0)
        
        # Correct the number of dates for all of the time series ID
        fold_df = pd.merge(correct_dates, fold_df, on=["ts_id","date"], how="left")
        fold_df["fold"] = k
        fold_df.drop(columns="index",inplace=True)
        # Add the corrected data to the summary
        summary.append(fold_df)
        
    summary = pd.concat(summary, axis=0).sort_values("date").reset_index(drop=True)
    return summary
    
class FeatureBuilder:
    def __init__(self):
        self.available_features = {}
        self.available_features["main"] = ['label_minus_43','label_minus_42','label_minus_41','label_minus_40','label_minus_39','label_minus_38','label_minus_37','label_minus_36','label_minus_35','label_minus_34','label_minus_33','label_minus_32','label_minus_31','label_minus_30','label_minus_29','label_minus_28','label_minus_27','label_minus_26','label_minus_25','label_minus_24','label_minus_23','label_minus_22','label_minus_21','label_minus_20','label_minus_19','label_minus_18','label_minus_17','label_minus_16','label_minus_15','label_minus_14','label_minus_13','label_minus_12','label_minus_11','label_minus_10','label_minus_9','label_minus_8','label_minus_7','label_minus_6','label_minus_5','label_minus_4','label_minus_3','label_minus_2','label_minus_1','label_minus_0','stock_initial_minus_43','stock_initial_minus_42','stock_initial_minus_41','stock_initial_minus_40','stock_initial_minus_39','stock_initial_minus_38','stock_initial_minus_37','stock_initial_minus_36','stock_initial_minus_35','stock_initial_minus_34','stock_initial_minus_33','stock_initial_minus_32','stock_initial_minus_31','stock_initial_minus_30','stock_initial_minus_29','stock_initial_minus_28','stock_initial_minus_27','stock_initial_minus_26','stock_initial_minus_25','stock_initial_minus_24','stock_initial_minus_23','stock_initial_minus_22','stock_initial_minus_21','stock_initial_minus_20','stock_initial_minus_19','stock_initial_minus_18','stock_initial_minus_17','stock_initial_minus_16','stock_initial_minus_15','stock_initial_minus_14','stock_initial_minus_13','stock_initial_minus_12','stock_initial_minus_11','stock_initial_minus_10','stock_initial_minus_9','stock_initial_minus_8','stock_initial_minus_7','stock_initial_minus_6','stock_initial_minus_5','stock_initial_minus_4','stock_initial_minus_3','stock_initial_minus_2','stock_initial_minus_1','stock_initial_minus_0','stock_received_minus_43','stock_received_minus_42','stock_received_minus_41','stock_received_minus_40','stock_received_minus_39','stock_received_minus_38','stock_received_minus_37','stock_received_minus_36','stock_received_minus_35','stock_received_minus_34','stock_received_minus_33','stock_received_minus_32','stock_received_minus_31','stock_received_minus_30','stock_received_minus_29','stock_received_minus_28','stock_received_minus_27','stock_received_minus_26','stock_received_minus_25','stock_received_minus_24','stock_received_minus_23','stock_received_minus_22','stock_received_minus_21','stock_received_minus_20','stock_received_minus_19','stock_received_minus_18','stock_received_minus_17','stock_received_minus_16','stock_received_minus_15','stock_received_minus_14','stock_received_minus_13','stock_received_minus_12','stock_received_minus_11','stock_received_minus_10','stock_received_minus_9','stock_received_minus_8','stock_received_minus_7','stock_received_minus_6','stock_received_minus_5','stock_received_minus_4','stock_received_minus_3','stock_received_minus_2','stock_received_minus_1','stock_received_minus_0','stock_end_minus_43','stock_end_minus_42','stock_end_minus_41','stock_end_minus_40','stock_end_minus_39','stock_end_minus_38','stock_end_minus_37','stock_end_minus_36','stock_end_minus_35','stock_end_minus_34','stock_end_minus_33','stock_end_minus_32','stock_end_minus_31','stock_end_minus_30','stock_end_minus_29','stock_end_minus_28','stock_end_minus_27','stock_end_minus_26','stock_end_minus_25','stock_end_minus_24','stock_end_minus_23','stock_end_minus_22','stock_end_minus_21','stock_end_minus_20','stock_end_minus_19','stock_end_minus_18','stock_end_minus_17','stock_end_minus_16','stock_end_minus_15','stock_end_minus_14','stock_end_minus_13','stock_end_minus_12','stock_end_minus_11','stock_end_minus_10','stock_end_minus_9','stock_end_minus_8','stock_end_minus_7','stock_end_minus_6','stock_end_minus_5','stock_end_minus_4','stock_end_minus_3','stock_end_minus_2','stock_end_minus_1','stock_end_minus_0','average_monthly_consumption_minus_43','average_monthly_consumption_minus_42','average_monthly_consumption_minus_41','average_monthly_consumption_minus_40','average_monthly_consumption_minus_39','average_monthly_consumption_minus_38','average_monthly_consumption_minus_37','average_monthly_consumption_minus_36','average_monthly_consumption_minus_35','average_monthly_consumption_minus_34','average_monthly_consumption_minus_33','average_monthly_consumption_minus_32','average_monthly_consumption_minus_31','average_monthly_consumption_minus_30','average_monthly_consumption_minus_29','average_monthly_consumption_minus_28','average_monthly_consumption_minus_27','average_monthly_consumption_minus_26','average_monthly_consumption_minus_25','average_monthly_consumption_minus_24','average_monthly_consumption_minus_23','average_monthly_consumption_minus_22','average_monthly_consumption_minus_21','average_monthly_consumption_minus_20','average_monthly_consumption_minus_19','average_monthly_consumption_minus_18','average_monthly_consumption_minus_17','average_monthly_consumption_minus_16','average_monthly_consumption_minus_15','average_monthly_consumption_minus_14','average_monthly_consumption_minus_13','average_monthly_consumption_minus_12','average_monthly_consumption_minus_11','average_monthly_consumption_minus_10','average_monthly_consumption_minus_9','average_monthly_consumption_minus_8','average_monthly_consumption_minus_7','average_monthly_consumption_minus_6','average_monthly_consumption_minus_5','average_monthly_consumption_minus_4','average_monthly_consumption_minus_3','average_monthly_consumption_minus_2','average_monthly_consumption_minus_1','average_monthly_consumption_minus_0','stock_stockout_days_minus_43','stock_stockout_days_minus_42','stock_stockout_days_minus_41','stock_stockout_days_minus_40','stock_stockout_days_minus_39','stock_stockout_days_minus_38','stock_stockout_days_minus_37','stock_stockout_days_minus_36','stock_stockout_days_minus_35','stock_stockout_days_minus_34','stock_stockout_days_minus_33','stock_stockout_days_minus_32','stock_stockout_days_minus_31','stock_stockout_days_minus_30','stock_stockout_days_minus_29','stock_stockout_days_minus_28','stock_stockout_days_minus_27','stock_stockout_days_minus_26','stock_stockout_days_minus_25','stock_stockout_days_minus_24','stock_stockout_days_minus_23','stock_stockout_days_minus_22','stock_stockout_days_minus_21','stock_stockout_days_minus_20','stock_stockout_days_minus_19','stock_stockout_days_minus_18','stock_stockout_days_minus_17','stock_stockout_days_minus_16','stock_stockout_days_minus_15','stock_stockout_days_minus_14','stock_stockout_days_minus_13','stock_stockout_days_minus_12','stock_stockout_days_minus_11','stock_stockout_days_minus_10','stock_stockout_days_minus_9','stock_stockout_days_minus_8','stock_stockout_days_minus_7','stock_stockout_days_minus_6','stock_stockout_days_minus_5','stock_stockout_days_minus_4','stock_stockout_days_minus_3','stock_stockout_days_minus_2','stock_stockout_days_minus_1','stock_stockout_days_minus_0','stock_ordered_minus_43','stock_ordered_minus_42','stock_ordered_minus_41','stock_ordered_minus_40','stock_ordered_minus_39','stock_ordered_minus_38','stock_ordered_minus_37','stock_ordered_minus_36','stock_ordered_minus_35','stock_ordered_minus_34','stock_ordered_minus_33','stock_ordered_minus_32','stock_ordered_minus_31','stock_ordered_minus_30','stock_ordered_minus_29','stock_ordered_minus_28','stock_ordered_minus_27','stock_ordered_minus_26','stock_ordered_minus_25','stock_ordered_minus_24','stock_ordered_minus_23','stock_ordered_minus_22','stock_ordered_minus_21','stock_ordered_minus_20','stock_ordered_minus_19','stock_ordered_minus_18','stock_ordered_minus_17','stock_ordered_minus_16','stock_ordered_minus_15','stock_ordered_minus_14','stock_ordered_minus_13','stock_ordered_minus_12','stock_ordered_minus_11','stock_ordered_minus_10','stock_ordered_minus_9','stock_ordered_minus_8','stock_ordered_minus_7','stock_ordered_minus_6','stock_ordered_minus_5','stock_ordered_minus_4','stock_ordered_minus_3','stock_ordered_minus_2','stock_ordered_minus_1','stock_ordered_minus_0','invalid_0_features_minus_43','invalid_0_features_minus_42','invalid_0_features_minus_41','invalid_0_features_minus_40','invalid_0_features_minus_39','invalid_0_features_minus_38','invalid_0_features_minus_37','invalid_0_features_minus_36','invalid_0_features_minus_35','invalid_0_features_minus_34','invalid_0_features_minus_33','invalid_0_features_minus_32','invalid_0_features_minus_31','invalid_0_features_minus_30','invalid_0_features_minus_29','invalid_0_features_minus_28','invalid_0_features_minus_27','invalid_0_features_minus_26','invalid_0_features_minus_25','invalid_0_features_minus_24','invalid_0_features_minus_23','invalid_0_features_minus_22','invalid_0_features_minus_21','invalid_0_features_minus_20','invalid_0_features_minus_19','invalid_0_features_minus_18','invalid_0_features_minus_17','invalid_0_features_minus_16','invalid_0_features_minus_15','invalid_0_features_minus_14','invalid_0_features_minus_13','invalid_0_features_minus_12','invalid_0_features_minus_11','invalid_0_features_minus_10','invalid_0_features_minus_9','invalid_0_features_minus_8','invalid_0_features_minus_7','invalid_0_features_minus_6','invalid_0_features_minus_5','invalid_0_features_minus_4','invalid_0_features_minus_3','invalid_0_features_minus_2','invalid_0_features_minus_1','invalid_0_features_minus_0','labels1','labels2','labels3','site_code','product_code','region','district','month','year']
        self.available_features["worldpop"] = ['pop_1km_children_plus_6', 'pop_1km_children_plus_7', 'pop_1km_children_plus_8', 'pop_1km_children_plus_9', 'pop_1km_children_plus_10', 'pop_1km_children_plus_11', 'pop_5km_children_plus_6', 'pop_5km_children_plus_7', 'pop_5km_children_plus_8', 'pop_5km_children_plus_9', 'pop_5km_children_plus_10', 'pop_5km_children_plus_11', 'pop_10km_children_plus_6', 'pop_10km_children_plus_7', 'pop_10km_children_plus_8', 'pop_10km_children_plus_9', 'pop_10km_children_plus_10', 'pop_10km_children_plus_11', 'pop_20km_children_plus_6', 'pop_20km_children_plus_7', 'pop_20km_children_plus_8', 'pop_20km_children_plus_9', 'pop_20km_children_plus_10', 'pop_20km_children_plus_11', 'pop_1km_youth_male_plus_0', 'pop_1km_youth_male_plus_1', 'pop_1km_youth_male_plus_2', 'pop_1km_youth_male_plus_3', 'pop_5km_youth_male_plus_0', 'pop_5km_youth_male_plus_1', 'pop_5km_youth_male_plus_2', 'pop_5km_youth_male_plus_3', 'pop_10km_youth_male_plus_0', 'pop_10km_youth_male_plus_1', 'pop_10km_youth_male_plus_2', 'pop_10km_youth_male_plus_3', 'pop_20km_youth_male_plus_0', 'pop_20km_youth_male_plus_1', 'pop_20km_youth_male_plus_2', 'pop_20km_youth_male_plus_3', 'pop_1km_youth_female_plus_0', 'pop_1km_youth_female_plus_1', 'pop_1km_youth_female_plus_2', 'pop_1km_youth_female_plus_3', 'pop_5km_youth_female_plus_0', 'pop_5km_youth_female_plus_1', 'pop_5km_youth_female_plus_2', 'pop_5km_youth_female_plus_3', 'pop_10km_youth_female_plus_0', 'pop_10km_youth_female_plus_1', 'pop_10km_youth_female_plus_2', 'pop_10km_youth_female_plus_3', 'pop_20km_youth_female_plus_0', 'pop_20km_youth_female_plus_1', 'pop_20km_youth_female_plus_2', 'pop_20km_youth_female_plus_3', 'pop_1km_adult_male_plus_0', 'pop_1km_adult_male_plus_1', 'pop_1km_adult_male_plus_2', 'pop_1km_adult_male_plus_3', 'pop_5km_adult_male_plus_0', 'pop_5km_adult_male_plus_1', 'pop_5km_adult_male_plus_2', 'pop_5km_adult_male_plus_3', 'pop_10km_adult_male_plus_0', 'pop_10km_adult_male_plus_1', 'pop_10km_adult_male_plus_2', 'pop_10km_adult_male_plus_3', 'pop_20km_adult_male_plus_0', 'pop_20km_adult_male_plus_1', 'pop_20km_adult_male_plus_2', 'pop_20km_adult_male_plus_3', 'pop_1km_adult_female_plus_0', 'pop_1km_adult_female_plus_1', 'pop_1km_adult_female_plus_2', 'pop_1km_adult_female_plus_3', 'pop_5km_adult_female_plus_0', 'pop_5km_adult_female_plus_1', 'pop_5km_adult_female_plus_2', 'pop_5km_adult_female_plus_3', 'pop_10km_adult_female_plus_0', 'pop_10km_adult_female_plus_1', 'pop_10km_adult_female_plus_2', 'pop_10km_adult_female_plus_3', 'pop_20km_adult_female_plus_0', 'pop_20km_adult_female_plus_1', 'pop_20km_adult_female_plus_2', 'pop_20km_adult_female_plus_3']
        self.available_features["awa"] = ['awa_all_ages', 'awa_Lower', 'awa_Upper','awa_females_adults_(15+)', 'awa_females_adults_(15+)_lower','awa_females_adults_(15+)_upper', 'awa_males_adults_(15+)','awa_males_adults_(15+)_lower', 'awa_males_adults_(15+)_upper','awa_females_10-19', 'awa_females_10-19_lower','awa_females_10-19_upper', 'awa_males_10-19', 'awa_males_10-19_lower','awa_males_10-19_upper', 'awa_females_young_people_(15-24)','awa_females_young_people_(15-24)_lower','awa_females_young_people_(15-24)_upper','awa_males_young_people_(15-24)','awa_males_young_people_(15-24)_lower','awa_males_young_people_(15-24)_upper', 'awa_females_50+','awa_females_50+_lower', 'awa_females_50+_upper', 'awa_males_50+','awa_males_50+_lower', 'awa_males_50+_upper']
        self.available_features["contra"] = ['implant_women_old_minus_0','implant_women_new_minus_0', 'injection2_women_old_minus_0','injection2_women_new_minus_0', 'injection3_women_old_minus_0','injection3_women_new_minus_0', 'pill_women_old_minus_0','pill_women_new_minus_0', 'iud_women_old_minus_0','iud_women_new_minus_0', 'iud_number_dispensed_minus_0','implant_number_dispensed_minus_0','injection2_number_dispensed_minus_0','injection3_number_dispensed_minus_0', 'pill_number_dispensed_minus_0','iud_number_received_minus_0', 'implant_number_received_minus_0','injection2_number_received_minus_0','injection3_number_received_minus_0', 'pill_number_received_minus_0','iud_stock_end_minus_0', 'implant_stock_end_minus_0','injection2_stock_end_minus_0', 'injection3_stock_end_minus_0','pill_stock_end_minus_0', 'implant_women_old_minus_1','implant_women_new_minus_1', 'injection2_women_old_minus_1','injection2_women_new_minus_1', 'injection3_women_old_minus_1','injection3_women_new_minus_1', 'pill_women_old_minus_1','pill_women_new_minus_1', 'iud_women_old_minus_1','iud_women_new_minus_1', 'iud_number_dispensed_minus_1','implant_number_dispensed_minus_1','injection2_number_dispensed_minus_1','injection3_number_dispensed_minus_1', 'pill_number_dispensed_minus_1','iud_number_received_minus_1', 'implant_number_received_minus_1','injection2_number_received_minus_1','injection3_number_received_minus_1', 'pill_number_received_minus_1','iud_stock_end_minus_1', 'implant_stock_end_minus_1','injection2_stock_end_minus_1', 'injection3_stock_end_minus_1','pill_stock_end_minus_1', 'implant_women_old_minus_2','implant_women_new_minus_2', 'injection2_women_old_minus_2','injection2_women_new_minus_2', 'injection3_women_old_minus_2','injection3_women_new_minus_2', 'pill_women_old_minus_2','pill_women_new_minus_2', 'iud_women_old_minus_2','iud_women_new_minus_2', 'iud_number_dispensed_minus_2','implant_number_dispensed_minus_2','injection2_number_dispensed_minus_2','injection3_number_dispensed_minus_2', 'pill_number_dispensed_minus_2','iud_number_received_minus_2', 'implant_number_received_minus_2','injection2_number_received_minus_2','injection3_number_received_minus_2', 'pill_number_received_minus_2','iud_stock_end_minus_2', 'implant_stock_end_minus_2','injection2_stock_end_minus_2', 'injection3_stock_end_minus_2','pill_stock_end_minus_2']
        self.available_features["national_holiday"] = ['holiday_event_plus_1', 'holiday_event_plus_2','holiday_event_plus_3']
        self.available_features["lagged"] = ['region_AS21126_minus_2', 'region_AS21126_minus_1','region_AS21126_minus_0', 'region_AS27000_minus_2','region_AS27000_minus_1', 'region_AS27000_minus_0','region_AS27132_minus_2', 'region_AS27132_minus_1','region_AS27132_minus_0', 'region_AS27133_minus_2','region_AS27133_minus_1', 'region_AS27133_minus_0','region_AS27134_minus_2', 'region_AS27134_minus_1','region_AS27134_minus_0', 'region_AS27137_minus_2','region_AS27137_minus_1', 'region_AS27137_minus_0','region_AS27138_minus_2', 'region_AS27138_minus_1','region_AS27138_minus_0', 'region_AS27139_minus_2','region_AS27139_minus_1', 'region_AS27139_minus_0','region_AS42018_minus_2', 'region_AS42018_minus_1','region_AS42018_minus_0', 'region_AS17005_minus_2','region_AS17005_minus_1', 'region_AS17005_minus_0','region_AS46000_minus_2', 'region_AS46000_minus_1','region_AS46000_minus_0','site_code_AS21126_minus_2','site_code_AS21126_minus_1', 'site_code_AS21126_minus_0','site_code_AS27000_minus_2', 'site_code_AS27000_minus_1','site_code_AS27000_minus_0', 'site_code_AS27132_minus_2','site_code_AS27132_minus_1', 'site_code_AS27132_minus_0','site_code_AS27133_minus_2', 'site_code_AS27133_minus_1','site_code_AS27133_minus_0', 'site_code_AS27134_minus_2','site_code_AS27134_minus_1', 'site_code_AS27134_minus_0','site_code_AS27137_minus_2', 'site_code_AS27137_minus_1','site_code_AS27137_minus_0', 'site_code_AS27138_minus_2','site_code_AS27138_minus_1', 'site_code_AS27138_minus_0','site_code_AS27139_minus_2', 'site_code_AS27139_minus_1','site_code_AS27139_minus_0', 'site_code_AS42018_minus_2','site_code_AS42018_minus_1', 'site_code_AS42018_minus_0','site_code_AS17005_minus_2', 'site_code_AS17005_minus_1','site_code_AS17005_minus_0', 'site_code_AS46000_minus_2','site_code_AS46000_minus_1', 'site_code_AS46000_minus_0','district_AS21126_minus_2','district_AS21126_minus_1', 'district_AS21126_minus_0','district_AS27000_minus_2', 'district_AS27000_minus_1','district_AS27000_minus_0', 'district_AS27132_minus_2','district_AS27132_minus_1', 'district_AS27132_minus_0','district_AS27133_minus_2', 'district_AS27133_minus_1','district_AS27133_minus_0', 'district_AS27134_minus_2','district_AS27134_minus_1', 'district_AS27134_minus_0','district_AS27137_minus_2', 'district_AS27137_minus_1','district_AS27137_minus_0', 'district_AS27138_minus_2','district_AS27138_minus_1', 'district_AS27138_minus_0','district_AS27139_minus_2', 'district_AS27139_minus_1','district_AS27139_minus_0', 'district_AS42018_minus_2','district_AS42018_minus_1', 'district_AS42018_minus_0','district_AS17005_minus_2', 'district_AS17005_minus_1','district_AS17005_minus_0', 'district_AS46000_minus_2','district_AS46000_minus_1', 'district_AS46000_minus_0']
        self.available_features["hierarchical"] = ['product_code_label_minus_43','product_code_label_minus_42', 'product_code_label_minus_41','product_code_label_minus_40', 'product_code_label_minus_39','product_code_label_minus_38', 'product_code_label_minus_37','product_code_label_minus_36', 'product_code_label_minus_35','product_code_label_minus_34', 'product_code_label_minus_33','product_code_label_minus_32', 'product_code_label_minus_31','product_code_label_minus_30', 'product_code_label_minus_29','product_code_label_minus_28', 'product_code_label_minus_27','product_code_label_minus_26', 'product_code_label_minus_25','product_code_label_minus_24', 'product_code_label_minus_23','product_code_label_minus_22', 'product_code_label_minus_21','product_code_label_minus_20', 'product_code_label_minus_19','product_code_label_minus_18', 'product_code_label_minus_17','product_code_label_minus_16', 'product_code_label_minus_15','product_code_label_minus_14', 'product_code_label_minus_13','product_code_label_minus_12', 'product_code_label_minus_11','product_code_label_minus_10', 'product_code_label_minus_9','product_code_label_minus_8', 'product_code_label_minus_7','product_code_label_minus_6', 'product_code_label_minus_5','product_code_label_minus_4', 'product_code_label_minus_3','product_code_label_minus_2', 'product_code_label_minus_1','product_code_label_minus_0','product_code-district_label_minus_43','product_code-district_label_minus_42','product_code-district_label_minus_41','product_code-district_label_minus_40','product_code-district_label_minus_39','product_code-district_label_minus_38','product_code-district_label_minus_37','product_code-district_label_minus_36','product_code-district_label_minus_35','product_code-district_label_minus_34','product_code-district_label_minus_33','product_code-district_label_minus_32','product_code-district_label_minus_31','product_code-district_label_minus_30','product_code-district_label_minus_29','product_code-district_label_minus_28','product_code-district_label_minus_27','product_code-district_label_minus_26','product_code-district_label_minus_25','product_code-district_label_minus_24','product_code-district_label_minus_23','product_code-district_label_minus_22','product_code-district_label_minus_21','product_code-district_label_minus_20','product_code-district_label_minus_19','product_code-district_label_minus_18','product_code-district_label_minus_17','product_code-district_label_minus_16','product_code-district_label_minus_15','product_code-district_label_minus_14','product_code-district_label_minus_13','product_code-district_label_minus_12','product_code-district_label_minus_11','product_code-district_label_minus_10','product_code-district_label_minus_9','product_code-district_label_minus_8','product_code-district_label_minus_7','product_code-district_label_minus_6','product_code-district_label_minus_5','product_code-district_label_minus_4','product_code-district_label_minus_3','product_code-district_label_minus_2','product_code-district_label_minus_1','product_code-district_label_minus_0','product_code-region_label_minus_43','product_code-region_label_minus_42','product_code-region_label_minus_41','product_code-region_label_minus_40','product_code-region_label_minus_39','product_code-region_label_minus_38','product_code-region_label_minus_37','product_code-region_label_minus_36','product_code-region_label_minus_35','product_code-region_label_minus_34','product_code-region_label_minus_33','product_code-region_label_minus_32','product_code-region_label_minus_31','product_code-region_label_minus_30','product_code-region_label_minus_29','product_code-region_label_minus_28','product_code-region_label_minus_27','product_code-region_label_minus_26','product_code-region_label_minus_25','product_code-region_label_minus_24','product_code-region_label_minus_23','product_code-region_label_minus_22','product_code-region_label_minus_21','product_code-region_label_minus_20','product_code-region_label_minus_19','product_code-region_label_minus_18','product_code-region_label_minus_17','product_code-region_label_minus_16','product_code-region_label_minus_15','product_code-region_label_minus_14','product_code-region_label_minus_13','product_code-region_label_minus_12','product_code-region_label_minus_11','product_code-region_label_minus_10','product_code-region_label_minus_9','product_code-region_label_minus_8','product_code-region_label_minus_7','product_code-region_label_minus_6','product_code-region_label_minus_5','product_code-region_label_minus_4','product_code-region_label_minus_3','product_code-region_label_minus_2','product_code-region_label_minus_1','product_code-region_label_minus_0','site_code_label_minus_43','site_code_label_minus_42', 'site_code_label_minus_41','site_code_label_minus_40', 'site_code_label_minus_39','site_code_label_minus_38', 'site_code_label_minus_37','site_code_label_minus_36', 'site_code_label_minus_35','site_code_label_minus_34', 'site_code_label_minus_33','site_code_label_minus_32', 'site_code_label_minus_31','site_code_label_minus_30', 'site_code_label_minus_29','site_code_label_minus_28', 'site_code_label_minus_27','site_code_label_minus_26', 'site_code_label_minus_25','site_code_label_minus_24', 'site_code_label_minus_23','site_code_label_minus_22', 'site_code_label_minus_21','site_code_label_minus_20', 'site_code_label_minus_19','site_code_label_minus_18', 'site_code_label_minus_17','site_code_label_minus_16', 'site_code_label_minus_15','site_code_label_minus_14', 'site_code_label_minus_13','site_code_label_minus_12', 'site_code_label_minus_11','site_code_label_minus_10', 'site_code_label_minus_9','site_code_label_minus_8', 'site_code_label_minus_7','site_code_label_minus_6', 'site_code_label_minus_5','site_code_label_minus_4', 'site_code_label_minus_3','site_code_label_minus_2', 'site_code_label_minus_1','site_code_label_minus_0']
        
    def _get_start_end(self, data):
        try: _, start, end = parse.parse("{}[{}:{}]", data)
        except:
            try: start, end, _ = parse.parse("[{}:{}]{}", data)
            except: 
                try: _, start, end, _ = parse.parse("{}[{}:{}]{}", data)
                except: start, end = None, None
        return start, end
    
    def _unpack_filters(self, filters):
        unpacked_filters = []
        for item in filters:
            start, end = self._get_start_end(item)
            if start is None: 
                unpacked_filters.append(item)
                continue

            start = int(start)
            end = int(end)
            substring = "[{}:{}]".format(start, end)
            for i in range(start, end):
                unpacked_filters.append(item.replace(substring, str(i)))
        return unpacked_filters

    def _normalize_digits(self, data):
        output = []
        if type(data) != type([]): data = [data]
        for f in data:
            parsed = parse.parse("{}_{}", f[-4:])
            if parsed is None:
                output.append(f)
                continue
            number = int(parsed[-1])
            if number < 10:  output.append(f[:-1]+"{:0>2}".format(number))
            else: output.append(f)
        return output
    
    def build(self, dataset, filters):
        '''
        - dataset is a string denoting which features we want to use from the respective dataset
        - filters is a list of substring that should exist in the feature of the corresponding dataset
        '''
        selected_features = []
        filters = self._unpack_filters(filters)        
        filters = self._normalize_digits(filters)

        for f in self.available_features[dataset]:
            f = self._normalize_digits(f)[0]
            for item in filters:
                item = item.split("&")
                select = all([x in f for x in item])  
                # If any of the filters hit, then feature is included
                if select == True: 
                    for i in range(10): f = f.replace("_0{}".format(i), "_{}".format(i))
                    selected_features.append(f)
                    break
        return selected_features
    
    
    
def build_features(args,max_show = 10):
    # Build the features 
    features = args["db_main_default"].split(",")
    builder = FeatureBuilder()   
    for db in ["main","worldpop","awa","contra","national_holiday","lagged","hierarchical"]:
        filters =  args["db_{}_filter".format(db)].split(",")
        if "".join(filters) == "": continue
        features = features + builder.build(db,filters)
    print("Using {} features".format(len(features)))

    # show some of the features
    for i, f in enumerate(features):
        if len(features) <= max_show or (i < max_show // 2 or i >= len(features) - max_show // 2):
            print("[{}] {}".format(i, f))
        elif len(features) > max_show and i == max_show // 2:
            print("...")
    return features