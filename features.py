from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute, make_forecasting_frame
from tsfresh import extract_features, select_features

import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def drop_sensors(train_data,irrelevant_sensors):

    drop_sensors = ['Sensor Measure'+str(i) for i in irrelevant_sensors]
    for_extraction = train_data

    for sensor in drop_sensors:
        for_extraction = for_extraction.drop([sensor], axis=1)

    return for_extraction

def drop_op(for_extraction,irrelevant_op):

    drop_op = ['OpSetting'+str(i) for i in irrelevant_op]
    
    for op in drop_op:
        for_extraction = for_extraction.drop([op], axis=1)

    return for_extraction

#When developing features, an attempt is made to improve the predictive performance of the learning algorithms 
#by creating features from raw data and to use these to simplify the learning process. 
#The new features are intended to provide additional information that cannot be clearly identified 
#in the original or existing feature groups and or is not easily visible. 
#The process is not trivial, and well-founded and productive decisions often require a certain amount of specialist knowledge.

#As shown in the following figure, we are using TSFRESH to automatically extract time series features from different domains.
#Sliding window method
def extract_features(for_extraction):

    #columns from which we want to extract features with tsfresh
    sens_cols = for_extraction.columns
    sens_cols = sens_cols.drop(['UnitNumber','RUL','Cycle'])    
    
    window_size = 5
    unit_id = for_extraction["UnitNumber"].unique()
    features = pd.DataFrame()
    RUL = pd.Series()

    for index in tqdm(unit_id):
    
        unit_data = for_extraction[for_extraction["UnitNumber"]==index]
        unit_data.reset_index(inplace=True,drop=True)
    
        df_extract, _ = make_forecasting_frame(unit_data["Sensor Measure2"], kind="x", 
                                             max_timeshift = window_size, 
                                             rolling_direction=1)
        del df_extract["kind"]
        df_extract["Sensor Measure2"] = df_extract["value"]
        del df_extract["value"]
        
        for column in sens_cols[1:]:
            temp, _ = make_forecasting_frame(unit_data[column], kind="x", 
                                             max_timeshift = window_size, 
                                             rolling_direction=1)
            df_extract[column] = temp["value"]

        features_extract = extract_features(df_extract, column_id='id', column_sort='time', default_fc_parameters=MinimalFCParameters(), impute_function=None, disable_progressbar=True, show_warnings=False, n_jobs=8)    
        features_extract = features_extract.iloc[window_size-1:]
        y = unit_data["RUL"].iloc[window_size-1:-1]
    
        features = pd.concat([features, features_extract])
        RUL = pd.concat([RUL, y])

    return features, RUL


def standardize_data(features):
    
    scaler = StandardScaler()
    features.values[:] = scaler.fit_transform(features)

    return features


#we take only extractions from last windows
def extract_test_features(test_data, features, for_extraction):

    #columns from which we want to extract features with tsfresh
    sens_cols = for_extraction.columns
    sens_cols = sens_cols.drop(['UnitNumber','RUL','Cycle']) 
    
    unit_id_test = test_data["UnitNumber"].unique()
    window_size = 5
    features_test = pd.DataFrame()
    feature_columns = features.columns

    for index in tqdm(unit_id_test):
        
        unit_data = test_data[test_data["UnitNumber"]==index] 
        unit_data = unit_data.iloc[-window_size:]
        unit_data = unit_data[sens_cols]
        unit_data["id"] = 1
        unit_data["time"] = range(unit_data.shape[0])
        unit_data.reset_index(inplace=True,drop=True)



        All_features = extract_features(unit_data, column_id="id", column_sort="time", 
                                    default_fc_parameters=MinimalFCParameters(),
                                    impute_function=None, disable_progressbar=True,
                                    show_warnings=False, n_jobs=8)

        All_features = All_features[feature_columns]
    

        features_test = pd.concat([features_test, All_features])

    return features_test

def standardize_test(features_test):

    scaler = StandardScaler()
    features_test.values[:] = scaler.transform(features_test)

    return features_test

