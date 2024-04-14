from Programs.FinalProject.features import extract_test_features, standardize_test
from Programs.FinalProject.plots import sensor_measurements_plot
import matplotlib.pyplot as plt
from loading import load_train_data, load_test_data, load_targets
from plots import cycle_frequency, sensor_measurements_plot, sensor_scatters, correlation_matrix, plot_prediction
from RUL import calculate_RUL
from features import extract_features, drop_sensors, drop_op, standardize_data
from model import build_model, make_prediction, evaluations


#loading data used for training
train_data = load_train_data()
#introduce few samples and shape
print(train_data.head(10))
print(train_data.shape)

#max number of cycles for each turbofan - histogram 
cycle_frequency(train_data)

#PLOT of the sensors for turbofan number 95; we can clearly see a trend of changed
#values from a certain point in time
#change unit number to see measurements for other units
unit_nr = 95
sensor_measurements_plot(train_data,unit_nr) 

#calculate RUL
train_data = calculate_RUL(train_data) 

#plots of sensor values which give us a good overview about a predictive
#power of taken measurements
sensor_scatters(train_data) 

#from scatter plot we can see that sensor measurements 1,5,6,10,16,18 and 19
#have no predictive power, therefore we drop those sensors
irrelevant_sensors = [1,5,6,10,16,18,19]
for_extraction = drop_sensors(train_data, irrelevant_sensors)

#print out remaining columns in dataset prepared for
#feature extraction
for col in for_extraction.columns:
    print(col)

#for further processing we take a look at the correlation matrix
#which gives us a good overview of feature/target dependencies
correlation_matrix(for_extraction)

#from the correlation matrix we see that operational settings have
#less to no correlation with RUL so we drop those features
irrelevant_op = [1,2,3]
for_extraction = drop_op(for_extraction, irrelevant_op)
print('Processed data: ', for_extraction.head())

#save processed data that is going to be used for feature extraction
for_extraction.to_csv('processed_data.csv', index=False)


#extract features from processed data set to later use for ML
if __name__ == '__main__':
    features, RUL = extract_features(for_extraction)

RUL.reset_index(drop=True, inplace=True)

#extracted features standardize/normalize
features = standardize_data(features)

#save extracted features and extracted RUL
features.to_csv('extracted_and_normalized_data.csv', index=False)
RUL.to_csv('target_data_processed.csv', index=False)

#we build our model using lgbmregressor
model = build_model(features, RUL, 'LGB')
#make predictions on training data
prediction = make_prediction(model, features)
#plot the predictions against real RUL
plot_prediction(prediction, RUL)
#evaluation of our model on training data
evaluations(prediction, RUL)

#load test and target data
test_data = load_test_data()
target_data = load_targets()

#extract features from the last window in test data and standardize data
features_test = extract_test_features(test_data, features, for_extraction)
features_test = standardize_test(features_test)

#make predictions on test data processed
pred_test = make_prediction(model, features_test)
#evaluations on test data
evaluations(pred_test, target_data)


#try with xgbregressor
model = build_model(features, RUL, 'XGB')
prediction_xgb = make_prediction(model, features)
evaluations(prediction_xgb, RUL)
pred_test_xgb = make_prediction(model, features_test)
evaluations(pred_test_xgb, target_data)

#try with linear regressor
model = build_model(features, RUL, 'LReg')
prediction_lin = make_prediction(model, features)
evaluations(prediction_lin, RUL)
pred_test_lin = make_prediction(model, features_test)
evaluations(pred_test_lin, target_data)