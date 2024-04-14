import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

def build_model(features, RUL, x):
    
    if (x=='LGB'):
        model = lgb.LGBMRegressor(random_state=42)
        model.fit(features, RUL)
    if (x=='XGB'):
        model = xgb.XGBRegressor()
        model.fit(features, RUL)
    if (x=='LReg'):
        model = LinearRegression()
        model.fit(features, RUL)

    return model

def make_prediction(model, features):
    
    prediction = model.predict(features)

    return prediction

def evaluations(prediction, RUL):

    mse = mean_squared_error(RUL.values.reshape(-1), prediction)
    print("Der MSE auf die Trainingsdaten lautet :", mse)
    r2 = r2_score(RUL.values.reshape(-1), prediction)
    print('R2 score: ', r2)
    rmse = sqrt(mse)
    print('RMSE: ', rmse)
