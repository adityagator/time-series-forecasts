import csv
import pandas as pd
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

from statsmodels.tsa.ar_model import AR
from random import random
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots

from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization

import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs


from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error


def formatRawData(raw_file):
    
    with open(raw_file,mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data=list(csv_reader)
    for i in data:
        i[0 : 2] = ['-'.join(i[0 : 2])]
    dict_data = {i[0]: i[1:] for i in data}

    for key in dict_data:
        nums = dict_data.get(key)
        for i in range(0,len(nums)):
            nums[i] = int(nums[i])

    return(dict_data)


def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def weighted_average(series, weights):
    """
        Calculate weighter average on series
    """
    result = 0.0
    weights.reverse()
    for n in range(len(weights)):
        result += series.iloc[-n-1] * weights[n]
    return result

def arima(data):
    model = ARIMA(data, order=(1, 1, 0))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(data), len(data), typ='levels')
    return(float(yhat))

def moving_average(data):
    model = ARMA(data, order=(0, 1))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(data), len(data))
    return(yhat)

def auto_reg(data):
    model = AR(data)
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.predict(len(data), len(data))
    return(yhat)

def arma_method(data):
    model = ARMA(data, order=(1, 0))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(data), len(data))
    return(yhat)

def sarima(data):
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(2, 2, 2, 2))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(data), len(data))
    return(yhat)

def ses(data):
    model = SimpleExpSmoothing(data)
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.predict(len(data), len(data))
    return(yhat)

def hwes(data):

    model = ExponentialSmoothing(data)
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.predict(len(data), len(data))
    return yhat


data = 'input.csv'
data = formatRawData(data)

for key,value in data.items():
    train = value[0:20]
    test = value[20:]
    predicted=[]
    print()
    print('key: ',key)
    print('values: ',value)
    print()
    
    #ARIMA Algorithm
    train_arima = train
    test_arima = test
    predicted_arima=[]
    for i in range(0,len(test_arima)):
        yhat_arima = float(arima(train_arima))
        train_arima.append(float(yhat_arima))
        predicted_arima.append(float(yhat_arima))
    
    mse_arima = mean_squared_error(test_arima, predicted_arima)
    
    print('ARIMA:')
    print('Predicted values for last 4 months : ', predicted_arima)
    print('Mean Squared Error: ', mse_arima)
    print()
    
    
    #Moving Average
    train_ma = train
    test_ma = test
    predicted_ma=[]
    for i in range(0,len(test_ma)):
        yhat_ma = float(moving_average(train_ma))
        train_ma.append(float(yhat_ma))
        predicted_ma.append(float(yhat_ma))
    
    mse_ma = mean_squared_error(test_ma, predicted_ma)
    
    print('Moving Average:')
    print('Predicted values for last 4 months : ', predicted_ma)
    print('Mean Squared Error: ', mse_ma)
    print()
    
    #Auto Reg
    train_autoreg = train
    test_autoreg = test
    predicted_autoreg=[]
    for i in range(0,len(test_autoreg)):
        yhat_autoreg = float(auto_reg(train_autoreg))
        train_autoreg.append(float(yhat_autoreg))
        predicted_autoreg.append(float(yhat_autoreg))
    
    mse_autoreg = mean_squared_error(test_autoreg, predicted_autoreg)
    
    print('Auto Regression:')
    print('Predicted values for last 4 months : ', predicted_autoreg)
    print('Mean Squared Error: ', mse_autoreg)
    print()
    
     #ARMA Method
    train_arma = train
    test_arma = test
    predicted_arma=[]
    for i in range(0,len(test_arma)):
        yhat_arma = float(arma_method(train_arma))
        train_arma.append(float(yhat_arma))
        predicted_arma.append(float(yhat_arma))
    
    mse_arma = mean_squared_error(test_arma, predicted_arma)
    
    print('ARMA Method :')
    print('Predicted values for last 4 months : ', predicted_arma)
    print('Mean Squared Error: ', mse_arma)
    print()
    
     #SARIMA Method
    train_sarima = train
    test_sarima = test
    predicted_sarima=[]
    for i in range(0,len(test_sarima)):
        yhat_sarima = float(sarima(train_sarima))
        train_sarima.append(float(yhat_sarima))
        predicted_sarima.append(float(yhat_sarima))
    
    mse_sarima = mean_squared_error(test_sarima, predicted_sarima)
    
    print('SARIMA Method :')
    print('Predicted values for last 4 months : ', predicted_sarima)
    print('Mean Squared Error: ', mse_sarima)
    print()

    #SES Method
    train_ses = train
    test_ses = test
    predicted_ses=[]
    for i in range(0,len(test_ses)):
        yhat_ses = float(sarima(train_ses))
        train_ses.append(float(yhat_ses))
        predicted_ses.append(float(yhat_ses))
    
    mse_ses = mean_squared_error(test_ses, predicted_ses)
    
    print('SARIMA Method :')
    print('Predicted values for last 4 months : ', predicted_ses)
    print('Mean Squared Error: ', mse_ses)
    print()

    #HWES Method
    train_hwes = train
    test_hwes = test
    predicted_hwes=[]
    for i in range(0,len(test_hwes)):
        yhat_hwes = float(sarima(train_hwes))
        train_hwes.append(float(yhat_hwes))
        predicted_hwes.append(float(yhat_hwes))
    
    mse_hwes = mean_squared_error(test_hwes, predicted_hwes)
    
    print('SARIMA Method :')
    print('Predicted values for last 4 months : ', predicted_hwes)
    print('Mean Squared Error: ', mse_hwes)
    print()
    
    print('____________________________')


  
    
    
    
    
    
    
















