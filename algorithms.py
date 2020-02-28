from statsmodels.tsa.ar_model import AR
from random import random
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def auto_reg(data):
    model = AR(data)
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.predict(len(data), len(data) + 12)
    print(yhat)

def moving_average(data):
    model = ARMA(data, order=(0, 1))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(data), len(data))
    print(yhat)

def arma_method(data):
    model = ARMA(data, order=(2, 1))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(data), len(data))
    print(yhat)

def arima(data):
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(data), len(data), typ='levels')
    print(yhat)

def sarima(data):
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(data), len(data))
    print(yhat)



