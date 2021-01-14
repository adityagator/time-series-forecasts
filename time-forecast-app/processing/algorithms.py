from processing.constants import Constants
import numpy
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.holtwinters import (ExponentialSmoothing,
                                         SimpleExpSmoothing)
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import np
import math
from sklearn.model_selection import TimeSeriesSplit
from processing.holt_winters import HoltWintersClass
from processing.croston import Croston
import warnings
from statsmodels.tsa.statespace.varmax import VARMAX
import pandas as pd
warnings.filterwarnings("ignore")
from processing.fnn import FeedForwardNeuralNetwork
from processing.lstm import lstm

class Algorithms:
    # constructor: initializing training and testing data
    def __init__(self, total, data, test):
        self.total = total
        self.data = data
        self.test = test
        self.rmse_pred = {}
        # print("TOTAL DATA")
        # print(len(self.total))
        # print(self.total[0])
        # print("TRAIN DATA")
        # print(len(self.data))
        # print(self.data[0])
        # print("TEST DATA")
        # print(len(self.test))
        # print(self.test[0])

    # rank algorithms
    def rankTopAlgorithms(self, unranked_dict):
        rmse_list =[]
        pred_list=[]
        min_rmse_list=[]
        min_key_index_list=[]
        min_key_list=[]
        min5Algorithms=[]
        min5RMSE = [] 
        min5Dict = {}

        if len(unranked_dict) == 0:
            unranked_dict["NOT ENOUGH DATA"] = [0, [0] * len(self.total + Constants.NUMBER_OF_PREDICTIONS)]
            return unranked_dict
        elif len(unranked_dict) < 6:
            return unranked_dict
        for values in unranked_dict.values():
            rmse_list.append(values[0])
            pred_list.append(values[1])
        
        while(len(rmse_list) != 0):
            
            min_rmse = min(rmse_list)
            # print("UNRANKED DICT")
            # print(unranked_dict)

            
            for key,values in unranked_dict.items():
                if(values[0]==min_rmse):
                    if isinstance(values[1], numpy.ndarray):
                        values[1] = values[1].tolist()
                    min5Dict[key] = values[1]
                    min5Algorithms.append(key)
                    min5RMSE.append(min_rmse)
                    rmse_list.remove(min_rmse)
           
        #print("Min 5 algorithms: ", min5Algorithms)
        
        # for alg in min5Algorithms:
        #     min5Dict[alg] = unranked_dict[alg]
        while len(min5Dict) > 5:
            min5Dict.popitem()
        return min5Dict
    
    # create a differenced series for ARIMA, AR etc
    def difference(self, dataset, interval=12):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return numpy.array(diff)
    
    # invert differences to get back original values
    def inverse_difference(self, history, yhat, interval=12):
        return yhat + history[-interval]

    # return mape
    def mean_absolute_percentage_error(self, y_pred):
        y_true, y_pred = numpy.array(self.test), numpy.array(y_pred)
        ans_arr = numpy.array([])
        for i in range(0, len(y_true)):
            if y_true[i] > 0:
                ans = abs((y_true[i] - y_pred[i]) / y_true[i]) * 100
                # ans_arr.append(ans)
                ans_arr = numpy.append(ans_arr, ans)
        if ans_arr.size == 0:
            return 0
        return numpy.mean(ans_arr) 
        # return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # return rmse
    def rmse(self, predicted):
        return round(math.sqrt(mean_squared_error(self.test, predicted)),2)

    # return rmse and mape
    def rmse_mape(self, predicted):
        # print("true: ", self.test, " pred: ", predicted)
        return round(math.sqrt(mean_squared_error(self.test, predicted)),2), round(self.mean_absolute_percentage_error(
                                                                                                        predicted),2)


    def get_predictions_rmse_mape(self, min_algo):
        if min_algo == Constants.ARIMA:
            model = ARIMA(self.data, order=(7, 0, 1))
            model_fit = model.fit()
        elif min_algo == Constants.MOVING_AVERAGE:
            model = ARMA(self.data, order=[0,1])
            model_fit = model.fit(disp=0)
        elif min_algo == Constants.AR:
            model = AR(self.data)
            model_fit = model.fit(disp=0)
        elif min_algo == Constants.ARMA:
            # model = ARMA(self.data, order=[1,0])
            model = ARMA(self.data, order=[2,1])
            model_fit = model.fit(disp=0)
        elif min_algo == Constants.SARIMA:
            model = SARIMAX(self.data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            model_fit = model.fit(disp=0)
        elif min_algo == Constants.SES:
            model = SimpleExpSmoothing(self.data)
            model_fit = model.fit()
        
        start_index = len(self.data)
        # end_index = start_index + 11
        end_index = start_index + len(self.test) - 1
        forecast = model_fit.predict(start=start_index, end=end_index)
        rmse, mape = self.rmse_mape(forecast)
        for i in range(0, len(forecast)):
            forecast[i] = round(forecast[i])
            if forecast[i] < 0:
                forecast[i] = 0
        # self.rmse2[min_algo] = rmse
        # self.pred[min_algo] = forecast
        self.rmse_pred[min_algo] = [rmse, forecast]
        return rmse, mape, forecast
    
    def get_predictions_rmse_mape_final(self, min_algo):
        if min_algo == Constants.ARIMA:
            model = ARIMA(self.total, order=(7, 0, 1))
            model_fit = model.fit()
        elif min_algo == Constants.MOVING_AVERAGE:
            model = ARMA(self.total, order=[0,1])
            model_fit = model.fit(disp=0)
        elif min_algo == Constants.AR:
            model = AR(self.total)
            model_fit = model.fit(disp=0)
        elif min_algo == Constants.ARMA:
            # model = ARMA(self.total, order=[1,0])
            model = ARMA(self.total, order=[2,1])
            model_fit = model.fit(disp=0)
        elif min_algo == Constants.SARIMA:
            model = SARIMAX(self.total, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            model_fit = model.fit(disp=0)
        elif min_algo == Constants.SES:
            model = SimpleExpSmoothing(self.total)
            model_fit = model.fit()
        
        start_index = len(self.total)
        # end_index = start_index + 11
        end_index = start_index + Constants.NUMBER_OF_PREDICTIONS - 1
        forecast = model_fit.predict(start=start_index, end=end_index)
        for i in range(0, len(forecast)):
            forecast[i] = round(forecast[i])
            if forecast[i] < 0:
                forecast[i] = 0
        return forecast
    
    # def rnn_calculate(self):
    #     yhat = lstm.rnn(self.total, len(self.test))
    #     rmse, mape = self.rmse_mape(yhat)
    #     return round(rmse,2), round(mape,2), yhat

    # def rnn_final(self):
    #     yhat = lstm.rnn_next_year(self.total)
    #     return yhat
    
    def fnn_calculate(self):
        yhat = FeedForwardNeuralNetwork.fnn(self.total, len(self.test))
        rmse, mape = self.rmse_mape(yhat)
        # self.rmse2[min_algo] = rmse
        # self.pred[min_algo] = yhat
        self.rmse_pred[Constants.FNN] = [rmse, yhat]
        return round(rmse,2), round(mape,2), yhat
    
    def fnn_final(self):
        yhat = FeedForwardNeuralNetwork.fnn_next_year(self.total)
        return yhat

    def holt_winters_function(self, params=[0.1, 0.1, 0.1], series=[], loss_function=mean_squared_error, slen=12):
        series = self.total
        errors = []

        # print('alpha beta gamma: ', params)

        values = self.total
        alpha, beta, gamma = params

        # set the number of folds for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        # print('tscv: ', tscv.split(values))
        # iterating over folds, train model on each, forecast and calculate error

        new_model = HoltWintersClass(
            series=self.data, slen=slen,
            alpha=alpha, beta=beta, gamma=gamma, n_preds=12)
        new_model.triple_exponential_smoothing()

        predictions = new_model.result[-12:]
        actual = self.test
        mape = self.mean_absolute_percentage_error(predictions)
        error = loss_function(predictions, actual)
        errors.append(error)
        rmse_hwes = self.rmse(predictions)
        
        return rmse_hwes

    def hwes_final(self, params):
        alpha, beta, gamma = params
        new_model = HoltWintersClass(
        series=self.total, slen=12,
        alpha=alpha, beta=beta, gamma=gamma, n_preds=12)
        new_model.triple_exponential_smoothing()

        predictions = new_model.result[-12:]
        for i in range(0, len(predictions)):
            predictions[i] = round(predictions[i], 2)
            if predictions[i] < 0:
                predictions[i] = 0
            
        return predictions
    
    def croston_calculate(self):
        # print("in croston calc")
        input_data = self.data
        predictions = []
        for i in range(0,len(self.test)):
	        yhat = Croston.croston_method(input_data)
	        input_data.append(yhat)
	        predictions.append(yhat)
        for i in range(0, len(predictions)):
            predictions[i] = round(predictions[i], 2)
            if predictions[i] < 0:
                predictions[i] = 0
        rmse, mape = self.rmse_mape(predictions)
        # print(predictions)
        return rmse, mape, predictions

    def croston_final(self):
        # print("in croston final")
        predictions = []
        input_data = self.total
        for i in range(0,12):
	        yhat = Croston.croston_method(input_data)
	        input_data.append(yhat)
	        predictions.append(yhat)
        for i in range(0, len(predictions)):
            predictions[i] = round(predictions[i], 2)
            if predictions[i] < 0:
                predictions[i] = 0
        # print(predictions)
        return predictions

    def varma_calculate(self):
        predictions = []
        input_data = numpy.array(self.data)
        input_data = numpy.log(input_data)
        input_data = self.difference(input_data)
        input_data = pd.DataFrame(input_data)
        input_data = input_data.dropna()
        for i in range(0,len(self.test)):
            model = VARMAX(input_data, order=(1,1))
            model_fit = model.fit(disp=False)
            yhat = model_fit.forecast()
            predictions.append(yhat)
            input_data.append(yhat)
        for i in range(0, len(predictions)):
            predictions[i] = round(predictions[i], 2)
            if predictions[i] < 0:
                predictions[i] = 0
        rmse, mape = self.rmse_mape(predictions)
        return rmse, mape, predictions
    
    def varma_final(self):
        predictions = []
        input_data = numpy.array(self.total)
        input_data = numpy.log(input_data)
        input_data = self.difference(input_data)
        input_data = pd.DataFrame(input_data)
        input_data = input_data.dropna()
        for i in range(0,len(self.test)):
            model = VARMAX(input_data, order=(1,1))
            model_fit = model.fit(disp=False)
            yhat = model_fit.forecast()
            predictions.append(yhat)
            input_data.append(yhat)
        for i in range(0, len(predictions)):
            predictions[i] = round(predictions[i], 2)
            if predictions[i] < 0:
                predictions[i] = 0
       
        return predictions
    
    def get_top_five(self):
        {k: v for k, v in sorted(self.predictions_rmse.items(), key=lambda item: item[1])}
        


    # return the predicted output for the least rmse algorithm given by min_algo
    def getPredictedValues(self, min_algo, params):
        if min_algo in Constants.SIMILAR_ALGORITHMS:
            return self.get_predictions_rmse_mape_final(min_algo)
        elif min_algo == Constants.HWES:
            return self.hwes_final(params)
        elif min_algo == Constants.CROSTON:
            return self.croston_final()
        elif min_algo == Constants.VARMA:
            return self.varma_final()
        elif min_algo == Constants.FNN:
            return FeedForwardNeuralNetwork.fnn_next_year(self.total)
        elif min_algo == Constants.RNN:
            return lstm.rnn_next_year(self.total)
        else:
            return [0,0,0,0,0,0,0,0,0,0,0,0]
