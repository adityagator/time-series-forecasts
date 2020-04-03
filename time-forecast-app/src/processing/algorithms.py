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
import warnings
warnings.filterwarnings("ignore")
# from processing.fnn import FeedForwardNeuralNetwork

class Algorithms:
    # constructor: initializing training and testing data
    def __init__(self, total, data, test):
        self.total = total
        self.data = data
        self.test = test
    
    # create a differenced series for ARIMA, AR etc
    def difference(self, dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return numpy.array(diff)
    
    # invert differences to get back original values
    def inverse_difference(self, history, yhat, interval=1):
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
    # return rmse and mape for ARIMA method
    # def arima_calculate(self):
    #     differenced = self.difference(self.data, 12)
    #     model = ARIMA(differenced, order=(7, 0, 1))
    #     model_fit = model.fit()
    #     start_index = len(differenced)
    #     end_index = start_index + 11
    #     forecast = model_fit.predict(start=start_index, end=end_index)
    #     history = [x for x in self.data]
    #     day = 1
    #     pred = []
    #     for yhat in forecast:
    #         inverted = self.inverse_difference(history, yhat, 12)
    #         pred.append(round(inverted, 2))
    #         # print('Day %d: %f' % (day, inverted))
    #         history.append(round(inverted, 2))
    #         day += 1
    #     for i in range(0, len(pred)):
    #         if(pred[i] < 0):
    #             pred[i] = 0
    #     return self.rmse_mape(pred)

    def arima_calculate(self):
        model = ARIMA(self.data, order=(7, 0, 1))
        model_fit = model.fit()
        start_index = len(self.data)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.data]
        # day = 1
        pred = []
        for yhat in forecast:
            pred.append(round(yhat,2))
            # print('Day %d: %f' % (day, yhat))
            history.append(round(yhat,2))
            # day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        # raise Exception("testing")
        return self.rmse_mape(pred)
    
    # return predictions for ARIMA if least rmse amongst others
    # def arima_final(self):
    #     differenced = self.difference(self.total, 12)
    #     model = ARIMA(differenced, order=(7, 0, 1))
    #     model_fit = model.fit()
    #     start_index = len(differenced)
    #     end_index = start_index + 11
    #     forecast = model_fit.predict(start=start_index, end=end_index)
    #     history = [x for x in self.total]
    #     day = 1
    #     pred = []
    #     for yhat in forecast:
    #         inverted = self.inverse_difference(history, yhat, 12)
    #         pred.append(round(inverted, 2))
    #         # print('Day %d: %f' % (day, inverted))
    #         history.append(round(inverted, 2))
    #         day += 1
    #     for i in range(0, len(pred)):
    #         if(pred[i] < 0):
    #             pred[i] = 0
    #     return pred
    
    def arima_final(self):
        model = ARIMA(self.total, order=(7, 0, 1))
        model_fit = model.fit()
        start_index = len(self.total)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.total]
        day = 1
        pred = []
        for yhat in forecast:
            pred.append(round(yhat,2))
            # print('Day %d: %f' % (day, yhat))
            history.append(round(yhat,2))
            day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        return pred


    # return rmse and mape for SARIMA method
    def sarima_calculate(self):
        model = SARIMAX(self.data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=0)
        start_index = len(self.data)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.data]
        day = 1
        pred = []
        for yhat in forecast:
            pred.append(round(yhat,2))
            # print('Day %d: %f' % (day, yhat))
            history.append(round(yhat,2))
            day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        return self.rmse_mape(pred)

    # return predictions for SARIMA if least rmse amongst others
    def sarima_final(self):
        model = SARIMAX(self.total, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=0)
        start_index = len(self.total)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.total]
        day = 1
        pred = []
        for yhat in forecast:
            pred.append(round(yhat,2))
            # print('Day %d: %f' % (day, yhat))
            history.append(round(yhat,2))
            day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        return pred

    def ar_calculate(self):
        model = AR(self.data)
        model_fit = model.fit(disp=0)
        start_index = len(self.data)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.data]
        day = 1
        pred = []
        for yhat in forecast:
            pred.append(round(yhat,2))
            # print('Day %d: %f' % (day, yhat))
            history.append(round(yhat,2))
            day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        return self.rmse_mape(pred)
    
    # def ar_calculate(self):
    #     differenced = self.difference(self.data, 12)
    #     model = AR(differenced)
    #     model_fit = model.fit(disp=0)
    #     start_index = len(differenced)
    #     end_index = start_index + 11
    #     forecast = model_fit.predict(start=start_index, end=end_index)
    #     history = [x for x in self.data]
    #     day = 1
    #     pred = []
    #     for yhat in forecast:
    #         inverted = self.inverse_difference(history, yhat, 12)
    #         pred.append(round(inverted, 2))
    #         # print('Day %d: %f' % (day, round(inverted, 2)))
    #         history.append(round(inverted, 2))
    #         day += 1
    #     for i in range(0, len(pred)):
    #         if pred[i] < 0:
    #             pred[i] = 0
    #     return self.rmse_mape(pred)

    def ar_final(self):
        model = AR(self.total)
        model_fit = model.fit(disp=0)
        start_index = len(self.total)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.total]
        day = 1
        pred = []
        for yhat in forecast:
            pred.append(round(yhat,2))
            # print('Day %d: %f' % (day, yhat))
            history.append(round(yhat,2))
            day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        return pred
    
    # def ar_final(self):
    #     differenced = self.difference(self.total, 12)
    #     model = AR(differenced)
    #     model_fit = model.fit(disp=0)
    #     start_index = len(differenced)
    #     end_index = start_index + 11
    #     forecast = model_fit.predict(start=start_index, end=end_index)
    #     history = [x for x in self.total]
    #     day = 1
    #     pred = []
    #     for yhat in forecast:
    #         inverted = self.inverse_difference(history, yhat, 12)
    #         pred.append(round(inverted, 2))
    #         # print('Day %d: %f' % (day, round(inverted, 2)))
    #         history.append(round(inverted, 2))
    #         day += 1
    #     for i in range(0, len(pred)):
    #         if (pred[i] < 0):
    #             pred[i] = 0
    #     return pred
    
    def arma_calculate(self):
        model = ARMA(self.data, order=[1,0])
        model_fit = model.fit(disp=0)
        start_index = len(self.data)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.data]
        day = 1
        pred = []
        for yhat in forecast:
            pred.append(round(yhat, 2))
            # print('Day %d: %f' % (day, round(yhat,2)))
            history.append(round(yhat, 2))
            day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        return self.rmse_mape(pred)
    
    def arma_final(self):
        model = ARMA(self.total, order=[1,0])
        model_fit = model.fit(disp=0)
        start_index = len(self.total)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.total]
        day = 1
        pred = []
        for yhat in forecast:
            pred.append(round(yhat,2))
            # print('Day %d: %f' % (day, round(yhat,2)))
            history.append(round(yhat,2))
            day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        return pred

    # Moving Average
    def ma_calculate(self):
        model = ARMA(self.data, order=[0,1])
        model_fit = model.fit(disp=0)
        start_index = len(self.data)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.data]
        day = 1
        pred = []
        for yhat in forecast:
            pred.append(round(yhat, 2))
            # print('Day %d: %f' % (day, round(yhat,2)))
            history.append(round(yhat, 2))
            day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        return self.rmse_mape(pred)
    
    def ma_final(self):
        model = ARMA(self.total, order=[0,1])
        model_fit = model.fit(disp=0)
        start_index = len(self.total)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.total]
        day = 1
        pred = []
        for yhat in forecast:
            pred.append(round(yhat,2))
            # print('Day %d: %f' % (day, round(yhat,2)))
            history.append(round(yhat,2))
            day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        return pred
    
    def ses_calculate(self):
        model = SimpleExpSmoothing(self.data)
        model_fit = model.fit()
        start_index = len(self.data)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.data]
        day = 1
        pred = []
        for yhat in forecast:
            # inverted = self.inverse_difference(history, yhat, 12)
            pred.append(round(yhat,2))
            # print('Day %d: %f' % (day, round(yhat,2)))
            history.append(round(yhat,2))
            day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        return self.rmse_mape(pred)

    def ses_final(self):
        model = SimpleExpSmoothing(self.total)
        model_fit = model.fit()
        start_index = len(self.total)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.data]
        day = 1
        pred = []
        for yhat in forecast:
            pred.append(round(yhat,2))
            # print('Day %d: %f' % (day, round(yhat,2)))
            history.append(round(yhat,2))
            day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        return pred
    
    # def weighted_average(self, series, weights):
    #     result = 0.0
    #     weights.reverse()
    #     for n in range(len(weights)):
    #         result += series.iloc[-n-1] * weights[n]
    #     return result

    # def moving_average(self, num_preds):
    #     train = self.data
    #     test = []
    #     for i in range(0, num_preds):
    #         model = ARMA(train, order=(0, 1))
    #         model_fit = model.fit(disp=False)
    #         yhat = float(model_fit.predict(len(train), len(train)))
    #         test.append(round(yhat,2))
    #         train.append(round(yhat,2))
    #     return test

    # def moving_average_calculate(self):
    #     predicted = self.moving_average(Constants.TESTING_MONTHS)
    #     rmse, mape = self.rmse_mape(predicted)
    #     print('Predicted values for last 4 months : ', predicted)
    #     return round(rmse,2), round(mape,2)
    
    def rnn_calculate(self, value):
        yhat, month_rnn = lstm.lstm.rnn(value, Constants.TESTING_MONTHS)
        # print('Recurrent Neural Network (LSTM):')
        # print('Actual values: ', self.test)
        # print('Predicted values: ', yhat)
        rmse, mape = self.rmse_mape(yhat)
        #print('RMSE: %.3f' % rmse)
        return round(rmse,2), round(mape,2), month_rnn
    
    # def fnn_calculate(self, value):
    #     yhat = FeedForwardNeuralNetwork.fnn(value, Constants.TESTING_MONTHS)
    #     print('Feed Forward Neural Network:')
    #     print('Actual values: ', self.test)
    #     print('Predicted values: ', yhat)
    #     rmse, mape = self.rmse_mape(yhat)
    #     #print('RMSE: %.3f' % rmse)
    #     return round(rmse,2), round(mape,2)

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

        return predictions

    # return the predicted output for the least rmse algorithm given by min_algo
    def getPredictedValues(self, min_algo, params):
        if min_algo == "ARIMA":
            return self.arima_final()
        elif min_algo == "MOVING AVERAGE":
            return self.ma_final()
        elif min_algo == "AR":
            return self.ar_final()
        elif min_algo == "ARMA":
            return self.arma_final()
        elif min_algo == "SARIMA":
            return self.sarima_final()
        elif min_algo == "SES":
            return self.ses_final()
        elif min_algo == "RNN":
            return []
        elif min_algo == "HWES":
            return self.hwes_final(params)
        # elif min_algo == "FNN":
        #     return FeedForwardNeuralNetwork.FeedForwardNeuralNetwork.fnn_next_year(self.total)
        else:
            return [0,0,0,0,0,0,0,0,0,0,0,0]