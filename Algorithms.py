from pandas import np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import HoltWintersClass
import Constants
import math
from scipy import linalg
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
import lstm
import FeedForwardNeuralNetwork
import numpy
month_rnn=[]
class Algorithms:

    def __init__(self, total, data, test):
        self.total = total
        self.data = data
        self.test = test
        self.constants = Constants.Constants()

    # def mean_absolute_percentage_error(self, y_true, y_pred):
    #     y_true, y_pred = check_arrays(y_true, y_pred)
    #     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # create a differenced series
    def difference(self, dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return numpy.array(diff)

    # invert differenced value
    def inverse_difference(self, history, yhat, interval=1):
        return yhat + history[-interval]

    def mean_absolute_percentage_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def rmse_mape(self, predicted):
        print("true: ", self.test, " pred: ", predicted)
        return round(math.sqrt(mean_squared_error(self.test, predicted)),2), round(self.mean_absolute_percentage_error(self.test,
                                                                                                        predicted),2)
# np.mean(np.abs((self.test - predicted) / self.test)) * 100
    #working

    #def ma_calculate(self):

    def arima_calculate(self):
        differenced = self.difference(self.data, 12)
        model = ARIMA(differenced, order=(7, 0, 1))
        model_fit = model.fit()
        start_index = len(differenced)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.data]
        day = 1
        pred = []
        for yhat in forecast:
            inverted = self.inverse_difference(history, yhat, 12)
            pred.append(round(inverted, 2))
            print('Day %d: %f' % (day, inverted))
            history.append(round(inverted, 2))
            day += 1
        for i in range(0, len(pred)):
            if(pred[i] < 0):
                pred[i] = 0
        return self.rmse_mape(pred)

    def arima_final(self):
        differenced = self.difference(self.total, 12)
        model = ARIMA(differenced, order=(7, 0, 1))
        model_fit = model.fit()
        start_index = len(differenced)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.total]
        day = 1
        pred = []
        for yhat in forecast:
            inverted = self.inverse_difference(history, yhat, 12)
            pred.append(round(inverted, 2))
            print('Day %d: %f' % (day, inverted))
            history.append(round(inverted, 2))
            day += 1
        for i in range(0, len(pred)):
            if(pred[i] < 0):
                pred[i] = 0
        return pred

    def sarima_calculate(self):
        # differenced = self.difference(self.data, 12)
        model = SARIMAX(self.data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=0)
        start_index = len(self.data)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.data]
        day = 1
        pred = []
        for yhat in forecast:
            # inverted = self.inverse_difference(history, yhat, 12)
            pred.append(round(yhat,2))
            print('Day %d: %f' % (day, yhat))
            history.append(round(yhat,2))
            day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        return self.rmse_mape(pred)

    def sarima_final(self):
        # differenced = self.difference(self.data, 12)
        model = SARIMAX(self.total, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=0)
        start_index = len(self.total)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.total]
        day = 1
        pred = []
        for yhat in forecast:
            # inverted = self.inverse_difference(history, yhat, 12)
            pred.append(round(yhat,2))
            print('Day %d: %f' % (day, yhat))
            history.append(round(yhat,2))
            day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        return pred

    def ar_calculate(self):
        differenced = self.difference(self.data, 12)
        model = AR(differenced)
        model_fit = model.fit(disp=0)
        start_index = len(differenced)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.data]
        day = 1
        pred = []
        for yhat in forecast:
            inverted = self.inverse_difference(history, yhat, 12)
            pred.append(round(inverted, 2))
            print('Day %d: %f' % (day, round(inverted, 2)))
            history.append(round(inverted, 2))
            day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        return self.rmse_mape(pred)

    def ar_final(self):
        differenced = self.difference(self.total, 12)
        model = AR(differenced)
        model_fit = model.fit(disp=0)
        start_index = len(differenced)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.total]
        day = 1
        pred = []
        for yhat in forecast:
            inverted = self.inverse_difference(history, yhat, 12)
            pred.append(round(inverted, 2))
            print('Day %d: %f' % (day, round(inverted, 2)))
            history.append(round(inverted, 2))
            day += 1
        for i in range(0, len(pred)):
            if (pred[i] < 0):
                pred[i] = 0
        return pred

    # def arma_calculate(self):
    #     differenced = self.difference(self.data, 12)
    #     model = ARMA(differenced, order=[1,0])
    #     model_fit = model.fit(disp=0)
    #     start_index = len(differenced)
    #     end_index = start_index + 11
    #     forecast = model_fit.predict(start=start_index, end=end_index)
    #     history = [x for x in self.data]
    #     day = 1
    #     pred = []
    #     for yhat in forecast:
    #         inverted = self.inverse_difference(history, yhat, 12)
    #         pred.append(inverted)
    #         print('Day %d: %f' % (day, inverted))
    #         history.append(inverted)
    #         day += 1
    #     for i in range(0, len(pred)):
    #         if pred[i] < 0:
    #             pred[i] = 0
    #     return self.rmse_mape(pred)

    def arma_calculate(self):
        # differenced = self.difference(self.data, 12)
        model = ARMA(self.data, order=[1,0])
        model_fit = model.fit(disp=0)
        start_index = len(self.data)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.data]
        day = 1
        pred = []
        for yhat in forecast:
            # inverted = self.inverse_difference(history, yhat, 12)
            pred.append(round(yhat, 2))
            print('Day %d: %f' % (day, round(yhat,2)))
            history.append(round(yhat, 2))
            day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        return self.rmse_mape(pred)

    def arma_final(self):
         # differenced = self.difference(self.data, 12)
        model = ARMA(self.total, order=[1,0])
        model_fit = model.fit(disp=0)
        start_index = len(self.total)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.total]
        day = 1
        pred = []
        for yhat in forecast:
            # inverted = self.inverse_difference(history, yhat, 12)
            pred.append(round(yhat,2))
            print('Day %d: %f' % (day, round(yhat,2)))
            history.append(round(yhat,2))
            day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        return pred

    def ses_calculate(self):
        # differenced = self.difference(self.data, 12)
        model = SimpleExpSmoothing(self.data)
        model_fit = model.fit()
        # model_fit = model.fit(smoothing_level=.5)
        start_index = len(self.data)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.data]
        day = 1
        pred = []
        for yhat in forecast:
            # inverted = self.inverse_difference(history, yhat, 12)
            pred.append(round(yhat,2))
            print('Day %d: %f' % (day, round(yhat,2)))
            history.append(round(yhat,2))
            day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        return self.rmse_mape(pred)

    def ses_final(self):
        # differenced = self.difference(self.data, 12)
        model = SimpleExpSmoothing(self.total)
        model_fit = model.fit()
        start_index = len(self.total)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.data]
        day = 1
        pred = []
        for yhat in forecast:
            # inverted = self.inverse_difference(history, yhat, 12)
            pred.append(round(yhat,2))
            print('Day %d: %f' % (day, round(yhat,2)))
            history.append(round(yhat,2))
            day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        return pred

    def hwes_calculate(self):
        # differenced = self.difference(self.data, 12)
        model = ExponentialSmoothing(self.data)
        # model_fit = model.fit()
        model_fit = model.fit(smoothing_level=.3, smoothing_slope=.05)
        start_index = len(self.data)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.data]
        day = 1
        pred = []
        for yhat in forecast:
            # inverted = self.inverse_difference(history, yhat, 12)
            pred.append(round(yhat,2))
            print('Day %d: %f' % (day, round(yhat,2)))
            history.append(round(yhat,2))
            day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        return self.rmse_mape(pred)

    def hwes_final(self):
        # differenced = self.difference(self.data, 12)
        model = ExponentialSmoothing(self.total)
        model_fit = model.fit()
        start_index = len(self.total)
        end_index = start_index + 11
        forecast = model_fit.predict(start=start_index, end=end_index)
        history = [x for x in self.data]
        day = 1
        pred = []
        for yhat in forecast:
            # inverted = self.inverse_difference(history, yhat, 12)
            pred.append(yhat)
            print('Day %d: %f' % (day, yhat))
            history.append(round(yhat,2))
            day += 1
        for i in range(0, len(pred)):
            if pred[i] < 0:
                pred[i] = 0
        return pred


    def weighted_average(self, series, weights):
        result = 0.0
        weights.reverse()
        for n in range(len(weights)):
            result += series.iloc[-n-1] * weights[n]
        return result

    def moving_average(self, num_preds):
        train = self.data
        test = []
        for i in range(0, num_preds):
            model = ARMA(train, order=(0, 1))
            model_fit = model.fit(disp=False)
            # make prediction
            yhat = float(model_fit.predict(len(train), len(train)))
            # print(yhat)
            test.append(round(yhat,2))
            train.append(round(yhat,2))
        return test

    # def holt_winters_function(self, params=[0.1, 0.1, 0.1],
    #                           series=[828, 324, 648, 720, 468, 612, 828, 1008, 1296, 1368, 648, 576, 648, 936, 720, 144,
    #                                   1008, 360, 432, 1080,
    #                                   756, 360, 324, 1656], loss_function=mean_squared_error, slen=12):
    #     errors = []

    #     print('alpha beta gamma: ', params)

    #     values = [828, 324, 648, 720, 468, 612, 828, 1008, 1296, 1368, 648, 576, 648, 936, 720, 144, 1008, 360, 432,
    #               1080,
    #               756, 360, 324, 1656]
    #     alpha, beta, gamma = params

    #     # set the number of folds for cross-validation
    #     tscv = TimeSeriesSplit(n_splits=5)

    #     # print('tscv: ', tscv.split(values))
    #     # iterating over folds, train model on each, forecast and calculate error

    #     new_model = HoltWintersClass.HoltWintersClass(
    #         series=[828, 324, 648, 720, 468, 612, 828, 1008, 1296, 1368, 648, 576,
    #                 648, 936, 720, 144, 1008, 360, 432, 1080], slen=slen,
    #         alpha=alpha, beta=beta, gamma=gamma, n_preds=4)
    #     new_model.triple_exponential_smoothing()

    #     predictions = new_model.result[-4:]
    #     print('predictions')
    #     print(predictions)
    #     actual = [756, 360, 324, 1656]
    #     print('actual')
    #     print(actual)
    #     mape = self.mean_absolute_percentage_error(actual, predictions)
    #     error = loss_function(predictions, actual)
    #     errors.append(error)

    #     print("rmse optimized hwes: ", np.mean(np.array(errors)))

    #     # for train, test in tscv.split(values):
    #     # print('training data')
    #     # print(train)
    #     # print('test')
    #     # print(test)
    #     # new_model = HoltWintersClass.HoltWintersClass(series=[828, 324, 648, 720, 468, 612, 828, 1008, 1296, 1368, 648, 576,
    #     #                                                       648, 936, 720, 144, 1008, 360, 432, 1080], slen=slen,
    #     #                                               alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
    #     # new_model.triple_exponential_smoothing()
    #     #
    #     # predictions = new_model.result[-len(test):]
    #     # print('predictions')
    #     # print(predictions)
    #     # actual = [756, 360, 324, 1656]
    #     # print('actual')
    #     # print(actual)
    #     # error = loss_function(predictions, actual)
    #     # errors.append(error)
    #     #
    #     # print("rmse optimized hwes: ", np.mean(np.array(errors)))

    #     return np.mean(np.array(errors))



    def moving_average_calculate(self):
        predicted = self.moving_average(self.constants.TESTING_MONTHS)
        rmse, mape = self.rmse_mape(predicted)
        print('Predicted values for last 4 months : ', predicted)
        return round(rmse,2), round(mape,2)
    
    def rnn_calculate(self, value):
       # test_rnn = value[-constants.TESTING_MONTHS:]
            
        yhat, month_rnn = lstm.lstm.rnn(value, self.constants.TESTING_MONTHS)
        print('Recurrent Neural Network (LSTM):')
        print('Actual values: ', self.test)
        print('Predicted values: ', yhat)
        rmse, mape = self.rmse_mape(yhat)
        #print('RMSE: %.3f' % rmse)
        return round(rmse,2), round(mape,2), month_rnn

    def fnn_calculate(self, value):
       # test_rnn = value[-constants.TESTING_MONTHS:]
            
        yhat = FeedForwardNeuralNetwork.FeedForwardNeuralNetwork.fnn(value, self.constants.TESTING_MONTHS)
        print('Feed Forward Neural Network:')
        print('Actual values: ', self.test)
        print('Predicted values: ', yhat)
        rmse, mape = self.rmse_mape(yhat)
        #print('RMSE: %.3f' % rmse)
        return round(rmse,2), round(mape,2)

    # def rnn_final(self):
    #     yhat, month_rnn = lstm.lstm.rnn(self.data, 0)
    #     return yhat
    
    
    # def fnn_calculate(self, value):
    #     #test_size = value[-6:]

    #     yhat = FeedForwardNeuralNetwork.FeedForwardNeuralNetwork.fnn(value, constants.TESTING_MONTHS)
    #     print('Feed Forward Neural Network:')
    #     print('Actual values: ', self.test)
    #     print('Predicted values: ', yhat)
    #     rmse_fnn = math.sqrt(mean_squared_error(test_size, yhat))
    #     mape_fnn = algo_obj.mean_absolute_percentage_error(test_size, yhat)
    #     print('RMSE: %.3f' % rmse_fnn)
    #     print('MAPE: ', mape_fnn)
    #     return rmse_fnn, mape_fnn

    def getPredictedValues(self, min_algo, month_rnn):
        print(min_algo)
        predicted = {
            "ARIMA": self.arima_final(),
            "MOVING AVERAGE": self.moving_average(self.constants.NUMBER_OF_PREDICTIONS),
            "AR": self.ar_final(),
            "ARMA": self.arma_final(),
            "SARIMA": self.sarima_final(),
            "SES": self.ses_final(),
            "HWES": self.hwes_final(),
            "RNN": month_rnn,
            "FNN": FeedForwardNeuralNetwork.FeedForwardNeuralNetwork.fnn_next_year(self.total)
        }
        return predicted.get(min_algo, "Failure")

