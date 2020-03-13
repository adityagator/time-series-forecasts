from pandas import np
# from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import HoltWintersClass
import Constants
import math
from scipy import linalg
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error

class Algorithms:

    def __init__(self, data, test):
        self.data = data
        self.test = test
        self.constants = Constants.Constants()

    # def mean_absolute_percentage_error(self, y_true, y_pred):
    #     y_true, y_pred = check_arrays(y_true, y_pred)
    #     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def mean_absolute_percentage_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def rmse_mape(self, predicted):
        print("true: ", self.test, " pred: ", predicted)
        return math.sqrt(mean_squared_error(self.test, predicted)), self.mean_absolute_percentage_error(self.test,
                                                                                                        predicted)
# np.mean(np.abs((self.test - predicted) / self.test)) * 100

    def weighted_average(self, series, weights):
        result = 0.0
        weights.reverse()
        for n in range(len(weights)):
            result += series.iloc[-n-1] * weights[n]
        return result

    def arima(self, num_preds):
        train = self.data
        test = []
        for i in range(0, num_preds):
            model = ARIMA(train, order=(1, 1, 0))
            model_fit = model.fit(disp=False)
            # make prediction
            yhat = float(model_fit.predict(len(train), len(train), typ='levels'))
            print(yhat)
            test.append(yhat)
            train.append(yhat)
        return test

    # def arima(self, num_preds):
    #     model = ARIMA(self.data, order=(1, 1, 0))
    #     model_fit = model.fit(disp=False)
    #     # make prediction
    #     yhat = model_fit.predict(len(self.data), len(self.data) + num_preds, typ='levels')
    #     return yhat

    def arima_calculate(self):
        # for i in range()
        predicted = self.arima(self.constants.TESTING_MONTHS)
        rmse, mape = self.rmse_mape(predicted)
        print('Predicted values for last 4 months : ', predicted)
        return rmse, mape

    # def moving_average(self, num_preds):
    #     model = ARMA(self.data, order=(0, 1))
    #     model_fit = model.fit(disp=False)
    #     # make prediction
    #     yhat = model_fit.predict(len(self.data), len(self.data) + num_preds)
    #     return yhat
    def moving_average(self, num_preds):
        train = self.data
        test = []
        for i in range(0, num_preds):
            model = ARMA(train, order=(0, 1))
            model_fit = model.fit(disp=False)
            # make prediction
            yhat = float(model_fit.predict(len(train), len(train)))
            # print(yhat)
            test.append(yhat)
            train.append(yhat)
        return test

    # def auto_reg(self, num_preds):
    #     model = AR(self.data)
    #     model_fit = model.fit()
    #     # make prediction
    #     yhat = model_fit.predict(len(self.data), len(self.data) + num_preds)
    #     return yhat
    def auto_reg(self, num_preds):
        train = self.data
        test = []
        for i in range(0, num_preds):
            model = AR(train)
            model_fit = model.fit()
            # make prediction
            yhat = float(model_fit.predict(len(train), len(train)))
            # print(yhat)
            test.append(yhat)
            train.append(yhat)
        return test

    # def arma_method(self, num_preds):
    #     model = ARMA(self.data, order=(1, 0))
    #     model_fit = model.fit(disp=False)
    #     # make prediction
    #     yhat = model_fit.predict(len(self.data), len(self.data) + num_preds)
    #     return yhat

    def arma_method(self, num_preds):
        train = self.data
        test = []
        for i in range(0, num_preds):
            model = ARMA(train, order=[1, 0])
            model_fit = model.fit(disp=False)
            # make prediction
            yhat = float(model_fit.predict(len(train), len(train)))
            # print(yhat)
            test.append(yhat)
            train.append(yhat)
        return test

    # def sarima(self, num_preds):
    #     model = SARIMAX(self.data, order=(1, 1, 1), seasonal_order=(2, 2, 2, 2))
    #     model_fit = model.fit(disp=False)
    #     # make prediction
    #     yhat = model_fit.predict(len(self.data), len(self.data) + num_preds)
    #     return yhat
    def sarima(self, num_preds):
        train = self.data
        test = []
        for i in range(0, num_preds):
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(2, 2, 2, 2), initialization='approximate_diffuse')
            model_fit = model.fit(disp=False)
            # make prediction
            yhat = float(model_fit.predict(len(train), len(train)))
            # print(yhat)
            test.append(yhat)
            train.append(yhat)
        return test


    # def ses(self, num_preds):
    #     model = SimpleExpSmoothing(self.data)
    #     model_fit = model.fit()
    #     # make prediction
    #     yhat = model_fit.predict(len(self.data), len(self.data) + num_preds)
    #     return yhat

    def ses(self, num_preds):
        train = self.data
        test = []
        for i in range(0, num_preds):
            model = SimpleExpSmoothing(train)
            model_fit = model.fit()
            # make prediction
            yhat = float(model_fit.predict(len(train), len(train)))
            # print(yhat)
            test.append(yhat)
            train.append(yhat)
        return test

    # def hwes(self, num_preds):
    #     model = ExponentialSmoothing(self.data)
    #     model_fit = model.fit()
    #     # make prediction
    #     yhat = model_fit.predict(len(self.data), len(self.data) + num_preds)
    #     return yhat
    def hwes(self, num_preds):
        train = self.data
        test = []
        for i in range(0, num_preds):
            model = ExponentialSmoothing(train)
            model_fit = model.fit()
            # make prediction
            yhat = float(model_fit.predict(len(train), len(train)))
            # print(yhat)
            test.append(yhat)
            train.append(yhat)
        return test

    def holt_winters_function(self, params=[0.1, 0.1, 0.1],
                              series=[828, 324, 648, 720, 468, 612, 828, 1008, 1296, 1368, 648, 576, 648, 936, 720, 144,
                                      1008, 360, 432, 1080,
                                      756, 360, 324, 1656], loss_function=mean_squared_error, slen=12):
        errors = []

        print('alpha beta gamma: ', params)

        values = [828, 324, 648, 720, 468, 612, 828, 1008, 1296, 1368, 648, 576, 648, 936, 720, 144, 1008, 360, 432,
                  1080,
                  756, 360, 324, 1656]
        alpha, beta, gamma = params

        # set the number of folds for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        # print('tscv: ', tscv.split(values))
        # iterating over folds, train model on each, forecast and calculate error

        new_model = HoltWintersClass.HoltWintersClass(
            series=[828, 324, 648, 720, 468, 612, 828, 1008, 1296, 1368, 648, 576,
                    648, 936, 720, 144, 1008, 360, 432, 1080], slen=slen,
            alpha=alpha, beta=beta, gamma=gamma, n_preds=4)
        new_model.triple_exponential_smoothing()

        predictions = new_model.result[-4:]
        print('predictions')
        print(predictions)
        actual = [756, 360, 324, 1656]
        print('actual')
        print(actual)
        mape = self.mean_absolute_percentage_error(actual, predictions)
        error = loss_function(predictions, actual)
        errors.append(error)

        print("rmse optimized hwes: ", np.mean(np.array(errors)))

        # for train, test in tscv.split(values):
        # print('training data')
        # print(train)
        # print('test')
        # print(test)
        # new_model = HoltWintersClass.HoltWintersClass(series=[828, 324, 648, 720, 468, 612, 828, 1008, 1296, 1368, 648, 576,
        #                                                       648, 936, 720, 144, 1008, 360, 432, 1080], slen=slen,
        #                                               alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
        # new_model.triple_exponential_smoothing()
        #
        # predictions = new_model.result[-len(test):]
        # print('predictions')
        # print(predictions)
        # actual = [756, 360, 324, 1656]
        # print('actual')
        # print(actual)
        # error = loss_function(predictions, actual)
        # errors.append(error)
        #
        # print("rmse optimized hwes: ", np.mean(np.array(errors)))

        return np.mean(np.array(errors))



    def moving_average_calculate(self):
        predicted = self.moving_average(self.constants.TESTING_MONTHS)
        rmse, mape = self.rmse_mape(predicted)
        print('Predicted values for last 4 months : ', predicted)
        return rmse, mape

    def auto_reg_calculate(self):
        predicted = self.auto_reg(self.constants.TESTING_MONTHS)
        rmse, mape = self.rmse_mape(predicted)
        print('Predicted values for last 4 months : ', predicted)
        return rmse, mape

    def arma_calculate(self):
        predicted = self.arma_method(self.constants.TESTING_MONTHS)
        rmse, mape = self.rmse_mape(predicted)
        print('Predicted values for last 4 months : ', predicted)
        return rmse, mape

    def sarima_calculate(self):
        predicted = self.sarima(self.constants.TESTING_MONTHS)
        rmse, mape = self.rmse_mape(predicted)
        print('Predicted values for last 4 months SARIMA: ', predicted)
        return rmse, mape

    def ses_calculate(self):
        predicted = self.ses(self.constants.TESTING_MONTHS)
        rmse, mape = self.rmse_mape(predicted)
        print('Predicted values for last 4 months SES: ', predicted)
        return rmse, mape

    def hwes_calculate(self):
        predicted = self.hwes(self.constants.TESTING_MONTHS)
        rmse, mape = self.rmse_mape(predicted)
        print('Predicted values for last 4 months : HWES', predicted)
        return rmse, mape

    def getPredictedValues(self, min_algo):
        print(min_algo)
        predicted = {
            "ARIMA": self.arima(self.constants.NUMBER_OF_PREDICTIONS),
            "MOVING AVERAGE": self.moving_average(self.constants.NUMBER_OF_PREDICTIONS),
            "AR": self.auto_reg(self.constants.NUMBER_OF_PREDICTIONS),
            "ARMA": self.arma_method(self.constants.NUMBER_OF_PREDICTIONS),
            "SARIMA": self.sarima(self.constants.NUMBER_OF_PREDICTIONS),
            "SES": self.ses(self.constants.NUMBER_OF_PREDICTIONS),
            "HWES": self.hwes(self.constants.NUMBER_OF_PREDICTIONS)
        }
        return predicted.get(min_algo, "Failure")

