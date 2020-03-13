from pandas import np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import HoltWintersClass


class Algorithms:

    def __init__(self, data):
        self.data = data

    def mean_absolute_percentage_error(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def weighted_average(self, series, weights):
        result = 0.0
        weights.reverse()
        for n in range(len(weights)):
            result += series.iloc[-n-1] * weights[n]
        return result

    def arima(self):
        model = ARIMA(self.data, order=(1, 1, 0))
        model_fit = model.fit(disp=False)
        # make prediction
        yhat = model_fit.predict(len(self.data), len(self.data), typ='levels')
        return float(yhat)

    def moving_average(self):
        model = ARMA(self.data, order=(0, 1))
        model_fit = model.fit(disp=False)
        # make prediction
        yhat = model_fit.predict(len(self.data), len(self.data))
        return yhat

    def auto_reg(self):
        model = AR(self.data)
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.predict(len(self.data), len(self.data))
        return yhat

    def arma_method(self):
        model = ARMA(self.data, order=(1, 0))
        model_fit = model.fit(disp=False)
        # make prediction
        yhat = model_fit.predict(len(self.data), len(self.data))
        return yhat

    def sarima(self):
        model = SARIMAX(self.data, order=(1, 1, 1), seasonal_order=(2, 2, 2, 2))
        model_fit = model.fit(disp=False)
        # make prediction
        yhat = model_fit.predict(len(self.data), len(self.data))
        return yhat

    def ses(self):
        model = SimpleExpSmoothing(self.data)
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.predict(len(self.data), len(self.data))
        return yhat

    def hwes(self):
        model = ExponentialSmoothing(self.data)
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.predict(len(self.data), len(self.data))
        return yhat

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def mean_absolute_percentage_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

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

