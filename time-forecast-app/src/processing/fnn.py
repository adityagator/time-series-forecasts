from math import sqrt
from numpy import array
from numpy import mean
from numpy import std
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy
# from processing.algorithms import Algorithms


class FeedForwardNeuralNetwork:
    # split a univariate dataset into train/test sets

    def train_test_split(data, n_test):
        return data[:-n_test], data[-n_test:]

    # transform list into supervised learning format
    def series_to_supervised(data, n_in=1, n_out=1):
        df = DataFrame(data)
        cols = list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
        # put it all together
        agg = concat(cols, axis=1)
        # drop rows with NaN values
        agg.dropna(inplace=True)
        return agg.values

    # root mean squared error or rmse
    def measure_rmse(actual, predicted):
        return sqrt(mean_squared_error(actual, predicted))

    # fit a model
    def model_fit(train, config):
        # unpack config
        n_input, n_nodes, n_epochs, n_batch = config
        # prepare data
        data = FeedForwardNeuralNetwork.series_to_supervised(
            train, n_in=n_input)
        # train_x,train_y=data[:,:-1],data[:,-1]
        train_x, train_y = data[:, :-1], data[:, -1]
        model = Sequential()
        model.add(Dense(n_nodes, activation='relu', input_dim=n_input))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        # fit
        model.fit(
            train_x,
            train_y,
            epochs=n_epochs,
            batch_size=n_batch,
            verbose=0)
        return model

    # forecast with a pre-fit model
    def model_predict(model, history, config):
        # unpack config
        n_input, _, _, _ = config
        # prepare data
        x_input = array(history[-n_input:]).reshape(1, n_input)
        # forecast
        yhat = model.predict(x_input, verbose=0)
        return yhat[0]

    # walk-forward validation for univariate data
    def walk_forward_validation(data, n_test, cfg):
        predictions = list()
        # split dataset
        train, test = FeedForwardNeuralNetwork.train_test_split(data, n_test)
        # fit model
        model = FeedForwardNeuralNetwork.model_fit(train, cfg)
        # seed history with training dataset
        history = [x for x in train]
        # step over each time-step in the test set
        for i in range(len(test)):
            # fit model and make forecast for history
            yhat = FeedForwardNeuralNetwork.model_predict(model, history, cfg)
            # store forecast in list of predictions
            predictions.append(numpy.round(yhat, 2))
            # add actual observation to history for the next loop
            history.append(test[i])
        # estimate prediction error
        error = FeedForwardNeuralNetwork.measure_rmse(test, predictions)
        print(' > %.3f' % error)
        return error, predictions

    # repeat evaluation of a config
    '''
    def repeat_evaluate(data, config, n_test, n_repeats=30):
    	# fit and evaluate the model n times
    	scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
    	return scores
    '''

    # summarize model performance
    def summarize_scores(name, scores):
        # print a summary
        scores_m, score_std = mean(scores), std(scores)
        print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))

    def mean_absolute_percentage_error(y_true, y_pred):

        return numpy.mean(numpy.abs((y_true - y_pred) / y_true)) * 100

    def fnn(data, no_of_test_values):

        data = data
        # data split
        n_test = no_of_test_values
        # define config = n_input, n_nodes, n_epochs, n_batch
        config = [10, 10, 25, 1]
        # grid search
        '''
        scores = repeat_evaluate(data, config, n_test)
        # summarize scores
        summarize_scores('mlp', scores)'''
        print('Feed Forward Neural Net')
        rmse, prediction_list = FeedForwardNeuralNetwork.walk_forward_validation(
            data, n_test, config)
        l = numpy.array(prediction_list).tolist()
        flat_list = [item for sublist in l for item in sublist]
        flat_list = [float(numpy.round(x)) for x in flat_list]
        #print('Feed Forward Neural Net')
        #print('Actual Values: ', data[-n_test:])
        #print('Predicted values: ',flat_list)
        #mape_ffn = mean_absolute_percentage_error(data[-n_test:], flat_list)
        #print('MAPE: %.3f' %mape_ffn)
        #print('RMSE: %.3f' %rmse)
        for i in range(0, len(flat_list)):
            flat_list[i] = round(flat_list[i], 2)
            if flat_list[i] < 0:
                flat_list[i] = 0
        return flat_list

    def fnn_next_year(total_data):
        train_data = total_data
        config = [10, 10, 25, 1]
        print("Next Year Pred using FNN")

        predictions = list()
        test = [None] * 12
        model = FeedForwardNeuralNetwork.model_fit(train_data, config)
        # seed history with training dataset
        history = [x for x in train_data]
        history = numpy.array(history)
        # step over each time-step in the test set
        for i in range(len(test)):
            # fit model and make forecast for history
            yhat = FeedForwardNeuralNetwork.model_predict(
                model, history, config)
            # store forecast in list of predictions
            predictions.append(numpy.round(yhat, 2))
            # add actual observation to history for the next loop
            history = numpy.append(history, numpy.round(yhat, 2))
            # history.append(numpy.round(yhat, 2))

        l = numpy.array(predictions).tolist()
        flat_list = [item for sublist in l for item in sublist]

        for i in range(0, len(flat_list)):
            flat_list[i] = round(flat_list[i], 2)
            if flat_list[i] < 0:
                flat_list[i] = 0

        return flat_list
        # l = numpy.array(predictions)
