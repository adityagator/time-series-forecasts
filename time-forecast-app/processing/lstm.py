from pandas import Series
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
# from matplotlib import pyplot
import numpy
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

class lstm:
    tb._SYMBOLIC_SCOPE.value = True
    # date-time parsing function for loading the dataset
    def parser(x):
        return datetime.strptime('190'+x, '%Y-%m')

    # frame a sequence as a supervised learning problem
    def timeseries_to_supervised(data, lag=1):
        df = DataFrame(data)
        columns = [df.shift(i) for i in range(1, lag+1)]
        columns.append(df)
        df = concat(columns, axis=1)
        df.fillna(0, inplace=True)
        return df

    # create a differenced series
    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return Series(diff)

    # invert differenced value
    def inverse_difference(history, yhat, interval=1):
        return yhat + history[-interval]

    # scale train and test data to [-1, 1]
    def scale(train, test):
        # fit scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train)
        # transform train
        train = train.reshape(train.shape[0], train.shape[1])
        train_scaled = scaler.transform(train)
        # transform test
        test = test.reshape(test.shape[0], test.shape[1])
        test_scaled = scaler.transform(test)
        return scaler, train_scaled, test_scaled

    # inverse scaling for a forecasted value
    def invert_scale(scaler, X, value):
        new_row = [x for x in X] + [value]
        array = numpy.array(new_row)
        array = array.reshape(1, len(array))
        inverted = scaler.inverse_transform(array)
        return inverted[0, -1]

    # fit an LSTM network to training data
    def fit_lstm(train, batch_size, nb_epoch, neurons):
        X, y = train[:, 0:-1], train[:, -1]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        for i in range(nb_epoch):
            model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
            model.reset_states()
        return model

    # make a one-step forecast
    def forecast_lstm(model, batch_size, X):
        X = X.reshape(1, 1, len(X))
        yhat = model.predict(X, batch_size=batch_size)
        return yhat[0,0]

    def mean_absolute_percentage_error(y_true, y_pred):

        return numpy.mean(numpy.abs((y_true - y_pred) / y_true)) * 100

    def rnn(data, test_months):
        raw_values = data
        # load dataset
        '''
        series = read_csv('two-test.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)

        # transform data to be stationary
        raw_values = series.values
        '''
        diff_values = lstm.difference(raw_values, 1)

        # transform data to be supervised learning
        supervised = lstm.timeseries_to_supervised(diff_values, 1)
        supervised_values = supervised.values

        # split data into train and test-sets
        train, test = supervised_values[0:-test_months], supervised_values[-test_months:]

        # transform the scale of the data
        scaler, train_scaled, test_scaled = lstm.scale(train, test)

        # fit the model
        lstm_model = lstm.fit_lstm(train_scaled, 1, 100, 10)
        # forecast the entire training dataset to build up state for forecasting
        train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
        lstm_model.predict(train_reshaped, batch_size=1)

        # walk-forward validation on the test data
        predictions = list()
        for i in range(len(test_scaled)):
        # make one-step forecast
            X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
            yhat = lstm.forecast_lstm(lstm_model, 1, X)
            # invert scaling
            yhat = lstm.invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = lstm.inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
            # store forecast
            predictions.append(max(0,yhat))
            expected = raw_values[len(train) + i + 1]
            #print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
        one_year_predictions=[]
        predictions = [float(numpy.round(x)) for x in predictions]
        for i in range(0,12):
            X, y = supervised_values[i, 0:-1], supervised_values[i, -1]
            yhat = lstm.forecast_lstm(lstm_model, 1, X)
            # invert scaling
            yhat = lstm.invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = lstm.inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
            # store forecast
            one_year_predictions.append(max(0,yhat))

        one_year_predictions = [float(numpy.round(x)) for x in one_year_predictions]
        # report performance
        #rmse = sqrt(mean_squared_error(raw_values[-4:], predictions))
        #print('Test RMSE: %.3f' % rmse)

        # line plot of observed vs predicted
        #pyplot.plot(raw_values[-4:])
        #pyplot.plot(predictions)
        #pyplot.show()
        print("")
        print("")
        print("RNN pred: final:")
        print(predictions)
        print("")
        print("")
        return predictions


    
    def rnn_next_year(total_data):
        train_data = total_data
        diff_values = lstm.difference(train_data, 1)
        supervised = lstm.timeseries_to_supervised(diff_values, 1)
        supervised_values = supervised.values

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train_data)
        # transform train
        train_data = train_data.reshape(train_data.shape[0], train_data.shape[1])
        train_scaled = scaler.transform(train_data)
        lstm_model = lstm.fit_lstm(train_scaled, 1, 100, 10)
        predictions = [None] * 12
        # test = [None] * 12
        # model = lstm.model_fit(train_data, 1, 100, 10)
        # seed history with training dataset
        # history = [x for x in train_data]
        # history = numpy.array(history)
        # step over each time-step in the test set
        train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
        # lstm_model.predict(train_reshaped, batch_size=1)
        for i in range(len(predictions)):
            # fit model and make forecast for history
            yhat = lstm_model.predict(train_reshaped, batch_size=1)
            # store forecast in list of predictions
            predictions.append(numpy.round(yhat, 2))
            # add actual observation to history for the next loop
            # history = numpy.append(history, numpy.round(yhat, 2))
            # history.append(numpy.round(yhat, 2))

        l = numpy.array(predictions).tolist()
        flat_list = [item for sublist in l for item in sublist]

        for i in range(0, len(flat_list)):
            flat_list[i] = round(flat_list[i], 2)
            if flat_list[i] < 0:
                flat_list[i] = 0

        return flat_list