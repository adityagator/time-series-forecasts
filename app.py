import csv
import warnings  # `do not disturbe` mode
import Algorithms
warnings.filterwarnings('ignore')
import math
import HoltWintersClass
from scipy.optimize import minimize              # for function minimization
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import lstm

def format_raw_data(raw_file):
    
    with open(raw_file,mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = list(csv_reader)
    for i in data:
        i[0 : 2] = ['-'.join(i[0 : 2])]
    dict_data = {i[0]: i[1:] for i in data}

    for key in dict_data:
        nums = dict_data.get(key)
        for i in range(0,len(nums)):
            nums[i] = int(nums[i])

    return dict_data

data = 'input/inputRealData.csv'
data = format_raw_data(data)

for key,value in data.items():
    train = value[0:20]
    test = value[20:]
    predicted=[]
    algo_obj = Algorithms.Algorithms(train)
    print()
    print('key: ',key)
    print('values: ',value)
    print()

    #ARIMA Algorithm
    train_arima = train
    test_arima = test
    predicted_arima=[]
    for i in range(0,len(test)):
        yhat_arima = float(algo_obj.arima())
        train_arima.append(float(yhat_arima))

        predicted_arima.append(float(yhat_arima))
    print("test_arima")
    print(test_arima)
    print("pred_arima")
    print(predicted_arima)
    rmse_arima = math.sqrt(mean_squared_error(test_arima, predicted_arima))
    mape_arima = algo_obj.mean_absolute_percentage_error(test_arima, predicted_arima)

    print('ARIMA:')
    print('Predicted values for last 4 months : ', predicted_arima)
    print('Mean Squared Error: %.3f' % rmse_arima)
    print('MAPE: %.3f' % mape_arima)
    print()


    #Moving Average
    train_ma = train
    test_ma = test
    predicted_ma=[]
    for i in range(0,len(test_ma)):
        yhat_ma = float(algo_obj.moving_average())
        train_ma.append(float(yhat_ma))
        predicted_ma.append(float(yhat_ma))

    rmse_ma = math.sqrt(mean_squared_error(test_ma, predicted_ma))

    print('Moving Average:')
    print('Predicted values for last 4 months : ', predicted_ma)
    print('Mean Squared Error: %.3f' % rmse_ma)
    print()

    #Auto Reg
    train_autoreg = train
    test_autoreg = test
    predicted_autoreg=[]
    for i in range(0,len(test_autoreg)):
        yhat_autoreg = float(algo_obj.auto_reg())
        train_autoreg.append(float(yhat_autoreg))
        predicted_autoreg.append(float(yhat_autoreg))

    rmse_autoreg = math.sqrt(mean_squared_error(test_autoreg, predicted_autoreg))

    print('Auto Regression:')
    print('Predicted values for last 4 months : ', predicted_autoreg)
    print('Mean Squared Error: %.3f' %rmse_autoreg)
    print()

     #ARMA Method
    train_arma = train
    test_arma = test
    predicted_arma=[]
    for i in range(0,len(test_arma)):
        yhat_arma = float(algo_obj.arma_method())
        train_arma.append(float(yhat_arma))
        predicted_arma.append(float(yhat_arma))

    rmse_arma = math.sqrt(mean_squared_error(test_arma, predicted_arma))

    print('ARMA Method :')
    print('Predicted values for last 4 months : ', predicted_arma)
    print('Mean Squared Error: %.3f' %rmse_arma)
    print()

     #SARIMA Method
    train_sarima = train
    test_sarima = test
    predicted_sarima=[]
    for i in range(0,len(test_sarima)):
        yhat_sarima = float(algo_obj.sarima())
        train_sarima.append(float(yhat_sarima))
        predicted_sarima.append(float(yhat_sarima))

    rmse_sarima = math.sqrt(mean_squared_error(test_sarima, predicted_sarima))
    mape_sarima = algo_obj.mean_absolute_percentage_error(test_sarima, predicted_sarima)
    print('SARIMA Method :')
    print('Predicted values for last 4 months : ', predicted_sarima)
    print('Mean Squared Error: %.3f' %rmse_sarima)
    print('MAPE:%.3f' %mape_sarima)
    print()

    #SES Method
    train_ses = train
    test_ses = test
    predicted_ses=[]
    for i in range(0,len(test_ses)):
        yhat_ses = float(algo_obj.ses())
        train_ses.append(float(yhat_ses))
        predicted_ses.append(float(yhat_ses))

    rmse_ses = math.sqrt(mean_squared_error(test_ses, predicted_ses))
    mape_ses = algo_obj.mean_absolute_percentage_error(test_ses, predicted_ses)
    print('SES Method :')
    print('Predicted values for last 4 months : ', predicted_ses)
    print('Mean Squared Error: %.3f' %rmse_ses)
    print('MAPE: %.3f' %mape_ses)
    print()

    #HWES Method
    train_hwes = train
    test_hwes = test
    predicted_hwes=[]
    for i in range(0,len(test_hwes)):
        yhat_hwes = float(algo_obj.hwes())
        train_hwes.append(float(yhat_hwes))
        predicted_hwes.append(float(yhat_hwes))

    rmse_hwes = math.sqrt(mean_squared_error(test_hwes, predicted_hwes))

    print('HWES Method :')
    print('Predicted values for last 4 months : ', predicted_hwes)
    print('Mean Squared Error: %.3f' %rmse_hwes)
    print()
    '''
    #Holt-Winters method
    #hard code input for testing
    print('optimised HWES Method :')
    data = [828, 324, 648, 720, 468, 612, 828, 1008, 1296, 1368, 648, 576, 648, 936, 720, 144, 1008, 360, 432, 1080,
            756, 360, 324, 1656]
    # data = ads.Ads[:-20]  # leave some data for testing

    # initializing model parameters alpha, beta and gamma
    x = [0.1, 0.1, 0.1]

    log_error = mean_squared_log_error([756, 360, 324, 1656], [1262.3403637583606, 1330.2384660692694, 606.1044120473597, 529.9378740380566])

    # Minimizing the loss function
    opt = minimize(algo_obj.holt_winters_function, x0=x,
                   args=(data, mean_squared_log_error),
                   method="TNC", bounds=((0, 1), (0, 1), (0, 1))
                   )

    # Take optimal values...
    alpha_final, beta_final, gamma_final = opt.x
    print('final values: ', alpha_final, beta_final, gamma_final)

    # ...and train the model with them, forecasting for the next 50 hours
    model = HoltWintersClass.HoltWintersClass(data, slen=12, alpha=alpha_final, beta=beta_final, gamma=gamma_final,
                                              n_preds=4, scaling_factor=2)
    model.triple_exponential_smoothing()

    print()

    print('____________________________')
    '''
    #Recurrent Neural Network - LSTM
    test_rnn = value[-6:]
    
    yhat = lstm.lstm.rnn(value, 6)
    print('Recurrent Neural Network (LSTM):')
    print('Actual values: ',test_rnn)
    print('Predictted values: ',yhat)  
    rmse_rnn = math.sqrt(mean_squared_error(test_rnn, yhat))
    print('RMSE: %.3f' %rmse_rnn)
    print('____________________________')

    


  
    
    
    
    
    
    
















