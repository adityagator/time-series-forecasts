import warnings  # `do not disturbe` mode
import Algorithms
import warnings  # `do not disturbe` mode

import Algorithms

warnings.filterwarnings('ignore')
import FileOperations
import Constants
import sys


# import LstmClass


class Main:
    constants = Constants.Constants()
    file = FileOperations.FileOperations()
    # lstm = LstmClass.LstmClass()
    dict_data = file.read_file()
    output_dict = {}
    # dict_rm

    for key, value in dict_data.items():
        min_rmse = sys.maxsize
        min_mape = sys.maxsize
        min_algo = ""
        algo_obj = Algorithms.Algorithms(value, value[0:constants.TRAINING_MONTHS], value[constants.TRAINING_MONTHS:])
        print()
        print('key: ', key)
        print('values: ', value)
        print()
           
        # ARIMA Algorithm
        rmse_arima, mape_arima = algo_obj.arima_calculate()
        if(rmse_arima < min_rmse):
            min_rmse = rmse_arima
            min_algo = "ARIMA"
            min_mape = mape_arima
        print("rmse is :", rmse_arima, " ", mape_arima, " ", "ARIMA")

        # Moving Average
        rmse_ma, mape_ma = algo_obj.moving_average_calculate()
        if(rmse_ma < min_rmse):
            min_rmse = rmse_ma
            min_algo = "MOVING AVERAGE"
            min_mape = mape_ma
        print("rmse is :", rmse_ma, " ", mape_ma, " ", "MA")

        # Auto Reg
        rmse_ar, mape_ar = algo_obj.ar_calculate()
        if (rmse_ar < min_rmse):
            min_rmse = rmse_ar
            min_algo = "AR"
            min_mape = mape_ar
        print("rmse is :", rmse_ar, " ", mape_ar, " ", "AR")

        # #ARMA Method
        rmse_arma, mape_arma = algo_obj.arma_calculate()
        if(rmse_arma < min_rmse):
            min_rmse = rmse_arma
            min_algo = "ARMA"
            min_mape = mape_arma
        print("rmse is :", rmse_arma, " ", mape_arma, " ", "ARMA")

        #  #SARIMA Method
        rmse_sarima, mape_sarima = algo_obj.sarima_calculate()
        if (rmse_sarima < min_rmse):
            min_rmse = rmse_sarima
            min_algo = "SARIMA"
            min_mape = mape_sarima
        print("rmse is :", rmse_sarima, " ", mape_sarima, " ", "SARIMA")

        # #SES Method
        rmse_ses, mape_ses = algo_obj.ses_calculate()
        if (rmse_ses < min_rmse):
            min_rmse = rmse_ses
            min_algo = "SES"
            min_mape = mape_ses
        print("rmse is :", rmse_ses, " ", mape_ses, " ", "SES")

        # #HWES Method
        rmse, mape = algo_obj.hwes_calculate()
        if (rmse < min_rmse):
            min_rmse = rmse
            min_algo = "HWES"
            min_mape = mape
        print("rmse is :", rmse, " ", mape, " ", "HWES")
        
        #RNN
        rmse, mape, pred_rnn = algo_obj.rnn_calculate(value)
        if (rmse < min_rmse):
            min_rmse = rmse
            min_algo = "RNN"
            min_mape = mape
        print("rmse is :", rmse, " ", mape, " ", "RNN")
        
        #FNN
        rmse, mape = algo_obj.fnn_calculate(value)
        if (rmse < min_rmse):
            min_rmse = rmse
            min_algo = "FNN"
            min_mape = mape
        print("rmse is :", rmse, " ", mape, " ", "FNN")
        
        # predicted_output = algo_obj.getPredictedValues(min_algo, pred)
        # print("")
        # print("")
        # print("final pred")
        # print(predicted_output)
        
        pred = []
        predicted_output = algo_obj.getPredictedValues(min_algo, pred_rnn)
        print("")
        print("")
        print("final algo: ", min_algo)
        print("final pred")
        print(predicted_output)
        
        
        output_dict[key] = [min_algo, min_rmse, min_mape, predicted_output]
        file.write_file(output_dict)

       

        # Feed forward neural network
        

        # #Holt-Winters method
        # #hard code input for testing
        # print('optimised HWES Method :')
        # data = [828, 324, 648, 720, 468, 612, 828, 1008, 1296, 1368, 648, 576, 648, 936, 720, 144, 1008, 360, 432, 1080,
        #         756, 360, 324, 1656]
        # # data = ads.Ads[:-20]  # leave some data for testing
        #
        # # initializing model parameters alpha, beta and gamma
        # x = [0.1, 0.1, 0.1]
        #
        # log_error = mean_squared_log_error([756, 360, 324, 1656], [1262.3403637583606, 1330.2384660692694, 606.1044120473597, 529.9378740380566])
        #
        # # Minimizing the loss function
        # opt = minimize(algo_obj.holt_winters_function, x0=x,
        #                args=(data, mean_squared_log_error),
        #                method="TNC", bounds=((0, 1), (0, 1), (0, 1))
        #                )
        #
        # # Take optimal values...
        # alpha_final, beta_final, gamma_final = opt.x
        # print('final values: ', alpha_final, beta_final, gamma_final)
        #
        # # ...and train the model with them, forecasting for the next 50 hours
        # model = HoltWintersClass.HoltWintersClass(data, slen=12, alpha=alpha_final, beta=beta_final, gamma=gamma_final,
        #                                           n_preds=4, scaling_factor=2)
        # model.triple_exponential_smoothing()
        #
        # print()
        #
        # print('____________________________')



  
    
    
    
    
    
    
















