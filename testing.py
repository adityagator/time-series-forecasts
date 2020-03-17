import csv
import math
import warnings  # `do not disturbe` mode
import Algorithms
import FeedForwardNeuralNetwork
import lstm

warnings.filterwarnings('ignore')
import HoltWintersClass
import FileOperations
import Constants
import sys
from scipy.optimize import minimize  # for function minimization
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error


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
        algo_obj = Algorithms.Algorithms(value[0:constants.TRAINING_MONTHS], value[constants.TRAINING_MONTHS:])
        print()
        print('key: ', key)
        print('values: ', value)
        print()

        # ARIMA Algorithm
        # rmse_arima, mape_arima = algo_obj.arima_calculate()
        # if (rmse_arima < min_rmse):
        #     min_rmse = rmse_arima
        #     min_algo = "ARIMA"
        #     min_mape = mape_arima
        # print("rmse is :", rmse_arima, " ", mape_arima, " ", "ARIMA")
        print("ARIMA")
        print(algo_obj.arima_calculate())
        #SARIMA
        print("SARIMA")
        print(algo_obj.sarima_calculate())
        #AR
        print("AR")
        print(algo_obj.ar_calculate())
        #ARMA
        print("ARMA")
        print(algo_obj.arma_calculate())
        #SES
        print("SES")
        print(algo_obj.ses_calculate())
        # HWES
        print("HWES")
        print(algo_obj.hwes_calculate())













