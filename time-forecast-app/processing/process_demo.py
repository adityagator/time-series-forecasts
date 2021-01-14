from django.core.files import File
from data.models import InputData, OutputData, UserHistory
from processing.constants import Constants
from processing.file_operations import FileOperations
from processing.algorithms import Algorithms
import os
from forecast_project import settings
import sys
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from processing.holt_winters import HoltWintersClass
from processing.cluster import Cluster
import warnings
import logging
import traceback
import boto3
from datetime import datetime
import numpy as np
warnings.filterwarnings("ignore")

def process_demo(dict_data):
    count = 0
    output_dict = {}
    output_obj = OutputData()
    top5_dict = {}
    for key, value in dict_data.items():
        min_rmse = sys.maxsize
        min_mape = sys.maxsize
        min_algo = ""
        min_pred = []
        test_num = int(len(value)/5)
        # algo_obj = Algorithms(value,
        #                     value[0:-Constants.TESTING_MONTHS],
        #                     value[-Constants.TESTING_MONTHS:])
        algo_obj = Algorithms(value,
                            value[0:-test_num],
                            value[-test_num:])
        min_params = []
        
        # Similar Algorithms: found in statsmodels
        for algo_name in Constants.SIMILAR_ALGORITHMS:
            try:
                rmse, mape, pred = algo_obj.get_predictions_rmse_mape(algo_name)
                
                if(rmse < min_rmse):
                    min_rmse = rmse
                    min_algo = algo_name
                    min_mape = mape
                    min_pred = pred
            except Exception as err:
                logging.error("Error while using %s on %s", algo_name, key)
                logging.error(traceback.format_exc())


    # Croston Algorithm
        # try:
        #     rmse_cros, mape_cros, pred_cros = algo_obj.croston_calculate()
        #     # algo_obj.rmse2[Constants.CROSTON] = rmse_cros
        #     # algo_obj.pred[Constants.CROSTON] = pred_cros
        #     algo_obj.rmse_pred[Constants.CROSTON] = [rmse_cros, [pred_cros]]
        #     if(rmse_cros < min_rmse):
        #         min_rmse = rmse_cros
        #         min_algo = "croston"
        #         min_mape = mape_cros
        #         min_pred = pred_cros
        # except Exception as err:
        #     logging.error("Error while using %s on %s", algo_name, key)
        #     logging.error(traceback.format_exc())

    

        # FNN
        try:
            rmse_fnn, mape_fnn, pred_fnn = algo_obj.fnn_calculate()
            if (rmse_fnn < min_rmse):
                min_rmse = rmse_fnn
                min_algo = "FNN"
                min_mape = mape_fnn
                min_pred = pred_fnn
        except Exception as err:
            logging.error("Error while using %s on %s", algo_name, key)
            logging.error(traceback.format_exc())

            # Holt-Winters method
        try:
            # print('optimised HWES Method :')
            data = value
            # data = ads.Ads[:-20]  # leave some data for testing

            # initializing model parameters alpha, beta and gamma
            x = [0.1, 0.1, 0.1]

            # log_error = mean_squared_log_error([756, 360, 324, 1656], [1262.3403637583606, 1330.2384660692694, 606.1044120473597, 529.9378740380566])

            # Minimizing the loss function
            opt = minimize(algo_obj.holt_winters_function, x0=x,
                        args=(data, mean_squared_error),
                        method="TNC", bounds=((0, 1), (0, 1), (0, 1))
                        )

            # Take optimal values...
            alpha_final, beta_final, gamma_final = opt.x
            # print('final values: ', alpha_final, beta_final, gamma_final)

            # ...and train the model with them, forecasting for the next 50 hours
            model = HoltWintersClass(
                data,
                slen=12,
                alpha=alpha_final,
                beta=beta_final,
                gamma=gamma_final,
                n_preds=12,
                scaling_factor=2)
            model.triple_exponential_smoothing()
            predictions = model.result[-12:]
            for i in range(0, len(predictions)):
                predictions[i] = round(predictions[i], 2)
                if predictions[i] < 0:
                    predictions[i] = 0
            rmse_hwes = algo_obj.rmse(predictions)
            if (rmse_hwes < min_rmse):
                min_rmse = round(rmse_hwes, 2)
                min_algo = "HWES"
                min_mape = round(algo_obj.mean_absolute_percentage_error(predictions), 2)
                min_pred = predictions
                min_params = [alpha_final, beta_final, gamma_final]
            # algo_obj.rmse2[Constants.HWES] = rmse
            # algo_obj.pred[Constants.HWES] = predictions
            algo_obj.rmse_pred[Constants.HWES] = [rmse, predictions]
        except Exception as err:
            logging.error("Error while using %s on %s", algo_name, key)
            logging.error(traceback.format_exc())

        predicted_output = algo_obj.getPredictedValues(min_algo, min_params)
        # round the predicted values
        for i in range(0, len(predicted_output)):
            predicted_output[i] = round(predicted_output[i])
        if isinstance(min_pred,(np.ndarray)):
            min_pred = min_pred.tolist()
        if isinstance(predicted_output,(np.ndarray)):
            predicted_output = predicted_output.tolist()
        output_dict[key] = [min_algo, min_rmse, min_mape, predicted_output, min_pred]
        count += 1
        top5 = algo_obj.rankTopAlgorithms(algo_obj.rmse_pred)
        top5_dict[key] = top5
    
    output_obj.output_dict = output_dict
    output_obj.top5_dict = top5_dict
    return output_obj   
