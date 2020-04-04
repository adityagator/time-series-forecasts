from django.core.files import File
from data.models import OutputData
from data.models import InputData
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
warnings.filterwarnings("ignore")

class Process():
    log_file = os.path.join(settings.MEDIA_ROOT ,"log/app.log")
    logging.basicConfig(filename=log_file, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    logging.info("Creating log file")
    def run(input):
        count = 0
        input_file = os.path.join(settings.MEDIA_ROOT, input.file.name)
        dict_data = FileOperations.read_file(file=input_file)
        output_dict = {}
        output_obj = OutputData()
        vol_cluster = {}
        int_cluster = {}
        if input.cluster:
            vol_cluster, int_cluster = Cluster.run(dict_data)
            print(vol_cluster)
            print(int_cluster)
        
        if input.forecast:
            for key, value in dict_data.items():
                min_rmse = sys.maxsize
                min_mape = sys.maxsize
                min_algo = ""
                algo_obj = Algorithms(value,
                                  value[0:-Constants.TESTING_MONTHS],
                                  value[-Constants.TESTING_MONTHS:])
                min_params = []
                ship_pt, prod_h, part_no = key.split("^")
            # ARIMA Algorithm
                try:
                    rmse_arima, mape_arima = algo_obj.arima_calculate()
                    if(rmse_arima < min_rmse):
                        min_rmse = rmse_arima
                        min_algo = "ARIMA"
                        min_mape = mape_arima
                except Exception as err:
                    logging.error("Error while using ARIMA method on ship_pt: %s, prod_hierarchy: %s, part_number: %s", ship_pt, prod_h, part_no)
                    logging.error(traceback.format_exc())
                
                # Moving Average
                try:
                    rmse_ma, mape_ma = algo_obj.ma_calculate()
                    if(rmse_ma < min_rmse):
                        min_rmse = rmse_ma
                        min_algo = "MOVING AVERAGE"
                        min_mape = mape_ma
                except Exception as err:
                    logging.error("Error while using Moving Average method on ship_pt: %s, prod_hierarchy: %s, part_number: %s", ship_pt, prod_h, part_no)
                    logging.error(traceback.format_exc())

                # Auto Reg
                try:
                    rmse_ar, mape_ar = algo_obj.ar_calculate()
                    if (rmse_ar < min_rmse):
                        min_rmse = rmse_ar
                        min_algo = "AR"
                        min_mape = mape_ar
                    # print("rmse is :", rmse_ar, " ", mape_ar, " ", "AR")
                except Exception as err:
                    logging.error("Error while using Auto Regression method on ship_pt: %s, prod_hierarchy: %s, part_number: %s", ship_pt, prod_h, part_no)
                    logging.error(traceback.format_exc())
                
                # ARMA
                try:
                    rmse_arma, mape_arma = algo_obj.arma_calculate()
                    if(rmse_arma < min_rmse):
                        min_rmse = rmse_arma
                        min_algo = "ARMA"
                        min_mape = mape_arma
                except Exception as err:
                    logging.error("Error while using ARMA method on ship_pt: %s, prod_hierarchy: %s, part_number: %s", ship_pt, prod_h, part_no)
                    logging.error(traceback.format_exc())

                # SARIMA
                try:
                    rmse_sarima, mape_sarima = algo_obj.sarima_calculate()
                    if (rmse_sarima < min_rmse):
                        min_rmse = rmse_sarima
                        min_algo = "SARIMA"
                        min_mape = mape_sarima
                except Exception as err:
                    logging.error("Error while using SARIMA method on ship_pt: %s, prod_hierarchy: %s, part_number: %s", ship_pt, prod_h, part_no)
                    logging.error(traceback.format_exc())

                # SES
                try:
                    rmse_ses, mape_ses = algo_obj.ses_calculate()
                    if (rmse_ses < min_rmse):
                        min_rmse = rmse_ses
                        min_algo = "SES"
                        min_mape = mape_ses
                except Exception as err:
                    logging.error("Error while using SES method on ship_pt: %s, prod_hierarchy: %s, part_number: %s", ship_pt, prod_h, part_no)
                    logging.error(traceback.format_exc())

                # FNN
                # if input.deepLearning:
                #     rmse, mape = algo_obj.fnn_calculate(value)
                #     if (rmse < min_rmse):
                #         min_rmse = rmse
                #         min_algo = "FNN"
                #         min_mape = mape
                #     print("rmse is :", rmse, " ", mape, " ", "FNN")

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
                    rmse_hwes = algo_obj.rmse(predictions)
                    if (rmse_hwes < min_rmse):
                        min_rmse = round(rmse_hwes, 2)
                        min_algo = "HWES"
                        min_mape = round(algo_obj.mean_absolute_percentage_error(predictions), 2)
                        min_params = [alpha_final, beta_final, gamma_final]
                except Exception as err:
                    logging.error("Error while using HWES method on ship_pt: %s, prod_hierarchy: %s, part_number: %s", ship_pt, prod_h, part_no)
                    logging.error(traceback.format_exc())
                
                predicted_output = algo_obj.getPredictedValues(min_algo, min_params)
                output_dict[key] = [min_algo, min_rmse, min_mape, predicted_output, min_params]
                output_file = File(
                open(
                    FileOperations.write_forecast_file(
                        output_dict, input, vol_cluster, int_cluster), encoding='utf-8-sig'))
                log_file = File(
                open(
                    os.path.join(settings.MEDIA_ROOT ,"log/app.log"), encoding='utf-8-sig'
                )
            )

        output_obj.input = input
        output_obj.output_file.save(
            'forecast.csv', output_file)
        
        output_obj.log_file.save(
            'app.log', log_file
        )

        # if os.path.exists(str(input_file)):
        #     os.remove(str(input_file))
        
        # if os.path.exists(os.path.abspath(output_file)):
        #     os.remove(os.path.abspath(output_file))
        
        # if os.path.exists(os.path.abspath(log_file)):
        #     os.remove(os.path.abspath(log_file))
        
        return True
        
