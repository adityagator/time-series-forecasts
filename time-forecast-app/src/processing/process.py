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
from processing.get_username import get_username
warnings.filterwarnings("ignore")

class Process():
    count = 0
    log_file = os.path.join(settings.MEDIA_ROOT ,"log/app.log")
    logging.basicConfig(filename=log_file, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    logging.info("Creating log file")

    def run(input):
        count = 0
        # AWS download
        # s3_client = boto3.client('s3')
        # print(input.file.name)
        # s3_client.download_file(settings.AWS_STORAGE_BUCKET_NAME, input.file.name, "files/"+input.file.name)

        input_file = os.path.join(settings.MEDIA_ROOT, input.file.name)
        dict_data = FileOperations.read_file(file=input_file)
        output_dict = {}
        output_obj = OutputData()
        user_history_obj = UserHistory()
        vol_cluster = {}
        int_cluster = {}
        if input.cluster:
            vol_cluster, int_cluster = Cluster.run(dict_data)

        if input.forecast:
            for key, value in dict_data.items():
                min_rmse = sys.maxsize
                min_mape = sys.maxsize
                min_algo = ""
                min_pred = []
                algo_obj = Algorithms(value,
                                  value[0:-Constants.TESTING_MONTHS],
                                  value[-Constants.TESTING_MONTHS:])
                min_params = []
                ship_pt, prod_h, part_no = key.split("^")

            # Croston Algorithm
                try:
                    rmse_cros, mape_cros, pred_cros = algo_obj.croston_calculate()
                    if(rmse_cros < min_rmse):
                        min_rmse = rmse_cros
                        min_algo = "croston"
                        min_mape = mape_cros
                        min_pred = pred_cros
                except Exception as err:
                    logging.error("Error while using Croston method on ship_pt: %s, prod_hierarchy: %s, part_number: %s", ship_pt, prod_h, part_no)
                    logging.error(traceback.format_exc())

                # # VARMA Algorithm
                # try:
                #     rmse_varma, mape_varma, pred_varma = algo_obj.varma_calculate()
                #     if(rmse_varma < min_rmse):
                #         min_rmse = rmse_varma
                #         min_algo = "VARMA"
                #         min_mape = mape_varma
                #         min_pred = pred_varma
                # except Exception as err:
                #     logging.error("Error while using VARMA method on ship_pt: %s, prod_hierarchy: %s, part_number: %s", ship_pt, prod_h, part_no)
                #     logging.error(traceback.format_exc())

            # ARIMA Algorithm
                try:
                    rmse_arima, mape_arima, pred_arima = algo_obj.arima_calculate()
                    if(rmse_arima < min_rmse):
                        min_rmse = rmse_arima
                        min_algo = "ARIMA"
                        min_mape = mape_arima
                        min_pred = pred_arima
                except Exception as err:
                    logging.error("Error while using ARIMA method on ship_pt: %s, prod_hierarchy: %s, part_number: %s", ship_pt, prod_h, part_no)
                    logging.error(traceback.format_exc())

                # Moving Average
                try:
                    rmse_ma, mape_ma, pred_ma = algo_obj.ma_calculate()
                    if(rmse_ma < min_rmse):
                        min_rmse = rmse_ma
                        min_algo = "MOVING AVERAGE"
                        min_mape = mape_ma
                        min_pred = pred_ma
                except Exception as err:
                    logging.error("Error while using Moving Average method on ship_pt: %s, prod_hierarchy: %s, part_number: %s", ship_pt, prod_h, part_no)
                    logging.error(traceback.format_exc())

                # Auto Reg
                try:
                    rmse_ar, mape_ar, pred_ar = algo_obj.ar_calculate()
                    if (rmse_ar < min_rmse):
                        min_rmse = rmse_ar
                        min_algo = "AR"
                        min_mape = mape_ar
                        min_pred = pred_ar
                    # print("rmse is :", rmse_ar, " ", mape_ar, " ", "AR")
                except Exception as err:
                    logging.error("Error while using Auto Regression method on ship_pt: %s, prod_hierarchy: %s, part_number: %s", ship_pt, prod_h, part_no)
                    logging.error(traceback.format_exc())

                # ARMA
                try:
                    rmse_arma, mape_arma, pred_arma = algo_obj.arma_calculate()
                    if(rmse_arma < min_rmse):
                        min_rmse = rmse_arma
                        min_algo = "ARMA"
                        min_mape = mape_arma
                        min_pred = pred_arma
                except Exception as err:
                    logging.error("Error while using ARMA method on ship_pt: %s, prod_hierarchy: %s, part_number: %s", ship_pt, prod_h, part_no)
                    logging.error(traceback.format_exc())

                # SARIMA
                try:
                    rmse_sarima, mape_sarima, pred_sarima = algo_obj.sarima_calculate()
                    if (rmse_sarima < min_rmse):
                        min_rmse = rmse_sarima
                        min_algo = "SARIMA"
                        min_mape = mape_sarima
                        min_pred = pred_sarima
                except Exception as err:
                    logging.error("Error while using SARIMA method on ship_pt: %s, prod_hierarchy: %s, part_number: %s", ship_pt, prod_h, part_no)
                    logging.error(traceback.format_exc())

                # SES
                try:
                    rmse_ses, mape_ses, pred_ses = algo_obj.ses_calculate()
                    if (rmse_ses < min_rmse):
                        min_rmse = rmse_ses
                        min_algo = "SES"
                        min_mape = mape_ses
                        min_pred = pred_ses
                except Exception as err:
                    logging.error("Error while using SES method on ship_pt: %s, prod_hierarchy: %s, part_number: %s", ship_pt, prod_h, part_no)
                    logging.error(traceback.format_exc())

                # FNN
                try:
                    rmse_fnn, mape_fnn, pred_fnn = algo_obj.fnn_calculate()
                    if (rmse_fnn < min_rmse):
                        min_rmse = rmse_fnn
                        min_algo = "FNN"
                        min_mape = mape_fnn
                        min_pred = pred_fnn
                except Exception as err:
                    logging.error("Error while using FNN method on ship_pt: %s, prod_hierarchy: %s, part_number: %s", ship_pt, prod_h, part_no)
                    logging.error(traceback.format_exc())

                # RNN
                # try:
                #     rmse_rnn, mape_rnn, pred_rnn = algo_obj.rnn_calculate()
                #     if (rmse_rnn < min_rmse):
                #         min_rmse = rmse_rnn
                #         min_algo = "RNN"
                #         min_mape = mape_rnn
                #         min_pred = pred_rnn
                # except Exception as err:
                #     logging.error("Error while using RNN method on ship_pt: %s, prod_hierarchy: %s, part_number: %s", ship_pt, prod_h, part_no)
                #     logging.error(traceback.format_exc())



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
                except Exception as err:
                    logging.error("Error while using HWES method on ship_pt: %s, prod_hierarchy: %s, part_number: %s", ship_pt, prod_h, part_no)
                    logging.error(traceback.format_exc())

                predicted_output = algo_obj.getPredictedValues(min_algo, min_params)
                output_dict[key] = [min_algo, min_rmse, min_mape, predicted_output, min_pred]
                count += 1
                print(count)

        output_file = File(
                open(
                    FileOperations.write_forecast_file(
                        output_dict, input, vol_cluster, int_cluster), 'rb'))
        log_file = File(
                open(
                    os.path.join(settings.MEDIA_ROOT ,"log/app.log"), 'rb'
                )
            )

        output_obj.input = input
        output_obj.output_dict = output_dict
        output_obj.input_dict = dict_data
        output_obj.volume_cluster = vol_cluster
        output_obj.int_cluster = int_cluster
        output_obj.output_file.save(
            'forecast.csv', output_file)

        output_obj.log_file.save(
            'app.log', log_file
        )

        FileOperations.write_excel(output_dict, input, vol_cluster, int_cluster)

        # user_history_obj.user = get_username().user
        # user_history_obj.input = input
        # user_history_obj.output = output_obj
        # user_history_obj.timestamp = datetime.now()
        # user_history_obj.save()




        # if os.path.exists(str(input_file)):
        #     os.remove(str(input_file))

        # if os.path.exists(os.path.abspath(output_file)):
        #     os.remove(os.path.abspath(output_file))

        # if os.path.exists(os.path.abspath(log_file)):
        #     os.remove(os.path.abspath(log_file))

        return True
