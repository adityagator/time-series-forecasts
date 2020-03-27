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
import warnings
warnings.filterwarnings("ignore")

class Process():
    def run(input):
        print(input.algorithms)
        input_file = os.path.join(settings.MEDIA_ROOT, input.file.name)
        dict_data = FileOperations.read_file(file=input_file)
        print("dict data:")
        print(dict_data)
        output_dict = {}
        output_obj = OutputData()
        for key, value in dict_data.items():
            min_rmse = sys.maxsize
            min_mape = sys.maxsize
            min_algo = ""
            algo_obj = Algorithms(value,
                                  value[0:Constants.TRAINING_MONTHS],
                                  value[Constants.TRAINING_MONTHS:])
            min_params = []
            # ARIMA Algorithm
            if "ARIMA" in input.algorithms:
                rmse_arima, mape_arima = algo_obj.arima_calculate()
                if(rmse_arima < min_rmse):
                    min_rmse = rmse_arima
                    min_algo = "ARIMA"
                    min_mape = mape_arima
                print("rmse is :", rmse_arima, " ", mape_arima, " ", "ARIMA")

            # Moving Average
            if "MA" in input.algorithms:
                rmse_ma, mape_ma = algo_obj.ma_calculate()
                if(rmse_ma < min_rmse):
                    min_rmse = rmse_ma
                    min_algo = "MOVING AVERAGE"
                    min_mape = mape_ma
                print("rmse is :", rmse_ma, " ", mape_ma, " ", "MA")

            # Auto Reg
            if "AR" in input.algorithms:
                rmse_ar, mape_ar = algo_obj.ar_calculate()
                if (rmse_ar < min_rmse):
                    min_rmse = rmse_ar
                    min_algo = "AR"
                    min_mape = mape_ar
                print("rmse is :", rmse_ar, " ", mape_ar, " ", "AR")

            # ARMA
            if "ARMA" in input.algorithms:
                rmse_arma, mape_arma = algo_obj.arma_calculate()
                if(rmse_arma < min_rmse):
                    min_rmse = rmse_arma
                    min_algo = "ARMA"
                    min_mape = mape_arma
                print("rmse is :", rmse_arma, " ", mape_arma, " ", "ARMA")

            # SARIMA
            if "SARIMA" in input.algorithms:
                rmse_sarima, mape_sarima = algo_obj.sarima_calculate()
                if (rmse_sarima < min_rmse):
                    min_rmse = rmse_sarima
                    min_algo = "SARIMA"
                    min_mape = mape_sarima
                print(
                    "rmse is :",
                    rmse_sarima,
                    " ",
                    mape_sarima,
                    " ",
                    "SARIMA")

            # SES
            if "SES" in input.algorithms:
                rmse_ses, mape_ses = algo_obj.ses_calculate()
                if (rmse_ses < min_rmse):
                    min_rmse = rmse_ses
                    min_algo = "SES"
                    min_mape = mape_ses
                print("rmse is :", rmse_ses, " ", mape_ses, " ", "SES")

            # FNN
            # if input.deepLearning:
            #     rmse, mape = algo_obj.fnn_calculate(value)
            #     if (rmse < min_rmse):
            #         min_rmse = rmse
            #         min_algo = "FNN"
            #         min_mape = mape
            #     print("rmse is :", rmse, " ", mape, " ", "FNN")

                # Holt-Winters method
            if "HWES" in input.algorithms:
                print('optimised HWES Method :')
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
                print('final values: ', alpha_final, beta_final, gamma_final)

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
                    min_rmse = rmse_hwes
                    min_algo = "HWES"
                    min_mape = algo_obj.mean_absolute_percentage_error(predictions)
                    min_params = [alpha_final, beta_final, gamma_final]
                print("rmse is :", rmse_hwes, " ", "HWES")
                print()

                print('____________________________')

            predicted_output = algo_obj.getPredictedValues(min_algo, min_params)
            print(predicted_output)
            output_dict[key] = [min_algo, min_rmse, min_mape, predicted_output, min_params]

        # forecast = FileOperations.write_forecast_file(output_dict, input)
        # f = open(forecast)
        # forecast_file = File(f)
        output_obj.input = input
        output_obj.forecast_file.save(
            'forecast', File(
                open(
                    FileOperations.write_forecast_file(
                        output_dict, input))))
        return True
        
