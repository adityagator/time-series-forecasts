from django.db import models
# from multiselectfield import MultiSelectField


# Create your models here.
class InputData(models.Model):
    # algorithms = models.TextField()
    # ALGO_CHOICES = (
    #     ('AR', 'Auto Regression'),
    #     ('ARIMA', 'ARIMA'),
    #     ('SARIMA', 'SARIMA'),
    #     ('ARMA', 'ARMA'),
    #     ('MA', 'Moving Average'),
    #     ('SES', 'Simple Exponential Smoothing'),
    #     ('HWES', 'HWES')
    # )
    # algorithms = models.CharField(max_length = 100, choices = ALGO_CHOICES)
    # algorithms = MultiSelectField(choices = ALGO_CHOICES, default=None)
    forecast = models.BooleanField(default=True)
    cluster = models.BooleanField(default=True)
    # log = models.BooleanField(default=False)
    # graph = models.BooleanField(default=True)
    # deepLearning = models.BooleanField(default=False)
    file = models.FileField(upload_to="input/")
    # isProcessed = models.BooleanField(default=False)

class OutputData(models.Model):
    input = models.OneToOneField(
        InputData,
        on_delete=models.CASCADE,
        primary_key=True,
        default=0
        )
    forecast_file = models.FileField()
    cluster_file = models.FileField()
    log_file = models.FileField()

    # def __init__(self, forecast, cluster, log):
    #     self.forecast_file = forecast
    #     self.cluster_file = cluster
    #     self.log_file = log
    
    def set_forecast_file(self, forecast):
        self.forecast_file = forecast

    def get_forecast_file():
        return self.forecast_file
    
    def set_cluster_file(self, cluster):
        self.cluster_file = cluster

    def get_cluster_file():
        return self.cluster_file
    
    def set_log_file(self, log):
        self.log_file = log

    def get_log_file():
        return self.log_file
