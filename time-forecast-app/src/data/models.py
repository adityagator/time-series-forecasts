from django.db import models
from multiselectfield import MultiSelectField

# Create your models here.
class InputData(models.Model):
    # algorithms = models.TextField()
    ALGO_CHOICES = (
        ('AR', 'Auto Regression'),
        ('ARIMA', 'ARIMA'),
    )
    # algorithms = models.CharField(max_length = 100, choices = ALGO_CHOICES)
    algorithms = MultiSelectField(choices = ALGO_CHOICES)
    cluster = models.BooleanField(default=True)
    log = models.BooleanField(default=False)
    graph = models.BooleanField(default=True)
    deepLearning = models.BooleanField(default=False)
    file = models.FileField(upload_to="input/")
    # isProcessed = models.BooleanField(default=False)

class OutputData(models.Model):
    forecast_file = models.FileField()
    cluster_file = models.FileField()
    log_file = models.FileField()