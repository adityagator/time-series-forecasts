from django.db import models

# Create your models here.
class InputData(models.Model):
    algorithms = models.TextField()
    # algorithms = models.ChoiceField()
    cluster = models.BooleanField(default=True)
    log = models.BooleanField(default=False)
    graph = models.BooleanField(default=True)
    deepLearning = models.BooleanField(default=False)
    file = models.FileField(upload_to="input/")

class OutputData(models.Model):
    forecast_file = models.FileField()
    cluster_file = models.FileField()
    log_file = models.FileField()