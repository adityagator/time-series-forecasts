from django.db import models

class InputData(models.Model):
    forecast = models.BooleanField(default=True)
    cluster = models.BooleanField(default=True)
    file = models.FileField(upload_to="input/", default=None)

class OutputData(models.Model):
    input = models.OneToOneField(
        InputData,
        on_delete=models.CASCADE,
        primary_key=True,
        default=0
        )
    output_file = models.FileField()
    log_file = models.FileField()
