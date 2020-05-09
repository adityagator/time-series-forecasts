from django.db import models
from jsonfield import JSONField
from django.utils import timezone

class InputData(models.Model):
    forecast = models.BooleanField(default=True)
    cluster = models.BooleanField(default=True)
    file = models.FileField(upload_to="input/", default=None)
    # email = models.EmailField(default=None, blank=True)
    # email = models.BooleanField(default=False)

class OutputData(models.Model):
    input = models.OneToOneField(
        InputData,
        on_delete=models.CASCADE,
        primary_key=True,
        default=0
        )
    output_file = models.FileField()
    log_file = models.FileField()
    input_dict = JSONField(default={'default': 'default'})
    output_dict = JSONField(default={'default': 'default'})
    volume_cluster = JSONField(default={'default': 'default'})
    int_cluster = JSONField(default={'default': 'default'})
    top5_dict = JSONField(default={'default': 'default'})
    vol_count = JSONField(default={'default': 'default'})
    int_count = JSONField(default={'default': 'default'})

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
    
class UserHistory(models.Model):
    user = models.TextField(unique=False, default="dummy")
    input = models.OneToOneField(
        InputData,
        on_delete=models.CASCADE,
        primary_key=True,
        default=0
        )
    output = models.OneToOneField(
        OutputData,
        on_delete=models.CASCADE,
        # primary_key=True,
        default=0
        )
    timestamp = models.DateTimeField(default=timezone.now())
