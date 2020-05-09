from django.contrib import admin
from data.models import InputData
from data.models import OutputData

# Register your models here.
admin.site.register(InputData)
admin.site.register(OutputData)