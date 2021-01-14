from django import forms
from .models import InputData
from django.forms import widgets

"""
data for input form

"""
class InputDataForm(forms.ModelForm):
    class Meta:
        model = InputData
        fields = [
            'forecast',
            'cluster',
            # email for future use
            # 'email',
            'file'
        ]