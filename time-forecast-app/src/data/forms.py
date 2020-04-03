from django import forms
from .models import InputData
from django.forms import widgets

class InputDataForm(forms.ModelForm):
    class Meta:
        model = InputData
        # fields = [
        #     'forecast',
        #     'cluster',
        #     'file'
        # ]
        fields = {
            'forecast': widgets.CheckboxInput(attrs={'class' : 'forecast-input'}),
            'cluster': widgets.CheckboxInput(attrs={'class' : 'forecast-input'}),
            'file': widgets.FileInput(attrs={'class' : 'file-input'}),
        }