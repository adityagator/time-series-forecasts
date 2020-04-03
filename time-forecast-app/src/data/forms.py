from django import forms
from .models import InputData

class InputDataForm(forms.ModelForm):
    class Meta:
        model = InputData
        fields = [
            'forecast',
            'cluster',
            'file'
        ]