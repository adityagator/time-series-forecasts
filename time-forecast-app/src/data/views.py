from django.shortcuts import render
from .forms import InputDataForm
from .models import InputData
from django.http import HttpResponseRedirect


def input_create_view(request):
    form = InputDataForm(request.POST, request.FILES)
    if form.is_valid():
        form.save()
        form = InputDataForm()
        return HttpResponseRedirect('/help')
    context = {
        'form' : form
    }
    return render(request, "input/input_create.html", context)
