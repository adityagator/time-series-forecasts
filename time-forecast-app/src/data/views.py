from django.shortcuts import render
from .forms import InputDataForm
from .models import InputData
from django.http import HttpResponseRedirect


def input_create_view(request):
    form = InputDataForm(request.POST, request.FILES)
    print("This is the form")
    print(form)
    if form.is_valid():
        form.save()
        # form = InputDataForm()
        return HttpResponseRedirect('/processing')
        # input_list = InputData.objects.all()
        # contextp = {
        #     'input_list' : input_list
        # }
        # return render(request, "input/processing.html", contextp)
    context = {
        'form' : form
    }
    return render(request, "input/input_create.html", context)

def processing_view(request):
    input_list = InputData.objects.all()
    
    # print(in)
    context = {
        "input_list" : input_list
    }
    
    return render(request, "input/processing.html", context)