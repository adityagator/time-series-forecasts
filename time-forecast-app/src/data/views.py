from django.shortcuts import render, get_object_or_404
from .forms import InputDataForm
from .models import InputData
from .models import OutputData
from django.http import HttpResponseRedirect
from processing.process import Process

def input_create_view(request):
    form = InputDataForm(request.POST, request.FILES)
    if form.is_valid():
        input_data = form.save()
        link_with_id = '/process/' + str(input_data.id)
        return HttpResponseRedirect(link_with_id)
    context = {
        'form' : form
    }
    return render(request, "input/input_create.html", context)

def output_detail_view(request, id):
    input = InputData.objects.get(id=id)
    # Process.run(input)
    # output_data = OutputData.objects.create(input=input, forecast_file=input.file)
    output_data = get_object_or_404(OutputData, input=input)
    context = {
        "output_data" : output_data
    }
    return render(request, "output/output_detail.html", context)
