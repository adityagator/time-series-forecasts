from django.shortcuts import render, get_object_or_404
from .forms import InputDataForm
from .models import InputData
from .models import OutputData
from django.http import HttpResponseRedirect
from processing.process import Process

def input_create_view(request):
    form = InputDataForm(request.POST, request.FILES)
    print(form)
    if form.is_valid():
        input_data = form.save()
        link_with_id = '/process/' + str(input_data.id)
        return HttpResponseRedirect(link_with_id)
    context = {
        'form' : form
    }
    return render(request, "input/input_create.html", context)

# def input_create_view(request):
#     print(request.POST)
#     forecast = request.POST.get('forecast')
#     cluster = request.POST.get('cluster')
#     input_file = request.POST.get('input_file')
#     context = {}
    
#     if input_file is None:
#         return render(request, "input/input_create.html", context)
#     input_data = InputData.objects.create(forecast=forecast, cluster=cluster, file=input_file)
#     link_with_id = '/process/' + str(input_data.id)
#     return HttpResponseRedirect(link_with_id)
    
    

def output_detail_view(request, id):
    input = InputData.objects.get(id=id)
    # Process.run(input)
    # output_data = OutputData.objects.create(input=input, forecast_file=input.file)
    output_data = get_object_or_404(OutputData, input=input)
    context = {
        "output_data" : output_data
    }
    return render(request, "output/output_detail.html", context)

def dashboard_calculate(request):
    return JsonResponse(data={
        'labels': ['Month 1', 'Month 2', 'Month 3'],
        'data': [20, 30, 25]
    })

def dashboard_view(request, id):
    input = InputData.objects.get(id=id)
    output = OutputData.objects.get(input=input)
    output_dict = output.output_dict
    input_dict = output.input_dict
    ship_pt_arr = []
    prod_h_arr = []
    part_no_arr = []
    for key, values in output_dict.items():
        pt, h, no = key.split("^")
        if pt not in ship_pt_arr:
            ship_pt_arr.append(pt)
        if h not in prod_h_arr:
            prod_h_arr.append(h)
        if no not in part_no_arr:
            part_no_arr.append(no)
        # print(ship_pt_arr)
        # print(part_no_arr)
        # print(prod_h_arr)

    context = {
        'ship_pt_arr': ship_pt_arr,
        'prod_h_arr': prod_h_arr,
        'part_no_arr': part_no_arr,
        # 'url': "dashcalculate/" + str(id),
        'labels' : ["Month1", "Month2", "Month3","Month4", "Month5", "Month6","Month7", "Month8", "Month9","Month10", "Month11", "Month12"],
        'data' : output_dict[key][3],
        'output_dict': output_dict
    }
    return render(request, "output/dashboard.html", context)