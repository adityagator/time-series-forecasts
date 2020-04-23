from django.shortcuts import render, get_object_or_404
from .forms import InputDataForm
from .models import InputData
from .models import OutputData
from django.http import HttpResponseRedirect
from processing.process import Process
# from django.core.mail import send_mail as sm
from django.core.mail import EmailMessage
from django.conf import settings

def input_create_view(request):
    form = InputDataForm(request.POST, request.FILES)
    print(form)
    if form.is_valid():
        input_data = form.save()
        link_with_id = '/process/' + str(input_data.id)
        return HttpResponseRedirect(link_with_id)
    context = {
        'form' : form,
        'Process': Process
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
    dash_url = '/dashboard/' + str(input.id)
    context = {
        "output_data" : output_data,
        "dash_url" : dash_url
    }
    return render(request, "output/output_detail.html", context)

def dashboard_calculate(request):
    return JsonResponse(data={
        'labels': ['Month 1', 'Month 2', 'Month 3'],
        'data': [20, 30, 25]
    })

def getSummary(cluster):
    low = 0
    mid = 0
    high = 0
    for key, value in cluster.items():
        if cluster[key] == "low":
            low = low + 1
        elif cluster[key] == "medium":
            mid  = mid + 1
        else:
            high = high + 1
    print(low)
    print(mid)
    print(high)
    return [low, mid, high]

def send_mail(output_file, log_file):
    mail = EmailMessage("Output file", "Body of email", settings.EMAIL_HOST_USER, ["adityagator1@gmail.com"])
    mail.attach(output_file.name, output_file.read())
    mail.attach(log_file.name, log_file.read())
    mail.send()
    print("Email sent")

def dashboard_view(request, id):
    input = InputData.objects.get(id=id)
    output_data = get_object_or_404(OutputData, input=input)
    output = OutputData.objects.get(input=input)
    send_mail(output.output_file, output.log_file)
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
    
    volume_cluster = []
    int_cluster = []
    cluster_flag = False
    if len(output.volume_cluster) > 0:
        cluster_flag = True
        volume_cluster = getSummary(output.volume_cluster)
        int_cluster = getSummary(output.int_cluster)
    
        # print(ship_pt_arr)
        # print(part_no_arr)
        # print(prod_h_arr)

    context = {
        'ship_pt_arr': ship_pt_arr,
        'prod_h_arr': prod_h_arr,
        'part_no_arr': part_no_arr,
        # 'url': "dashcalculate/" + str(id),
        # 'labels' : ["Month1", "Month2", "Month3","Month4", "Month5", "Month6","Month7", "Month8", "Month9","Month10", "Month11", "Month12"],
        # 'data' : output_dict[key][3],
        'output_dict': output_dict,
        'input_dict': input_dict,
        'inputObj': input,
        'volume_cluster': volume_cluster,
        'int_cluster': int_cluster,
        'cluster_flag': cluster_flag,
        'output_data': output_data,
        'i': 0
    }
    return render(request, "output/dashboard.html", context)




    