from django.shortcuts import render, get_object_or_404
from .forms import InputDataForm
from .models import InputData
from .models import OutputData
from django.http import HttpResponseRedirect
from processing.process import Process
from django.core.mail import EmailMessage
from django.conf import settings
import boto3
import boto3.session
from django.contrib.auth.decorators import login_required
from collections import Counter
from processing.process_demo import process_demo
import requests

@login_required
def input_create_view(request):
    form = InputDataForm(request.POST, request.FILES)
    if form.is_valid():
        input_data = form.save()
        link_with_id = '/process/' + str(input_data.id)
        return HttpResponseRedirect(link_with_id)
    context = {
        'form' : form,
        'Process': Process
    }
    return render(request, "input/input_create.html", context)

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

# def send_mail(input, output_file, log_file):
#     try:
#         mail = EmailMessage("Output file", "Body of email", settings.EMAIL_HOST_USER, [input.email])
#         mail.attach(output_file.name, output_file.read())
#         mail.attach(log_file.name, log_file.read())
#         mail.send()
#         print("Email sent")
#         return True
#     except:
#         print("E-mail was not sent")
#         return False

def combine_cluster(vol, int, res = [[0]*3]*3):
    final_res = []
    for i in range(0, len(vol)):
        for j in range(0, len(int)):
            # res[i][j] = vol[i] + int[j]
            final_res.append(vol[i] + int[j])
    temp = final_res[2]
    final_res[2] = final_res[0]
    final_res[0] = temp

    temp = final_res[5]
    final_res[5] = final_res[3]
    final_res[3] = temp

    temp = final_res[8]
    final_res[8] = final_res[6]
    final_res[6] = temp
    return final_res

@login_required
def dashboard_view(request, id):
    input = InputData.objects.get(id=id)
    output_data = get_object_or_404(OutputData, input=input)
    output = OutputData.objects.get(input=input)
    # email_flag = send_mail(input, output.output_file, output.log_file)
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
    combine_vol_int = []
    vol_count = []
    int_count = []
    cluster_flag = False
    if len(output.volume_cluster) > 0:
        cluster_flag = True
        volume_cluster = getSummary(output.volume_cluster)
        int_cluster = getSummary(output.int_cluster)
        combine_vol_int = combine_cluster(volume_cluster, int_cluster, combine_vol_int)
        vol_count = output_data.vol_count
        int_count = output_data.int_count
    
    context = {
        'ship_pt_arr': ship_pt_arr,
        'prod_h_arr': prod_h_arr,
        'part_no_arr': part_no_arr,
        'output_dict': output_dict,
        'input_dict': input_dict,
        'inputObj': input,
        'volume_cluster': volume_cluster,
        'int_cluster': int_cluster,
        'cluster_flag': cluster_flag,
        'output_data': output_data,
        'i': 0,
        'combine_vol_int': combine_vol_int,
        'top5_dict': output_data.top5_dict,
        'vol_count': vol_count,
        'int_count': int_count,
    }
    return render(request, "output/dashboard.html", context)

def covid_view(request):
    response = requests.get("https://api.covidtracking.com/v1/us/daily.csv").text
    splits = response.splitlines()
    keys = splits[0]
    values = splits[1:]

    covid_dict = {}

    date = []
    hospitalized = []
    in_icu = []
    on_ventilator = []
    death = []
    positive = []

    for value in reversed(values):
        value_split = value.split(",")
        if value_split[22] == "" or value_split[20] == "" or value_split[7] == "" or value_split[9] == "" or value_split[19] == "":
            continue;
        date.append(value_split[0])
        positive.append(int(value_split[2]))
        hospitalized.append(int(value_split[5]))
        in_icu.append(int(value_split[7]))
        on_ventilator.append(int(value_split[9]))
        death.append(int(value_split[12]))

    key_split = keys.split(",")
    covid_dict[key_split[2]] = positive
    covid_dict[key_split[5]] = hospitalized
    covid_dict[key_split[7]] = in_icu
    covid_dict[key_split[9]] = on_ventilator
    covid_dict[key_split[12]] = death

    output_obj = process_demo(covid_dict)

    # print(covid_dict)

    context = {
        'date_arr': date,
        'input_dict': covid_dict,
        'output_dict': output_obj.output_dict,
        'top5_dict': output_obj.top5_dict
    }
    return render(request, "output/covid.html", context)
