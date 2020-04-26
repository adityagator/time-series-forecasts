from django.shortcuts import render
from data.models import InputData
from data.models import OutputData
from processing.process import Process
from django.http import HttpResponseRedirect
import threading
import os
from forecast_project import settings
from processing.file_operations import FileOperations
from django.contrib.auth.decorators import login_required

@login_required
def processing_view(request, id):
    input = InputData.objects.get(id=id)
    if Process.run(input):
        link_with_id = '/dashboard/' + str(input.id)
        return HttpResponseRedirect(link_with_id)
    
    context = {
        "input" : input
    }
    return render(request, "processing.html", context)

# def start_processing(input):
#     print("Thread starting")
#     Process.run(input)
#     print("Thread done")
#     link_with_id = '/dashboard/' + str(input.id)
#     return HttpResponseRedirect(link_with_id)
    


# def processing_view(request, id):
#     input = InputData.objects.get(id=id)
#     input_file = os.path.join(settings.MEDIA_ROOT, input.file.name)
#     dict_data = FileOperations.read_file(file=input_file)
#     input_length = len(dict_data)
#     print("before creating thread")
#     x = threading.Thread(target=start_processing, args=(input,))
#     print("before starting thread")
#     x.start()
#     print("waiting for thread to complete")
#     context = {
#         "input" : input,
#         "process": Process,
#         "x": x,
#         "input_length": input_length
#     }
#     return render(request, "processing.html", context)

