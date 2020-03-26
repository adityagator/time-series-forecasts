from django.shortcuts import render
from data.models import InputData
from data.models import OutputData
from processing.process import Process
from django.http import HttpResponseRedirect

# Create your views here.
def processing_view(request, id):
    input = InputData.objects.get(id=id)
    if Process.run(input):
        link_with_id = '/output/' + str(input.id)
        return HttpResponseRedirect(link_with_id)
    
    context = {
        "input" : input
    }
    return render(request, "processing.html", context)
