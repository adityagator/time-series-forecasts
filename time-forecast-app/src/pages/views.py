from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def home_view(request, *args, **kwargs):
    # return HttpResponse("<h1>Hello World</h1>")
    return render(request, "home.html", {})

def about_view(request, *args, **kwargs):
    my_context = {
        "my_text" : "test",
        "number" : "123"
    }
    return render(request, "about.html", my_context)

def help_view(request, *args, **kwargs):
    return render(request, "help.html", {})

def contact_view(request, *args, **kwargs):
    return render(request, "contact.html", {})
