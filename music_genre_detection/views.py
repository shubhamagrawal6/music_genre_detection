from django.http import HttpResponse
from django.shortcuts import redirect, render

from files.forms import MaterialsForm

def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request,"about.html")

def results(request):

    genre = "blues"



    return render(request, "results.html", {"genre": genre, })