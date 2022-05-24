from django.http import HttpResponse
from django.shortcuts import redirect, render

from files.forms import MaterialsForm

def home(request):
    if request.method == 'POST':
        form = MaterialsForm(request.POST, request.FILES)
        if form.is_valid:
            form.save()
            return redirect('results')
    else:
        form = MaterialsForm()

    return render(request, 'home.html', {'form': form})

def about(request):
    return render(request,"about.html")

def results(request):
    return render(request,"results.html")