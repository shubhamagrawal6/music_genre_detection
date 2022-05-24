from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.decorators import login_required
from .forms import MaterialsForm
from .models import materials
import os

# Create your views here.
def upload(request):
    if request.method == 'POST':
        form = MaterialsForm(request.POST, request.FILES)
        if form.is_valid:
            form.save()

        return redirect('app-results')

    else:
        form = MaterialsForm()

    return render(request, 'upload.html', {'form': form})

    
    
