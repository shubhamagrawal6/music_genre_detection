from django import forms
from .models import materials

class MaterialsForm(forms.ModelForm):
    class Meta:
        model = materials
        fields = ('title', 'files')