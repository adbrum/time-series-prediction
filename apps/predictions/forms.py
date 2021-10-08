from django import forms
from .models import SAGRAData


class DocumentForm(forms.ModelForm):
    class Meta:
        model = SAGRAData
        fields = '__all__'
