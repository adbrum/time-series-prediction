from django.urls import path
from . import views

urlpatterns = [
    #     path('', views.home, name='home'),
    #     path('open/', views.open_file, name='file'),
    #     path('open_file_prediction/', views.open_file_prediction,
    #          name='simple_upload_prediction'),
    #     path('uploads/simple/', views.simple_upload, name='simple_upload'),
    #     path('uploads/simple_prediction/', views.simple_upload_prediction,
    #     name='simple_upload_prediction'),
    path('', views.automodel_prediction, name='automodel'),
    #     path('uploads/form/', views.model_form_upload, name='model_form_upload'),

]
