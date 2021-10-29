# -*- encoding: utf-8 -*-
from django.db import models
from django.contrib.auth.models import User


class SAGRAData(models.Model):
    EMA = models.CharField(max_length=100)
    date_occurrence = models.DateTimeField()
    # 'ºC'
    average_temperature = models.CharField(max_length=10)
    maximum_temperature = models.CharField(max_length=10)
    minimum_temperature = models.CharField(max_length=10)
    # '%'
    average_humidity = models.CharField(max_length=10)
    maximum_humidity = models.CharField(max_length=10)
    minimum_humidity = models.CharField(max_length=10)
    # 'm/s'
    average_wind_speed = models.CharField(max_length=10)
    maximum_wind_speed = models.CharField(max_length=10)
    # 'ºC'
    average_grass_temperature = models.CharField(max_length=10)
    maximum_grass_temperature = models.CharField(max_length=10)
    minimum_grass_temperature = models.CharField(max_length=10)
    # 'mm'
    rainfall = models.CharField(max_length=10)
    # 'kj/m2'
    RSG = models.CharField(max_length=10)
    # 'graus'
    DV = models.CharField(max_length=10)
    # 'mm'
    ET0 = models.CharField(max_length=10)

    # created_on = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.pk) + ' - ' + self.EMA + ' - ' + str(self.id)
