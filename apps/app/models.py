# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.db import models
from django.contrib.auth.models import User

from django.contrib import admin

from apps.predictions.models import SAGRAData


class SAGRADataAdmin(admin.ModelAdmin):
    pass


admin.site.register(SAGRAData, SAGRADataAdmin)
