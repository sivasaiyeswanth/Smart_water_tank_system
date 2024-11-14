from django.contrib import admin
from atexit import register
# Register your models here.
from .models import *

admin.site.register(TankData)