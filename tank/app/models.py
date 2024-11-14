import profile
from pyexpat import model
from unicodedata import category
from django.db import models
from django.contrib.auth.models import User


# Assuming you have a model for the task
class Task(models.Model):
    name = models.CharField(max_length=255)
    repeat = models.FloatField()  # Store repeat as seconds, FloatField to allow precision if needed
    # Other fields you may have...

class TankData(models.Model):
    current_level = models.IntegerField()  # e.g., 0 to 100
    threshold = models.IntegerField(default=50)  # User-defined threshold
    pump_status = models.BooleanField(default=False)  # On or Off
    last_updated = models.DateTimeField(auto_now_add=True)




class WaterUsageData(models.Model):
    timestamp = models.DateTimeField(primary_key=True)
    current_water_level = models.FloatField()
    people_using_water = models.FloatField()
    temperature = models.FloatField()
    season = models.TextField()
    time_of_day = models.TextField()
    water_usage_last_hour = models.FloatField()
    required_water = models.FloatField()

    class Meta:
        db_table = 'water_usage_data'
