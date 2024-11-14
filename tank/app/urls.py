from django.urls import path
from . import views

urlpatterns = [
    path('', views.tank_status, name='tank_status'),
    path('predict_water_usage/', views.predict_water_usage, name='predict_water_usage'),
    path('update_tank_data/', views.update_tank_data, name='update_tank_data'),
    ]
