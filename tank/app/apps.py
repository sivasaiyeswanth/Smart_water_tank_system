from django.apps import AppConfig

class AppConfig(AppConfig):
    name = 'app'

    def ready(self):
        from .tasks import retrain_model  # Import tasks inside ready method
        retrain_model(repeat=3600)  # Call the task
        from .tasks import add_water_usage_record
        # Schedule the task to run every hour
        add_water_usage_record(repeat=3600)
