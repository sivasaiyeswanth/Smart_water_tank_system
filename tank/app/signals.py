from django.db.models.signals import post_migrate
from django.dispatch import receiver
from .tasks import retrain_model
from background_task.models import Task
from datetime import timedelta

@receiver(post_migrate)
def run_retrain_model(sender, **kwargs):
    print("Setting up retrain_model to run every hour.")

    # Check if a retrain_model task is already scheduled
    if not Task.objects.filter(task_name='app.tasks.retrain_model').exists():
        # Schedule the task to run every hour
        retrain_model(repeat=timedelta(hours=1))





from datetime import timedelta
from background_task import background
from .models import Task

# Define the task with a repeat interval in seconds
@background(schedule=60)
def retrain_model(repeat):
    # Perform the task you want to run periodically
    print("Retraining model...")

    # You can schedule the next run by converting `repeat` into seconds (timedelta).
    # This assumes the repeat interval is in seconds.
    next_run_in_seconds = repeat
    retrain_model(repeat=next_run_in_seconds, schedule=next_run_in_seconds)

# Signal or method where the task is scheduled to run every hour
def run_retrain_model():
    repeat_duration = timedelta(hours=1).total_seconds()  # Convert 1 hour to seconds

    # Schedule the retrain_model task to run every hour (repeat is in seconds)
    retrain_model(repeat=repeat_duration)
