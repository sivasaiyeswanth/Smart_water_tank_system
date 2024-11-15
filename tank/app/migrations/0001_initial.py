# Generated by Django 4.1.7 on 2024-08-18 12:15

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="TankData",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("current_level", models.IntegerField()),
                ("threshold", models.IntegerField(default=50)),
                ("pump_status", models.BooleanField(default=False)),
                ("last_updated", models.DateTimeField(auto_now=True)),
            ],
        ),
    ]
