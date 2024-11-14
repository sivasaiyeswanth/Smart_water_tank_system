import os
import django
import pandas as pd
from tank.settings import BASE_DIR  # Assuming your project is named 'tank'

# Configure Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tank.settings")
django.setup()

# Import models after Django setup
from app.models import TankData, WaterUsageData

# Retrieve data and convert it to a DataFrame
data = WaterUsageData.objects.all()
df = pd.DataFrame.from_records(data.values())
print(df.columns)
print(df.head())
