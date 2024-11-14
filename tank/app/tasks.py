# your_app/tasks.py
from celery import shared_task
from .models import WaterUsageData
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import joblib
import os
import numpy as np
import random
from datetime import timedelta, datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.tsa.stattools import adfuller

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from background_task import background
from background_task.models import Task
from datetime import timedelta
from background_task import background
from datetime import datetime
from random import uniform
from .models import WaterUsageData


@background(schedule=3600)  # Start immediately after migration
def retrain_model():
    print("Starting model retraining with all available data...\n")
    
    # Load all data from the database
    data = WaterUsageData.objects.all()
    df = pd.DataFrame.from_records(data.values())

    # Check if data is empty
    if df.empty:
        print("No data available for retraining.")
        return "No data available for retraining."
    
    # Print the columns to check what is available
    print("Columns in the data:", df.columns)

    # Initialize LabelEncoder for categorical columns
    label_encoder = LabelEncoder()

    # Check if 'Season' column exists before applying LabelEncoder
    if 'season' in df.columns:
        df['season'] = label_encoder.fit_transform(df['season'])
    else:
        print("'Season' column not found in data.")

    # Check if 'Time of Day' column exists before applying LabelEncoder
    if 'time_of_day' in df.columns:
        df['time_of_day'] = label_encoder.fit_transform(df['time_of_day'])
    else:
        print("'Time of Day' column not found in data.")

    # Handle missing values (forward fill)
    df.ffill(inplace=True)

    # Feature scaling (Standardization) for relevant columns
    scaler = StandardScaler()
    scaled_columns = ['current_water_level', 'people_using_water', 'temperature', 'water_usage_last_hour']
    df[scaled_columns] = scaler.fit_transform(df[scaled_columns])
    print("hi")
    # Set timestamp as index for time-based modeling
    df.set_index('timestamp', inplace=True)
    df.index = pd.to_datetime(df.index)  # Ensure index is datetime
    df = df.asfreq('H')  # Set frequency to hourly (adjust if different)

    # Train SARIMAX model
    print("hiiii")
    sarimax_model = SARIMAX(df['required_water'], 
                exog=df.drop(columns=['required_water']), 
                order=(1, 0, 1), 
                seasonal_order=(1, 0, 1, 24))

    sarimax_result = sarimax_model.fit(maxiter=1, disp=True)
    print("hiiiiiiiiiiii")
    # Save the retrained model
    # Get the directory where the current script is located
    current_dir = os.path.dirname(__file__)

    # Define the path to the 'models' directory (make sure it exists)
    models_dir = os.path.join(current_dir, 'models')

    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Define the model path inside the 'models' directory
    model_path = os.path.join(models_dir, 'latest_sarimax_model.pkl')

    # Save the SARIMAX model
    joblib.dump(sarimax_result, model_path)
    print(f"Model retrained and saved at {model_path}.")
        # Extract coefficients
    coefficients = sarimax_result.params
    print(coefficients.shape)
    # Path to save model coefficients
    coefficients_path = os.path.join(os.path.dirname(__file__), 'models', 'model_coefficients.txt')

    # Ensure the directory exists
    os.makedirs(os.path.dirname(coefficients_path), exist_ok=True)

    # Delete the old file if it exists
    if os.path.exists(coefficients_path):
        os.remove(coefficients_path)
        print(f"Old coefficients file removed: {coefficients_path}")
    
    # Save new coefficients to the file
    np.savetxt(coefficients_path, coefficients)
    print(f"Model coefficients saved at {coefficients_path}.")

    return "Model retrained and saved with all available data."


@background(schedule=3600)  # Schedule to run every hour (3600 seconds)
def add_water_usage_record():
    # Randomly generating values for the fields (you can replace this with actual data)
    current_water_level = uniform(0, 100)  # Example random water level
    people_using_water = uniform(1, 10)    # Example random number of people
    temperature = uniform(20, 40)          # Example random temperature
    season = 'Summer'  # Example fixed value, can be dynamically calculated
    time_of_day = 'Afternoon'  # Example fixed value, can be dynamically calculated
    water_usage_last_hour = uniform(0, 100)  # Example random usage
    required_water = uniform(50, 100)  # Example required water

    # Create a new record in the WaterUsageData model
    # WaterUsageData.objects.create(
    #     timestamp=datetime.now(),
    #     current_water_level=current_water_level,
    #     people_using_water=people_using_water,
    #     temperature=temperature,
    #     season=season,
    #     time_of_day=time_of_day,
    #     water_usage_last_hour=water_usage_last_hour,
    #     required_water=required_water
    # )
    print("successfully created")