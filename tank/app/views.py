import json
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from .models import TankData, WaterUsageData
from sklearn.preprocessing import StandardScaler
from django.views.decorators.csrf import csrf_exempt
import os

# Load SARIMAX model and scaler (assuming you've saved them after training)
# For example:
# scaler = StandardScaler()
# scaler.fit(X_train)

def tank_status(request):
    # Get the latest tank data
    tank = TankData.objects.latest('last_updated')
    
    # Check if the request is an AJAX call
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        # Return JSON data for AJAX requests
        data = {
            'current_level': tank.current_level,
            'threshold': tank.threshold,
            'pump_status': tank.pump_status,
            'last_updated': tank.last_updated.strftime('%Y-%m-%d %H:%M:%S')
        }
        return JsonResponse(data)

    # Handle non-AJAX requests (normal page load)
    if tank.current_level < tank.threshold:
        tank.pump_status = True
    else:
        tank.pump_status = False

    if request.method == 'POST':
        threshold = request.POST.get('threshold')
        tank.threshold = int(threshold)
        tank.save()

    return render(request, 'tank_status.html', {'tank': tank})



from django.http import JsonResponse
from django.shortcuts import render
import numpy as np
import pandas as pd
from .models import WaterUsageData
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import datetime
import datetime
import pandas as pd

import datetime
import os

import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib
from .models import WaterUsageData  # Make sure to import the correct model

# def predict_water_usage(request):
#     predicted_percentage = None  # Initialize a variable to hold the prediction result

#     if request.method == 'POST':
#         # Gather data from POST request (get user input or use default if not provided)
#         season = request.POST.get('season', None)
#         time_of_day = request.POST.get('time_of_day', None)
#         people_using_water = request.POST.get('people_using_water', None)
#         temperature = request.POST.get('temperature', None)
#         water_usage_last_hour = request.POST.get('water_usage_last_hour', None)

#         # Fill missing values with averages from past 24 hours
#         past_data = WaterUsageData.objects.order_by('-timestamp')[:24]
#         avg_values = {
#             'people_using_water': np.mean([data.people_using_water for data in past_data]),
#             'temperature': np.mean([data.temperature for data in past_data]),
#             'water_usage_last_hour': np.mean([data.water_usage_last_hour for data in past_data])
#         }

#         # If user has not filled the form, use the average from past data
#         people_using_water = float(people_using_water) if people_using_water else avg_values['people_using_water']
#         temperature = float(temperature) if temperature else avg_values['temperature']
#         water_usage_last_hour = float(water_usage_last_hour) if water_usage_last_hour else avg_values['water_usage_last_hour']

#         # Encode 'season' and 'time_of_day' columns using LabelEncoder or predefined mappings
#         label_encoder = LabelEncoder()
#         season_encoded = label_encoder.fit_transform([season])[0] if season else 0
#         time_of_day_encoded = label_encoder.fit_transform([time_of_day])[0] if time_of_day else 0

#         # Prepare input data for the model (using pandas DataFrame)
#         input_data = pd.DataFrame([{
#             'season': season_encoded,
#             'time_of_day': time_of_day_encoded,
#             'people_using_water': people_using_water,
#             'temperature': temperature,
#             'water_usage_last_hour': water_usage_last_hour
#         }])

#         # Add a timestamp column for prediction (you can choose not to use this if your model doesn't need it)
#         input_data['timestamp'] = datetime.datetime.now()

#         # Load the retrained SARIMAX model (change this to the path where your model is stored)
#         model_path = os.path.join(os.path.dirname(__file__), 'models', 'latest_sarimax_model1.pkl')
#         model = joblib.load(model_path)

#         # Ensure input data is properly scaled (use StandardScaler to scale continuous variables)
#         scaler = StandardScaler()
#         scaled_columns = ['people_using_water', 'temperature', 'water_usage_last_hour']
#         input_data[scaled_columns] = scaler.fit_transform(input_data[scaled_columns])

#         # Set the timestamp as the index (if your model requires this)
#         input_data.set_index('timestamp', inplace=True)

#         # Check if 'required_water' exists before trying to drop it (assuming it's the target variable)
#         if 'required_water' in input_data.columns:
#             exog_data = input_data.drop(columns=['required_water'])  # Explanatory variables for prediction
#         else:
#             exog_data = input_data  # If column doesn't exist, use input_data as is

#         # Make the prediction using the SARIMAX model
#         prediction = model.predict(start=len(input_data), end=len(input_data), exog=exog_data)

#         # Calculate the predicted percentage (assuming it's a value that needs to be scaled to percentage)
#         predicted_percentage = prediction.iloc[0] * 100  # Accessing the first prediction

#         # Return the prediction result as a JSON response
#         return JsonResponse({'predicted_percentage': predicted_percentage})

#     # Render the page with the prediction result (if available)
#     return render(request, 'predict_water_usage.html', {'predicted_percentage': predicted_percentage})




import os
import joblib
from django.http import JsonResponse
from sklearn.preprocessing import StandardScaler

def predict_water_usage(request):
    """
    This function predicts the required water level based on the SARIMAX model.
    It uses input features from the request, prepares them, and applies the SARIMAX model to predict the required water.
    """
    default_current_level = TankData.objects.first().current_level if TankData.objects.exists() else 50  # Set to 50 if no data
        
    # Example: Get input features from the request POST data
    current_water_level = request.POST.get('current_water_level') # Current water level
    people_using_water = request.POST.get('people_using_water') # Number of people using water
    temperature = request.POST.get('temperature')  # Temperature
    
    current_water_level = float(current_water_level) if current_water_level else float(default_current_level)

    print(people_using_water)
    # Get the past 24 data points for season and time of day (dummy data used here, replace with actual logic)
    # This could be implemented based on the actual application logic, such as fetching from the database.
    past_data = WaterUsageData.objects.all().order_by('-timestamp')[:24]  # Replace with actual model data
    past_data1 = WaterUsageData.objects.all().order_by('-timestamp')[:5]
    for row in past_data1:
        print(row) 
        for field in row._meta.fields:
            field_name = field.name
            field_value = getattr(row, field_name)  # Get the value of the field
            print(f"{field_name}: {field_value}")
    print("-" * 40)  # Print a separator between rows
    print(past_data1)
    if not people_using_water:
        people_using_water_values = [data.people_using_water for data in past_data]
        people_using_water = np.mean(people_using_water_values) if people_using_water_values else 0
    
    if not temperature:
        temperature_values = [data.temperature for data in past_data]
        temperature = np.mean(temperature_values) if temperature_values else 0
            
    season_data = [data.season for data in past_data]  # Replace with actual logic for encoding 'season'
    time_of_day_data = [data.time_of_day for data in past_data]  # Replace with actual logic for encoding 'time_of_day'
    


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

    model = sarimax_model.fit(maxiter=1, disp=True)



    
    season_encoded = np.mean(label_encoder.fit_transform(season_data)) if season_data else 0
    time_of_day_encoded = np.mean(label_encoder.fit_transform(time_of_day_data)) if time_of_day_data else 0
    
    # Calculate water usage in the last hour (example calculation)
    last_water_level = past_data[0].current_water_level if past_data else 0
    previous_water_level = past_data[1].current_water_level if len(past_data) > 1 else 0
    water_usage_last_hour = last_water_level - previous_water_level
        
    from datetime import datetime, timedelta

    # Step 3: Prepare the feature vector (same order as the model's coefficients)
    # Get the current timestamp and calculate the next hour's timestamp
    current_timestamp = datetime.now()
    next_hour_timestamp = current_timestamp + timedelta(hours=1)

    # Format the timestamp as a string (you can adjust the format as needed)
    next_hour_timestamp_str = next_hour_timestamp.strftime('%Y-%m-%d %H:%M:%S')

    # Step 3: Prepare the feature vector (same order as the model's coefficients)
    print(current_water_level)
    print(people_using_water)
    input_data = {
        'timestamp':[next_hour_timestamp_str],
        'current_water_level': [current_water_level],
        'people_using_water': [people_using_water],
        'temperature': [temperature],
        'season': [season_encoded],
        'time_of_day': [time_of_day_encoded],
        'water_usage_last_hour': [water_usage_last_hour],
        
    }
    
    input_data = pd.DataFrame(input_data)
    exog_data = input_data  # If column doesn't exist, use input_data as is
    # Convert timestamp to datetime and set as index
    input_data['timestamp'] = pd.to_datetime(input_data['timestamp'])
    input_data.set_index('timestamp', inplace=True)
    # Step 4: Scale the continuous variables using StandardScaler (if needed)
    
    input_data[scaled_columns] = scaler.transform(input_data[scaled_columns])
    print(input_data)
    # Define the features and target for prediction
    if 'required_water' in input_data.columns:
        X_test = input_data.drop(columns=['required_water'])
    else:
        # Handle the case where the column is missing, maybe set X_test to input_data itself
        X_test = input_data
   
    
    forecast = model.get_forecast(steps=1, exog=X_test)
    forecast_df = forecast.conf_int()  # Confidence intervals for forecast
    forecast_df['Predicted Required Water (%)'] = forecast.predicted_mean
    
    print("\n\n\n\n")
    print(forecast_df[['Predicted Required Water (%)']])
    k=forecast_df[['Predicted Required Water (%)']]
    print("\n\n\n\n")
    
    print(k)
# Access the confidence intervals (lower and upper bounds)
    predicted_value = k.iloc[0, 0]
    print("Predicted Required Water (%):", predicted_value) 
    # Step 8: Return the predicted result as a JSON response
    return JsonResponse({"predicted_water_usage": float(predicted_value)})
from django.utils import timezone


@csrf_exempt
def update_tank_data(request):
    if request.method == 'POST':
        try:
            # Parse JSON data from POST request
            data = json.loads(request.body)
            current_level = data.get("current_level")

            # Update the tank data in the database
            tank_data, created = TankData.objects.get_or_create(id=1)  # Adjust ID if needed
            tank_data.current_level = current_level
            tank_data.last_updated = timezone.now()
            tank_data.save()

            return JsonResponse({"status": "success", "message": "Tank data updated"})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    else:
        return JsonResponse({"status": "error", "message": "Only POST method is allowed"}, status=405)