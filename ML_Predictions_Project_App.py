#Import Modules
import flask
from flask import request, jsonify, render_template, redirect, url_for
import numpy as np
import pandas as pd
from itertools import repeat

#Define Flask App Environment
app = flask.Flask(__name__)
app.config["DEBUG"] = True

#Import Baseline Weather Data
weather_data = pd.read_csv('https://github.com/jherberg462/ml-predictions-project/blob/master/aus_weather/weatherAUS_clean.csv')

#Split Weather Data into X & Y Sets
x_values =  weather_data.drop(['rain_tomorrow_b'], axis = 1)
y_values = weather_data['rain_tomorrow_b']

#Create Training & Testing Data Sets
x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, random_state = 42)

#Get Scalar Value for X Training Data
x_scaler = StandardScaler().fit(x_train)

#Define Path for USA Fast Food Route
@app.route('/predict/<min_temp>/<max_temp>/<rainfall>/<evaporation>/<sunshine>/<wind_gust_speed>/<wind_speed_9>/<wind_speed_3>/<humidity_9>/<humidity_3>/<pressure_9>/<pressure_3>/<cloud_9>/<cloud_3>/<temp_9>/<temp_3>/<rain_today_b>/<wind_gust_dir>/<wind_dir_9>/<wind_dir_3>', methods = ['GET'])

#Define Function for Dashboard Content
def weather_predict(min_temp, max_temp, rainfall, evaporation, sunshine, wind_gust_speed, wind_speed_9, wind_speed_3, humidity_9, humidity_3, pressure_9, pressure_3, cloud_9, cloud_3, temp_9, temp_3, rain_today_b, wind_gust_dir, wind_dir_9, wind_dir_3):

    #Create Blank Lists for Wind Direction Variables
    gust_dir = list(repeat(0, 16))
    wind_9_dir = list(repeat(0, 16))
    wind_3_dir = list(repeat(0, 16))

    #Overwrite Appropriate List Value for Inputted Wind Gust Direction
    if wind_gust_dir.upper() == 'E':
        gust_dir[0] = 1
    elif wind_gust_dir.upper() == 'ENE':
        gust_dir[1] = 1
    elif wind_gust_dir.upper() == 'ESE':
        gust_dir[2] = 1
    elif wind_gust_dir.upper() == 'N':
        gust_dir[3] = 1
    elif wind_gust_dir.upper() == 'NE':
        gust_dir[4] = 1
    elif wind_gust_dir.upper() == 'NNE':
        gust_dir[5] = 1
    elif wind_gust_dir.upper() == 'NNW':
        gust_dir[6] = 1
    elif wind_gust_dir.upper() == 'NW':
        gust_dir[7] = 1
    elif wind_gust_dir.upper() == 'S':
        gust_dir[8] = 1
    elif wind_gust_dir.upper() == 'SE':
        gust_dir[9] = 1
    elif wind_gust_dir.upper() == 'SSE':
        gust_dir[10] = 1
    elif wind_gust_dir.upper() == 'SSW':
        gust_dir[11] = 1
    elif wind_gust_dir.upper() == 'SW':
        gust_dir[12] = 1
    elif wind_gust_dir.upper() == 'W':
        gust_dir[13] = 1
    elif wind_gust_dir.upper() == 'WNW':
        gust_dir[14] = 1
    elif wind_gust_dir.upper() == 'WSW':
        gust_dir[15] = 1

    #Overwrite Appropriate List Value for Inputted 9 AM Wind Direction
    if wind_dir_9.upper() == 'E':
        wind_9_dir[0] = 1
    elif wind_dir_9.upper() == 'ENE':
        wind_9_dir[1] = 1
    elif wind_dir_9.upper() == 'ESE':
        wind_9_dir[2] = 1
    elif wind_dir_9.upper() == 'N':
        wind_9_dir[3] = 1
    elif wind_dir_9.upper() == 'NE':
        wind_9_dir[4] = 1
    elif wind_dir_9.upper() == 'NNE':
        wind_9_dir[5] = 1
    elif wind_dir_9.upper() == 'NNW':
        wind_9_dir[6] = 1
    elif wind_dir_9.upper() == 'NW':
        wind_9_dir[7] = 1
    elif wind_dir_9.upper() == 'S':
        wind_9_dir[8] = 1
    elif wind_dir_9.upper() == 'SE':
        wind_9_dir[9] = 1
    elif wind_dir_9.upper() == 'SSE':
        wind_9_dir[10] = 1
    elif wind_dir_9.upper() == 'SSW':
        wind_9_dir[11] = 1
    elif wind_dir_9.upper() == 'SW':
        wind_9_dir[12] = 1
    elif wind_dir_9.upper() == 'W':
        wind_9_dir[13] = 1
    elif wind_dir_9.upper() == 'WNW':
        wind_9_dir[14] = 1
    elif wind_dir_9.upper() == 'WSW':
        wind_9_dir[15] = 1

    #Overwrite Appropriate List Value for Inputted 3 PMWind Direction
    if wind_dir_3.upper() == 'E':
        wind_3_dir[0] = 1
    elif wind_dir_3.upper() == 'ENE':
        wind_3_dir[1] = 1
    elif wind_dir_3.upper() == 'ESE':
        wind_3_dir[2] = 1
    elif wind_dir_3.upper() == 'N':
        wind_3_dir[3] = 1
    elif wind_dir_3.upper() == 'NE':
        wind_3_dir[4] = 1
    elif wind_dir_3.upper() == 'NNE':
        wind_3_dir[5] = 1
    elif wind_dir_3.upper() == 'NNW':
        wind_3_dir[6] = 1
    elif wind_dir_3.upper() == 'NW':
        wind_3_dir[7] = 1
    elif wind_dir_3.upper() == 'S':
        wind_3_dir[8] = 1
    elif wind_dir_3.upper() == 'SE':
        wind_3_dir[9] = 1
    elif wind_dir_3.upper() == 'SSE':
        wind_3_dir[10] = 1
    elif wind_dir_3.upper() == 'SSW':
        wind_3_dir[11] = 1
    elif wind_dir_3.upper() == 'SW':
        wind_3_dir[12] = 1
    elif wind_dir_3.upper() == 'W':
        wind_3_dir[13] = 1
    elif wind_dir_3.upper() == 'WNW':
        wind_3_dir[14] = 1
    elif wind_dir_3.upper() == 'WSW':
        wind_3_dir[15] = 1

    #Create Lists of Input Values for Machine Learning Models
    mvlr_input = [[]]
    knn_input = [[]]
    rf_input = [[]]
    svm_input = [[]]

    #Create Dictioary of Machine Learning Model Predictions
    predictions = {'MVLR': mvlr, 'KNN': knn, 'RF': rf, 'SVM': svm}

    #Create JSON Object from Prediction Dictionary
    rain_prediction = jsonify(predictions)
    
    #Call HTML File & Pass in Dictionary with JSON Object
    return rain_prediction

#Initialize Flask App
app.run()