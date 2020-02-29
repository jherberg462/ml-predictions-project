#Import Modules
import flask
from flask import request, jsonify, render_template, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

#Define Flask App Environment
app = flask.Flask(__name__)
app.config["DEBUG"] = True

#Define Path for USA Fast Food Route
@app.route('/predict/<min_temp>/<max_temp>/<rainfall>/<evaporation>/<sunshine>/<wind_gust_speed>/<wind_speed_9>/<wind_speed_3>/<humidity_9>/<humidity_3>/<pressure_9>/<pressure_3>/<cloud_9>/<cloud_3>/<temp_9>/<temp_3>/<rain_today_b>/<wind_gust_dir>/<wind_dir_9>/<wind_dir_3>', methods = ['GET'])

#Define Function for Dashboard Content
def weather_predict(min_temp, max_temp, rainfall, evaporation, sunshine, wind_gust_speed, wind_speed_9, wind_speed_3, humidity_9, humidity_3, pressure_9, pressure_3, cloud_9, cloud_3, temp_9, temp_3, rain_today_b, wind_gust_dir, wind_dir_9, wind_dir_3):

    #Import Baseline Weather Data
    weather_data = pd.read_csv('C:/Users/mjknj/Desktop/UNCC/Projects/Final Project/ml-predictions-project/aus_weather/weatherAUS_feature_engineer.csv')

    #Split Weather Data into X & Y Sets
    x_values_1 =  weather_data.drop(['rain_tomorrow_b'], axis = 1)
    y_values_1 = weather_data['rain_tomorrow_b']

    x_values_2 =  weather_data.drop(['rain_tomorrow_b', 'wind_change_direction', 'wind_gust_change_3', 'wind_gust_change_9'], axis = 1)
    y_values_2 = weather_data['rain_tomorrow_b']

    #Create Training & Testing Data Sets
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_values_1, y_values_1, random_state = 42, train_size = 0.8)
    x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_values_2, y_values_2, random_state = 42, train_size = 0.8)

    #Get Scalar Value for X Training Data
    x_scalar_1 = StandardScaler().fit(x_train_1)
    x_scalar_2 = StandardScaler().fit(x_train_2)

    #Determine Whether Wind Direction Changed
    if wind_dir_9 == wind_dir_3:
        wind_change_dir = 1
    else:
        wind_change_dir = 0

    if wind_dir_9 == wind_gust_dir:
        wind_gust_change_9 = 0
    else:
        wind_gust_change_9 = 1

    if wind_dir_3 == wind_gust_dir:
        wind_gust_change_3 = 0
    else:
        wind_gust_change_3 = 1

    if rain_today_b == True:
        rain_today = 1
    else:
        rain_today = 0

    #Calculate Temperature/Humidity/Pressure Changes
    temp_change_9to3 = abs(float(temp_9) - float(temp_3))
    temp_change_min_max = abs(float(min_temp) - float(max_temp))
    humidity_change = float(humidity_9) - float(humidity_3)
    pressure_change = float(pressure_9) - float(pressure_3)

    #Calculate Humidity Change Percentage
    if float(humidity_9) > 0:
        humidity_change_percent = humidity_change / int(humidity_9)
    else:
        humidity_change_percent = int(weather_data['Humidity9am'].max())

    #Create Lists of Input Values for Machine Learning Models
    mvlr_knn_svm_input = [[float(min_temp), float(max_temp), float(rainfall), float(evaporation), float(sunshine), float(wind_gust_speed), float(wind_speed_9), float(wind_speed_3), float(humidity_9), float(humidity_3), float(pressure_9), float(pressure_3), float(cloud_9), float(cloud_3), float(temp_9), float(temp_3), rain_today, temp_change_9to3, temp_change_min_max, humidity_change, humidity_change_percent, pressure_change, wind_change_dir, wind_gust_change_3, wind_gust_change_9]]
    rf_input = [[float(min_temp), float(max_temp), float(rainfall), float(evaporation), float(sunshine), float(wind_gust_speed), float(wind_speed_9), float(wind_speed_3), float(humidity_9), float(humidity_3), float(pressure_9), float(pressure_3), float(cloud_9), float(cloud_3), float(temp_9), float(temp_3), rain_today, temp_change_9to3, temp_change_min_max, humidity_change, humidity_change_percent, pressure_change]]

    #Transform Input Lists with Training Data Scalar Value
    mvlr_knn_svm_transformed = x_scalar_1.transform(mvlr_knn_svm_input)
    rf_transformed = x_scalar_2.transform(rf_input)

    #Import Machine Learning Models
    mvlr_model = joblib.load('logistic.sav')
    knn_model = joblib.load('knn.sav')
    rf_model = joblib.load('random_forest_engineered.sav')
    svm_model = joblib.load('svm.sav')

    #Generate Weather Predictions from Machine Learning Models
    mvlr_predict = mvlr_model.predict(mvlr_knn_svm_transformed)
    knn_predict = knn_model.predict(mvlr_knn_svm_transformed)
    rf_predict = rf_model.predict(rf_transformed)
    svm_predict = svm_model.predict(mvlr_knn_svm_transformed)

    #Convert Weather Predictions to Boolean Values
    if mvlr_predict == 1:
        mvlr = True
    else:
        mvlr = False

    if knn_predict == 1:
        knn = True
    else:
        knn = False

    if rf_predict == 1:
        rf = True
    else:
        rf = False

    if svm_predict == 1:
        svm = True
    else:
        svm = False

    #Determine Aggregate Weather Prediction
    if mvlr == True and knn == True and rf == True and svm == True:
        agg = True
    elif mvlr == False and knn == True and rf == True and svm == True:
        agg = True
    elif mvlr == True and knn == False and rf == True and svm == True:
        agg = True
    elif mvlr == True and knn == True and rf == False and svm == True:
        agg = True
    elif mvlr == True and knn == True and rf == True and svm == False:
        agg = True
    elif mvlr == True and knn == True and rf == False and svm == False:
        agg = False
    elif mvlr == True and knn == False and rf == True and svm == False:
        agg = True
    elif mvlr == True and knn == False and rf == False and svm == True:
        agg = False
    elif mvlr == False and knn == True and rf == True and svm == False:
        agg = True
    elif mvlr == False and knn == True and rf == False and svm == True:
        agg = False
    elif mvlr == False and knn == False and rf == True and svm == True:
        agg = True
    elif mvlr == True and knn == False and rf == False and svm == False:
        agg = False
    elif mvlr == False and knn == True and rf == False and svm == False:
        agg = False
    elif mvlr == False and knn == False and rf == True and svm == False:
        agg = False
    elif mvlr == False and knn == False and rf == False and svm == True:
        agg = False
    elif mvlr == False and knn == False and rf == False and svm == False:
        agg = False

    #Create Dictioary of Machine Learning Model Predictions
    predictions = {'MVLR': mvlr, 'KNN': knn, 'RF': rf, 'SVM': svm, 'AGG': agg}

    #Create JSON Object from Prediction Dictionary
    rain_prediction = jsonify(predictions)
    
    #Call HTML File & Pass in Dictionary with JSON Object
    return rain_prediction

#Initialize Flask App
app.run()