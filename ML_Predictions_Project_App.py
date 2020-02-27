#Import Modules
import flask
from flask import request, jsonify, render_template, redirect, url_for
import numpy as np
import pandas as pd
import datetime as dt
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
from statistics import mean
import json

#Define Flask App Environment
app = flask.Flask(__name__)
app.config["DEBUG"] = True

#Define Path for USA Fast Food Route
@app.route('/', methods = ['GET'])

#Define Function for Dashboard Content
def weather_predict():

    #Import USA Fast Food Data Table from SQL DB as Pandas Data Frame
    USA_food = pd.read_sql('SELECT * FROM "USA_Fast_Food"', engine)

    #Convert Data Frame to Dictionary & Convert Dictionary to JSON Object
    full_data = json.dumps(USA_food.to_dict(orient='records'), indent = 2)

    #Add JSON Object to Dictionary
    data = {'chart_data': full_data}
    
    #Call HTML File & Pass in Dictionary with JSON Object
    return render_template("index_data.html", data = data)

#Initialize Flask App
app.run()