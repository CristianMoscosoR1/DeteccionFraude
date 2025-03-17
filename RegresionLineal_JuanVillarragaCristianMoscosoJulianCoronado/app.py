import re
from flask import Flask, render_template, request
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression
from linealRegression601N import calculateConsumption
import joblib

app = Flask (_name_)

@app.route("/")
def home():
    return "Un diablo se callo al agua y otro diablo lo saco" 

@app.route("/hello/<name>")
def hello_there(name):
    now = datetime.now()

    match_object = re.match(" [a-zA-Z] +",name)

    if match_object:
        clean_name = match_object.group(0)
    else: 
     clean_name = " friend"

    
    content = "hello there, " + clean_name + "! hour: " + str(now)
    return content   

@app.route("/indexHTML")
def indexHTML():
    return render_template("index.html")



@app.route("/Temperatura", methods=["GET", "POST"])
def predict_consumption():
    predict_result = None
    if request.method == "POST":
        temperature = float(request.form["temperature"])
        predict_result = calculateConsumption(temperature)

    return render_template("linearRegression.html", result=predict_result)