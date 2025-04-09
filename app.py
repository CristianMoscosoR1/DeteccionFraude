import re
from flask import Flask, render_template, request
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from linealRegression601N import calculateConsumption
from RL import regresion_logisitica, prediccion



app = Flask (__name__)

@app.route("/")

def home():
    return render_template("mainpage.html")

@app.route("/hello/<name>")
def hello_there(name):
    now = datetime.now()

    match_object = re.match("[a-zA-Z\s]+", name)
    
    if match_object:
        clean_name = match_object.group(0)
    else: 
     clean_name = " friend"

    
    content = "hello there, " + clean_name + "! hour: " + str(now)
    return content   

@app.route("/indexHTML")
def indexHTML():
    return render_template("index.html")

@app.route("/mapaMental")
def mapa_mental():
    return render_template("mapaMental.html")

@app.route("/Temperatura", methods=["GET", "POST"])
def predict_consumption():
    predict_result = None
    if request.method == "POST":
        temperature = float(request.form["temperature"])
        predict_result = calculateConsumption(temperature)

    return render_template("linearRegression.html", result=predict_result)

@app.route('/regresionL', methods=['GET', 'POST'])
def regresionL():
    prediction = None
    result, plot_url = regresion_logisitica()

    if request.method == 'POST':
        Edad = float(request.form['Edad'])
        Tiempo_Permanencia = float(request.form['Tiempo_Permanencia'])
        Dispositivo = int(request.form['Dispositivo'])
        prediction = prediccion(Edad, Tiempo_Permanencia, Dispositivo)
    return render_template('regresionL.html', result=result, prediction=prediction, plot_url=plot_url)