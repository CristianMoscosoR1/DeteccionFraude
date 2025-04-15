import re
from flask import Flask, render_template, request
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from linealRegression601N import calculateConsumption
from RL import regresion_logisitica, prediccion
from Clasificacion import Bclasi, clasib, entreno
from flask import send_file
import os
import joblib
from sklearn.metrics import classification_report

app = Flask(__name__)

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

@app.route('/prestamos_inicio', methods=['GET', 'POST'])
def clasificacion():
    if request.method == 'POST':
        archivo = request.files.get('archivo')

        if not archivo:
            return "No se subió ningún archivo"

        try:
            temp_path = "temp_upload.xlsx"
            archivo.save(temp_path)
            
            resultado, error = Bclasi(temp_path)
            
            os.remove(temp_path)
            
            if error:
                return error
            
            modelo, result, skaler = entreno()
            
            resultado.to_excel("ResultadoClasificacion.xlsx", index=False)
            return render_template('resultados.html', result=result, tablas=[resultado.to_html(classes='data', index=False)])
            
        except Exception as e:
            return f"Ocurrió un error: {str(e)}"

    return render_template('prestamos_inicio.html')

@app.route('/descargar_resultados')
def descargar_resultados():
    ruta_resultado = os.path.join(os.path.dirname(__file__), 'ResultadoClasificacion.xlsx')

    if os.path.exists(ruta_resultado):
        return send_file(ruta_resultado, as_attachment=True)
    else:
        return "El archivo de resultados no existe. Primero clasifica un archivo Excel."

if __name__ == "__main__":
    app.run(debug=True)