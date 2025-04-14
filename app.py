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

@app.route("/prestamos")
def prestamos_inicio():
    return render_template("prestamos_inicio.html")

@app.route("/prestamos/cargar", methods=["GET", "POST"])
def cargar_datos():
    global resultados_df
    if request.method == "POST":
        archivo = request.files["archivo"]
        if archivo:
            df = pd.read_excel(archivo)

            columnas_esperadas = ['historial_crediticio', 'ingreso', 'deuda_actual']
            if not all(col in df.columns for col in columnas_esperadas):
                return "El archivo no contiene las columnas esperadas."

            predicciones = modelo.predict(df[columnas_esperadas])
            df["Resultado_Prediccion"] = predicciones
            resultados_df = df

            os.makedirs("static/resultados", exist_ok=True)
            df.to_csv("static/resultados/resultados.csv", index=False)

            return redirect("/prestamos/resultados")

    return render_template("cargar_datos.html")


@app.route("/prestamos/resultados")
def mostrar_resultados():
    global resultados_df
    if not resultados_df.empty:
        tabla_html = resultados_df.to_html(classes="table", index=False)
        return render_template("resultados.html", tablas=[tabla_html])
    else:
        return render_template("resultados.html", tablas=None)

@app.route("/prestamos/exportar")
def exportar_csv():
    archivo = "static/resultados/resultados.csv"
    if os.path.exists(archivo):
        return send_file(archivo, as_attachment=True)
    else:
        return "No hay archivo para exportar."