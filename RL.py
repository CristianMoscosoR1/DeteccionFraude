import matplotlib
matplotlib.use('Agg')
import json
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os

def cargaDatos():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ruta1_json = os.path.join(base_dir, "datos_ent.json")
    ruta2_json = os.path.join(base_dir, "datos_test.json")

    if not os.path.exists(ruta1_json):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta1_json}")
    if not os.path.exists(ruta2_json):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta2_json}")

    with open(ruta1_json, "r", encoding="utf-8") as file:
        df_train = pd.DataFrame(json.load(file))

    with open(ruta2_json, "r", encoding="utf-8") as file:
        df_test = pd.DataFrame(json.load(file))

    df = pd.concat([df_train, df_test], ignore_index=True)
    return df

def regresion_logisitica():
    df = cargaDatos()

    X = df[['Edad', 'Tiempo_Permanencia', 'Dispositivo']]
    y = df['Click']

    X = pd.get_dummies(X, columns=['Dispositivo'], drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    result = {
        'accuracy': round(acc * 100, 2),
        'precision': round(prec * 100, 2),
        'recall': round(rec * 100, 2)
    }

    return result, plot_url

def prediccion(Edad, Tiempo_Permanencia, Dispositivo):
    df = cargaDatos()

    X = df[['Edad', 'Tiempo_Permanencia', 'Dispositivo']]
    X = pd.get_dummies(X, columns=['Dispositivo'], drop_first=True)

    dispositivo_col = 'Dispositivo_2'
    input_dict = {
        'Edad': Edad,
        'Tiempo_Permanencia': Tiempo_Permanencia,
        dispositivo_col: 1 if Dispositivo == 2 else 0
    }
    X_input = pd.DataFrame([input_dict])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_input_scaled = scaler.transform(X_input)

    model = LogisticRegression()
    model.fit(X_scaled, df['Click'])

    prediction = model.predict(X_input_scaled)[0]
    return prediction