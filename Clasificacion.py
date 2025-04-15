import json
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import pickle 

def cargaDatos():
    ruta1_json = os.path.join(os.path.dirname(__file__), "DatosRL/datos_eclas.json")
    ruta2_json = os.path.join(os.path.dirname(__file__), "DatosRL/datos_tclas.json")

    with open(ruta1_json, "r", encoding="utf-8") as file:
        data = json.load(file)
    df_train = pd.DataFrame(data)

    with open(ruta2_json, "r", encoding="utf-8") as file:
        data = json.load(file)
    df_test = pd.DataFrame(data)

    return pd.concat([df_train, df_test], ignore_index=True)

def entreno():
        df_entrenamiento = cargaDatos()
        
        X = df_entrenamiento[['historial_crediticio', 'ingreso', 'deuda_actual']]
        y = df_entrenamiento['prestamo_aprobado']
        
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_res)
        
        modelo = BaggingClassifier(
            estimator=LogisticRegression(class_weight='balanced', C=0.1, solver='liblinear'),
            n_estimators=50,
            max_samples=0.8,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_res, test_size=0.2, random_state=42
        )
        modelo.fit(X_train, y_train)
        
        
        y_pred = modelo.predict(X_test)

        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        
        
        conf_mat = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
        plt.title("Matriz de Confusión")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        
        # Convertir a imagen base64
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        with open(f"pepinillo.pkl", "wb") as f:
            pickle.dump(modelo,f)
        
        result = {
            'accuracy': round(acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'plot_url': plot_url
        }
        return modelo, result, scaler

def Bclasi(file_path):
    try:
        
        modelo, result, scaler = entreno()
        df_nuevo = pd.read_excel(file_path)
        df_nuevo.columns = df_nuevo.columns.str.strip().str.lower().str.replace(' ', '_')
        
        required = ['historial_crediticio', 'ingreso', 'deuda_actual']
        if not all(col in df_nuevo.columns for col in required):
            missing = [col for col in required if col not in df_nuevo.columns]
            return None, f"Faltan columnas requeridas: {', '.join(missing)}"
        
        X_nuevo = df_nuevo[required]
        X_nuevo_scaled = scaler.transform(X_nuevo)
        
        # Realizar predicciones
        df_nuevo['prediccion'] = modelo.predict(X_nuevo_scaled)
        df_nuevo['probabilidad_aprobacion'] = modelo.predict_proba(X_nuevo_scaled)[:, 1]
        

        return df_nuevo, None
        
    except Exception as e:
        return None, {'error': f"Error al procesar: {str(e)}"}

def clasib(historial_crediticio, ingreso, deuda_actual):
    df = cargaDatos()
    required_columns = ['historial_crediticio', 'ingreso', 'deuda_actual']
    
    if not all(col in df.columns for col in required_columns):
        return "Error: Faltan columnas requeridas en los datos de entrenamiento"
    
    X = df[required_columns]
    y = df['prestamo_aprobado']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = BaggingClassifier(
        estimator=LogisticRegression(),
        n_estimators=10,
        random_state=42
    )
    model.fit(X_scaled, y)
    
    input_data = np.array([[historial_crediticio, ingreso, deuda_actual]])
    input_scaled = scaler.transform(input_data)
    
    return model.predict(input_scaled)[0]