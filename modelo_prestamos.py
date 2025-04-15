import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import json

# Cargar datos desde un archivo JSON con nuevo nombre
with open('datos_eclas.json', 'r') as f:
    datos = json.load(f)

# Crear el DataFrame desde los datos del JSON
df = pd.DataFrame(datos)

# Separar variables independientes (X) y variable objetivo (y)
X = df[['historial_crediticio', 'ingreso', 'deuda_actual']]
y = df['prestamo_aprobado']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Realizar predicciones
y_pred = modelo.predict(X_test)

# Evaluar el modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

# Función para predicción individual
def predecir_prestamo(historial_crediticio, ingreso, deuda_actual):
    entrada = [[historial_crediticio, ingreso, deuda_actual]]
    prediccion = modelo.predict(entrada)
    return "Préstamo aprobado" if prediccion[0] == 1 else "Préstamo no aprobado"
