import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


data = {
    "Temperature": [
        61.138585, 14.828688, 27.244501, 50.264025, 58.358275,
        19.332321, 37.081978, 30.893391, 56.702328, 42.939019
    ],
    "Consumption": [
        248.700508, 88.901216, 130.274534, 213.060802, 240.684679,
        107.498232, 168.305477, 140.225682, 230.546492, 183.642091
    ]
}

df = pd.DataFrame(data)


X = df[["Temperature"]]
Y = df["Consumption"]


model = LinearRegression()
model.fit(X, Y)

def calculateConsumption(temperature):
    result = model.predict([[temperature]])[0]
    return round(result, 2)