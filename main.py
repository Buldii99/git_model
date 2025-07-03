# Plik: train_model.py
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

X = np.array([
    [25, 70, 180, 1, 0],
    [40, 80, 175, 1, 1],
    [30, 65, 165, 0, 0],
    [35, 90, 190, 1, 2],
    [50, 60, 160, 0, 1],
    [20, 55, 170, 0, 2],
])
y = np.array([2100, 2400, 1900, 3000, 1700, 2300])

model = LinearRegression()
model.fit(X, y)

with open("calorie_model.pkl", "wb") as f:
    pickle.dump(model, f)
