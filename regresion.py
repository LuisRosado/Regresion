import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Datos para la primera función
m = 100
X1 = 6 * np.random.rand(m, 1) - 3
y1 = 0.5 * X1 ** 2 + X1 + 2 + np.random.randn(m, 1)

# Datos para la segunda función
m = 50
X2 = np.linspace(0, 8, m).reshape(-1, 1)
y2 = np.sin(X2) * np.exp(X2 / 1.1)

# Definir los grados polinomiales a probar
grados = [1, 3, 6]

# Realizar la regresión para la primera función
plt.figure(figsize=(15, 5))

for i, grado in enumerate(grados):
    model = make_pipeline(PolynomialFeatures(degree=grado), LinearRegression())
    model.fit(X1, y1)
    y_pred = model.predict(X1)

    plt.subplot(1, len(grados), i + 1)
    plt.scatter(X1, y1)
    plt.plot(X1, y_pred, color='red')
    plt.title(f'Grado {grado}, MSE: {mean_squared_error(y1, y_pred):.2f}')

plt.tight_layout()
plt.show()

# Realizar la regresión para la segunda función
plt.figure(figsize=(15, 5))

for i, grado in enumerate(grados):
    model = make_pipeline(PolynomialFeatures(degree=grado), LinearRegression())
    model.fit(X2, y2)
    y_pred = model.predict(X2)

    plt.subplot(1, len(grados), i + 1)
    plt.scatter(X2, y2)
    plt.plot(X2, y_pred, color='red')
    plt.title(f'Grado {grado}, MSE: {mean_squared_error(y2, y_pred):.2f}')

plt.tight_layout()
plt.show()
