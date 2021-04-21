# Salvamos el modelo
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import pickle
import pandas as pd


# Cargar el data set de boston
boston = datasets.load_boston()

# Separación de los datos para predicir
X = pd.DataFrame(boston.data, columns=boston.feature_names)
print(X.head())
Y = pd.DataFrame(boston.target, columns=["MEDV"])
print(Y.head())

model = RandomForestRegressor()
model.fit(X, Y)

pickle.dump(model, open('boston_model.pkl', 'wb'))