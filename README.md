# Boston Housing Price Prediction
1. [Importar librerías ](#schema1)

<hr>

<a name="schema1"></a>

# 1. Importar librerías y cargar los datos
~~~python
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

boston = datasets.load_boston()
~~~
# 2. Título 
~~~python
st.write("""
# Boston House Price Prediction App

This app predicts the **Boston House Price**!
""")
st.write('---')
~~~

# 3 Separación de los datos para predicir
~~~python

X = pd.DataFrame(boston.data, columns=boston.feature_names)

Y = pd.DataFrame(boston.target, columns=["MEDV"])
~~~
# 4. Barra lateral
### Cabecera de la entrada de parámetros
He tenido que poner delante de cada datos `float` porque me daba error streamlit que no reoconicía los valores y al ponerle que eran decimales ya funciona perfecto
~~~Python

st.sidebar.header('Specify Input Parameters')

def user_input_features():
    CRIM = st.sidebar.slider('CRIM', float(X.CRIM.min()), float(X.CRIM.max()), float(X.CRIM.mean()))
    ZN = st.sidebar.slider('ZN', float(X.ZN.min()), float(X.ZN.max()), float(X.ZN.mean()))
    INDUS = st.sidebar.slider('INDUS', float(X.INDUS.min()), float(X.INDUS.max()),float( X.INDUS.mean()))
    CHAS = st.sidebar.slider('CHAS', float(X.CHAS.min()), float(X.CHAS.max()), float(X.CHAS.mean()))
    NOX = st.sidebar.slider('NOX', float(X.NOX.min()), float(X.NOX.max()), float(X.NOX.mean()))
    RM = st.sidebar.slider('RM', float(X.RM.min()), float(X.RM.max()), float(X.RM.mean()))
    AGE = st.sidebar.slider('AGE',float(X.AGE.min()), float(X.AGE.max()),float(X.AGE.mean()))
    DIS = st.sidebar.slider('DIS', float(X.DIS.min()), float(X.DIS.max()), float(X.DIS.mean()))
    RAD = st.sidebar.slider('RAD', float(X.RAD.min()), float(X.RAD.max()), float(X.RAD.mean()))
    TAX = st.sidebar.slider('TAX', float(X.TAX.min()), float(X.TAX.max()), float(X.TAX.mean()))
    PTRATIO = st.sidebar.slider('PTRATIO',float(X.PTRATIO.min()), float(X.PTRATIO.max()), float(X.PTRATIO.mean()))
    B = st.sidebar.slider('B', float(X.B.min()), float(X.B.max()), float(X.B.mean()))
    LSTAT = st.sidebar.slider('LSTAT', float(X.LSTAT.min()), float(X.LSTAT.max()), float(X.LSTAT.mean()))
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features
~~~

Devuelve un dataframe con los datos del ususario
~~~python
df = user_input_features()
~~~



# 5. Panel central

~~~python
st.header('Specified Input parameters')
st.write(df)
st.write('---')
~~~


# 6.Construir modelo y aplicarlo

Creamos un archivo `.pickle` para no tener que estar generando el modelo.
Solo lo tenemos que cargar.
~~~python

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
~~~
# 7. Aplicamos el modelo.
Primero leemos el archivo y luego hacemos la predicción

~~~python

# Leemos el archivo salvado con el modelo
load_model = pickle.load(open('boston_model.pkl', 'rb'))

# Aplicamos el modelor y obtenemos predicciones
prediction = load_model.predict(df)
st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')
~~~
# 8. Usando SHAP values

~~~Python

# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')

~~~