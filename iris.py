import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Cargar conjunto de datos Iris
iris = load_iris()
modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo.fit(iris.data, iris.target)

# Título de la app
st.title("Clasificador de la especie Iris")

# Campos de entrada interactivos
sepal_length = st.number_input("Longitud del sépalo (cm)", 
                              min_value=4.0, max_value=8.0, step=0.1)
sepal_width = st.number_input("Ancho del sépalo (cm)", 
                             min_value=2.0, max_value=4.5, step=0.1)
petal_length = st.number_input("Longitud del pétalo (cm)", 
                              min_value=1.0, max_value=7.0, step=0.1)
petal_width = st.number_input("Ancho del pétalo (cm)", 
                             min_value=0.1, max_value=2.5, step=0.1)

# Botón de predicción
if st.button("Predecir especie"):
    muestra = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediccion = modelo.predict(muestra)
    especie = iris.target_names[prediccion[0]]
    st.success(f"La especie predicha es: **{especie}**")