import os
import joblib
import pandas as pd
import streamlit as st

# 1. Configuracion de la interfaz
st.set_page_config(page_title="Evaluacion ISIL - Titanic", layout="centered")

# Datos obligatorios del alumno en la barra lateral
st.sidebar.markdown("### Datos del Estudiante")
st.sidebar.write("**Nombre:** Edwin Caballero")
st.sidebar.write("**Codigo ISIL:** 76621458")

st.sidebar.markdown(
    "[https://colab.research.google.com/drive/1dggu6jZu87fE331UpIdimgdhZMhX7g4Z?usp=sharing)"
)

st.title("Aplicativo de Prediccion: Titanic")
st.write(
    "Modifica los parametros para evaluar las posibilidades de supervivencia utilizando modelos ya entrenados."
)


# 2. Cargar modelos .pkl guardados de la carpeta 'modelos'
@st.cache_resource
def cargar_modelos():
    ruta_rf = os.path.join("modelos", "modelo_rf.pkl")
    ruta_lr = os.path.join("modelos", "modelo_lr.pkl")

    mod_rf = joblib.load(ruta_rf)
    mod_lr = joblib.load(ruta_lr)
    return mod_rf, mod_lr


try:
    modelo_rf, modelo_lr = cargar_modelos()
    st.success(
        "Modelos de Machine Learning cargados correctamente desde la carpeta integrada."
    )
except Exception as e:
    st.error(
        "Error al cargar los modelos. Asegurate de haber subido la carpeta 'modelos' con los archivos .pkl a GitHub."
    )
    st.stop()

# 3. Seleccion de Modelo por parte del usuario
st.subheader("Configuracion del Algoritmo")
opcion_modelo = st.selectbox(
    "Selecciona el modelo de Machine Learning a usar:",
    ["Random Forest Classifier", "Logistic Regression"],
)

# Seleccionar el objeto del modelo correcto
modelo_activo = modelo_rf if opcion_modelo == "Random Forest Classifier" else modelo_lr

# 4. Formulario interactivo en Espanol
st.subheader("Caracteristicas del Pasajero Ficticio")
col1, col2 = st.columns(2)

with col1:
    genero = st.radio("Genero:", ["Mujer", "Hombre"])
    edad = st.slider("Edad (en anos):", 1, 80, 30)
    clase = st.selectbox(
        "Clase de Boleto (Socioeconomico):",
        [1, 2, 3],
        format_func=lambda x: f"{x} Clase",
    )


with col2:
    tarifa = st.slider("Precio del Boleto (Tarifa en libras):", 0, 500, 32)
    esposo_hermanos = st.number_input(
        "Hermanos o Conyuges a bordo:", min_value=0, max_value=8, value=0
    )
    padres_hijos = st.number_input(
        "Padres o Hijos a bordo:", min_value=0, max_value=6, value=0
    )

# Mapear entrada a formato numerico que entiende el modelo
sexo_numerico = 1 if genero == "Hombre" else 0

# 5. Prediccion
st.markdown("---")
if st.button("Ejecutar Prediccion de Supervivencia", use_container_width=True):
    # Generar DataFrame con la estructura identica a la que fue entrenada
    pasajero = pd.DataFrame(
        [[clase, sexo_numerico, edad, esposo_hermanos, padres_hijos, tarifa]],
        columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"],
    )

    prediccion = modelo_activo.predict(pasajero)
    probabilidad = modelo_activo.predict_proba(pasajero)

    # Extraer probabilidad de la clase positiva (sobrevivir)
    porcentaje = probabilidad * 100

    if prediccion == 1:
        st.success(
            f"El pasajero sobrevive. Segun el modelo {opcion_modelo}, tiene un {porcentaje:.1f}% de probabilidad de salvarse."
        )
    else:
        st.error(
            f"El pasajero no sobrevive. Segun el modelo {opcion_modelo}, tiene un {porcentaje:.1f}% de probabilidad de fallecer."
        )
