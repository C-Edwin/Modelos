import pandas as pd
import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. Configuración de la página
st.set_page_config(page_title="Predicción de Diabetes", layout="wide")
st.title("🩺 Modelo de Predicción de Diabetes")
st.write("Aplicativo estudiantil usando el dataset clásico de Scikit-Learn.")

# 2. Carga de datos (Simulando nuestra base de datos)
@st.cache_data  # Optimiza la app para que no recargue los datos siempre
def cargar_datos():
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df["progression"] = diabetes.target
    return df, diabetes.feature_names


df, features = cargar_datos()

# Mostrar datos en la app
st.subheader("📊 Vista previa de los datos libres")
st.dataframe(df.head())

# 3. Entrenar el modelo (Caja negra de Ciencia de Datos)
X = df[features]
y = df["progression"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 4. Formulario interactivo para el usuario
st.subheader("🔮 Realizar una nueva predicción")
st.write("Modifica los valores del paciente para ver la progresión simulada:")

col1, col2 = st.columns(2)
inputs = {}

# Generar sliders automáticos para cada variable (Edad, IMC, Presión, etc.)
for i, feature in enumerate(features):
    with col1 if i % 2 == 0 else col2:
        inputs[feature] = st.slider(
            f"Valor de {feature}",
            float(df[feature].min()),
            float(df[feature].max()),
            float(df[feature].mean()),
        )

# Botón para predecir
if st.button("Calcular Progresión"):
    input_df = pd.DataFrame([inputs])
    prediccion = model.predict(input_df)[0]
    st.success(f"📈 El índice de progresión estimado es: {prediccion:.2f}")
