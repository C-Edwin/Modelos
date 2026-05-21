import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Configuración de la interfaz
st.set_page_config(page_title="Sobrevivientes del Titanic", layout="centered")
st.title("🚢 Predicción de Supervivencia: Titanic")
st.write("Modifica las características del pasajero para ver si habría sobrevivido al naufragio.")

# 2. Carga de datos limpios desde un GitHub público
@st.cache_data
def cargar_datos():
    url = "https://githubusercontent.com"
    df = pd.read_csv(url)
    columnas = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    df = df[columnas].dropna()
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
    return df

df = cargar_datos()

# 3. Entrenamiento del Modelo (Random Forest)
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = RandomForestClassifier(random_state=42, n_estimators=100)
modelo.fit(X_train, y_train)

# 4. Formulario interactivo en Español
st.subheader("👤 Datos del Pasajero Ficticio")

col1, col2 = st.columns(2)

with col1:
    genero = st.radio("Género:", ["Mujer", "Hombre"])
    edad = st.slider("Edad (en años):", 1, 80, 30)
    # CORREGIDO: Añadida la lista [1, 2, 3] y eliminada la doble coma
    clase = st.selectbox("Clase de Boleto:", [1, 2, 3], format_func=lambda x: f"Clase {x}")

with col2:
    tarifa = st.slider("Precio del Boleto (Tarifa):", 0, 500, 32)
    esposo_hermanos = st.number_input("Hermanos o Cónyuges a bordo:", min_value=0, max_value=8, value=0)
    padres_hijos = st.number_input("Padres o Hijos a bordo:", min_value=0, max_value=6, value=0)

sexo_numerico = 1 if genero == "Hombre" else 0

# 5. Botón de Predicción y Resultado Visual
st.markdown("---")
if st.button("🔮 Evaluar Supervivencia", use_container_width=True):
    pasajero = pd.DataFrame([[clase, sexo_numerico, edad, esposo_hermanos, padres_hijos, tarifa]], 
                            columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])
    
    prediccion = modelo.predict(pasajero)
    probabilidad = modelo.predict_proba(pasajero) * 100
    
    if prediccion == 1:
        st.success(f"🟢 **¡Sobrevive!** El pasajero tiene un **{probabilidad[0][1]:.1f}%** de probabilidad de salvarse.")
        st.balloons()
    else:
        st.error(f"🔴 **No sobrevive.** El pasajero tiene solo un **{probabilidad[0][1]:.1f}%** de probabilidad de salvarse.")

