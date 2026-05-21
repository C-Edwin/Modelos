import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Configuración de la interfaz
st.set_page_config(page_title="Sobrevivientes del Titanic", layout="centered")
st.title("🚢 Predicción de Supervivencia: Titanic")
st.write("Modifica las características del pasajero para ver si habría sobrevivido al naufragio.")

# 2. Carga de datos local/segura (Sin usar URL de internet)
@st.cache_data
def cargar_datos():
    # Descarga el dataset oficial del Titanic desde la biblioteca OpenML de sklearn
    titanic = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
    df = titanic.frame
    
    # Renombrar columnas clave al formato estándar
    df = df.rename(columns={'survived': 'Survived', 'pclass': 'Pclass', 'sex': 'Sex', 'age': 'Age', 'sibsp': 'SibSp', 'parch': 'Parch', 'fare': 'Fare'})
    
    # Filtrar solo columnas numéricas e intuitivas y limpiar nulos
    columnas = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    df = df[columnas].dropna()
    
    # Convertir categorías a tipos numéricos limpios para el modelo
    df["Survived"] = df["Survived"].astype(int)
    df["Pclass"] = df["Pclass"].astype(int)
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0}).astype(int)
    df["Age"] = df["Age"].astype(float)
    df["Fare"] = df["Fare"].astype(float)
    
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
    
    prediccion = modelo.predict(pasajero)[0]
    probabilidad = modelo.predict_proba(pasajero)[0]
    
    # Extraer porcentaje según la clase predicha
    porcentaje = probabilidad[1] * 100 if prediccion == 1 else probabilidad[0] * 100
    
    if prediccion == 1:
        st.success(f"🟢 **¡Sobrevive!** El pasajero tiene un **{porcentaje:.1f}%** de probabilidad de salvarse.")
        st.balloons()
    else:
        st.error(f"🔴 **No sobrevive.** El pasajero tiene un **{porcentaje:.1f}%** de probabilidad de fallecer.")

