import json
from pathlib import Path
import joblib
import numpy as np
import streamlit as st
import keras

st.set_page_config(page_title="Semana 13 | RNA Diabetes", page_icon="🧠", layout="wide")
BASE = Path(__file__).resolve().parent

@st.cache_resource
def cargar_modelo():
    model = keras.models.load_model(str(BASE / "modelo_rna.keras"))
    scaler = joblib.load(BASE / "scaler.pkl")
    meta = json.loads((BASE / "metadata_modelo.json").read_text(encoding="utf-8"))
    return model, scaler, meta

st.title("Clasificación orientativa de registros de diabetes · RNA")
st.caption("Proyecto académico USIL — Semana 13 — Grupo 11")
st.warning("Demostración educativa. NO es una prueba diagnóstica, no estima riesgo clínico validado ni sustituye evaluación médica.")

try:
    model, scaler, meta = cargar_modelo()
except Exception as exc:
    st.error(f"Faltan archivos del modelo o no pueden abrirse: {exc}")
    st.stop()

FEATURES = meta["features"]
LABELS = {
"Poliuria":"Poliuria (orinar frecuentemente)",
"Polidipsia":"Polidipsia (sed excesiva)",
"Perdida_repentina_peso":"Pérdida repentina de peso",
"Debilidad":"Debilidad", "Polifagia":"Polifagia (aumento de apetito)",
"Candidiasis_genital":"Candidiasis genital", "Vision_borrosa":"Visión borrosa",
"Picazon":"Picazón", "Irritabilidad":"Irritabilidad",
"Cicatrizacion_lenta":"Cicatrización lenta", "Paresia_parcial":"Paresia parcial",
"Rigidez_muscular":"Rigidez muscular", "Alopecia":"Alopecia", "Obesidad":"Obesidad"}

with st.form("paciente"):
    a,b,c=st.columns(3)
    with a:
        edad=st.number_input("Edad", min_value=16, max_value=90, value=45, step=1)
        genero=st.selectbox("Género (según codificación del dataset)", ["Masculino (1)","Femenino (2)"])
    valores={"Edad":float(edad),"Genero":1.0 if genero.startswith("Masculino") else 2.0}
    sintomas=[f for f in FEATURES if f not in ("Edad","Genero")]
    cols=[a,b,c]
    for i, f in enumerate(sintomas):
        with cols[(i+2)%3]:
            valores[f]=float(st.checkbox(LABELS.get(f,f),key=f))
    enviar=st.form_submit_button("Clasificar registro", use_container_width=True)

if enviar:
    muestra=np.array([[valores[f] for f in FEATURES]],dtype=np.float32)
    muestra_sc=scaler.transform(muestra)
    prob=float(model.predict(muestra_sc,verbose=0).reshape(-1)[0])
    clase= "Positivo" if prob >= 0.5 else "Negativo"
    st.metric("Predicción del modelo (umbral 0.5)",clase)
    st.write(f"Puntuación estimada por el modelo para la clase Positivo: **{prob:.1%}**")
    st.progress(float(np.clip(prob,0,1)))
    st.info("La salida del modelo refleja patrones de la muestra de entrenamiento; no es una probabilidad médica calibrada ni un diagnóstico.")
    with st.expander("Datos de entrada utilizados"):
        st.dataframe({"Característica":FEATURES,"Valor":[valores[f] for f in FEATURES]},hide_index=True)

st.divider()
st.caption("Modelo académico entrenado sobre 520 registros históricos, sin validación clínica externa. Interpretar junto al informe de evaluación.")
