"""Streamlit dashboard for project results and API-backed image demo."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
API_URL = os.getenv("KNEE_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Artrosis de rodilla | ML1", page_icon="🦴", layout="wide")


@st.cache_data
def load_results() -> dict:
    with (REPORTS_DIR / "modelos_avanzados_metrics.json").open(encoding="utf-8") as file:
        return json.load(file)


results = load_results()
table = pd.DataFrame(results["comparison_table"]).sort_values("F1 Macro", ascending=False)
winner = table.iloc[0]

st.title("Clasificación de artrosis de rodilla")
st.caption("Proyecto académico · Machine Learning 1 · FIUNA")
st.warning("Demostración académica. No es una herramienta diagnóstica ni sustituye la evaluación profesional.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Radiografías", "1.650")
col2.metric("Grados", "5")
col3.metric("Mejor modelo", str(winner["Modelo"]))
col4.metric("F1 macro de test", f"{float(winner['F1 Macro']):.3f}")

tab_results, tab_figures, tab_demo, tab_method = st.tabs(
    ["Resultados", "Gráficos", "Probar una imagen", "Metodología"]
)

with tab_results:
    st.subheader("Comparación en el conjunto de prueba")
    metric = st.selectbox("Métrica del gráfico", ["F1 Macro", "Accuracy (%)", "Balanced Acc (%)"])
    fig = px.bar(
        table,
        x="Modelo",
        y=metric,
        color="Modelo",
        text=metric,
        title=f"Modelos comparados · {metric}",
    )
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=metric)
    fig.update_traces(texttemplate="%{text:.3f}" if metric == "F1 Macro" else "%{text:.1f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("La selección de hiperparámetros se hizo con validación; las métricas de esta tabla corresponden al conjunto de prueba (N=248).")

with tab_figures:
    st.subheader("Evaluación e interpretación")
    for filename, caption in [
        ("distribucion_clases.png", "Distribución de las cinco clases del conjunto de datos."),
        ("modelos_avanzados_confusion_matrices.png", "Matrices de confusión: aciertos y errores por grado."),
        ("modelos_avanzados_learning_curves.png", "Curvas de aprendizaje de SVM y XGBoost con validación cruzada estratificada."),
        ("modelos_avanzados_feature_importance.png", "Descriptores con mayor importancia según XGBoost."),
    ]:
        image_path = FIGURES_DIR / filename
        if image_path.exists():
            st.image(str(image_path), caption=caption, use_container_width=True)
        else:
            st.info(f"Todavía no está disponible el gráfico {filename}.")

with tab_demo:
    st.subheader("Clasificación demostrativa")
    st.write("Cargá una radiografía en escala de grises. La imagen se envía a la API local para mostrar la salida del modelo.")
    uploaded = st.file_uploader("Radiografía (PNG o JPG)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    if uploaded is not None:
        st.image(uploaded, caption="Imagen seleccionada", use_container_width=True)
        if st.button("Ejecutar predicción", type="primary"):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
                    timeout=120,
                )
                response.raise_for_status()
                prediction = response.json()
                st.success(f"Clase estimada por el modelo: {prediction['class_index']} · {prediction['class_name']}")
                st.metric("Confianza del modelo", f"{prediction['confidence']:.1%}")
                prob_df = pd.DataFrame(
                    {"Grado": list(prediction["probabilities"]), "Probabilidad": list(prediction["probabilities"].values())}
                )
                st.plotly_chart(px.bar(prob_df, x="Grado", y="Probabilidad", range_y=[0, 1]), use_container_width=True)
                st.caption(prediction["notice"])
            except requests.ConnectionError:
                st.error("No se pudo conectar con la API. Iniciá primero el servicio descrito en el README.")
            except requests.RequestException as error:
                st.error(f"La API devolvió un error: {error}")

with tab_method:
    st.subheader("Datos y método")
    st.markdown(
        """
        - **Tarea:** clasificación multiclase de cinco grados Kellgren–Lawrence.
        - **Partición:** 1.155 imágenes para entrenamiento, 247 para validación y 248 para prueba.
        - **Características:** 320 descriptores (intensidad, histograma, estadísticas espaciales, Sobel, Laplaciano y HOG).
        - **Modelos:** SVD por subespacios, SVM RBF, Random Forest, XGBoost y regresión logística como baseline.
        - **Métrica principal:** F1 macro, para dar igual peso a cada grado.

        Las imágenes bilaterales se separan en dos vistas y sus descriptores se promedian para conservar una fila por estudio.
        """
    )
    st.markdown("**Limitaciones:** conjunto de tamaño reducido, desbalance entre grados y clasificación basada en imágenes ya procesadas. El resultado no debe usarse para decisiones clínicas.")
