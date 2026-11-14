# Clasificación de artrosis de rodilla con Machine Learning

Proyecto académico de Machine Learning 1 (FIUNA, 2026) para clasificar radiografías en cinco grados Kellgren–Lawrence: Normal, Dudosa, Leve, Moderada y Severa. El propósito es comparar modelos clásicos y documentar un flujo reproducible. No es una herramienta clínica.

## Datos y método

El dataset tiene 1.650 imágenes: 1.155 para entrenamiento, 247 para validación y 248 para prueba. La partición está estratificada por grado. Hay 1.464 imágenes de 300 × 162 y 186 de 640 × 161. Algunas radiografías muestran ambas rodillas; el preprocesamiento separa sus vistas para describirlas y promedia los descriptores para conservar una fila por estudio.

El extractor produce 320 características: estadísticas de intensidad, histogramas, estadísticas espaciales, gradientes Sobel, Laplaciano y HOG. Se comparan regresión logística como baseline, SVD por subespacios, SVM RBF, Random Forest y XGBoost. F1 macro es la métrica principal porque da el mismo peso a cada grado. Los hiperparámetros se seleccionan con validación y se informa el desempeño final en test.

## Resultados de test

| Modelo | Accuracy | Balanced accuracy | F1 macro |
|---|---:|---:|---:|
| XGBoost (300 árboles) | 72,18 % | 68,84 % | 0,7057 |
| SVM RBF | 68,95 % | 66,18 % | 0,6702 |
| Random Forest (200 árboles) | 62,10 % | 57,73 % | 0,5798 |
| Regresión logística L2 | 57,66 % | 59,03 % | 0,5676 |
| SVD por subespacios | 55,24 % | 52,96 % | 0,5516 |

En esta partición XGBoost obtuvo el mejor F1 macro. Estos resultados son experimentales; no validan uso clínico ni generalización a otros centros.

## Estructura

- `notebooks/01_EDA.ipynb` a `05_Resultados.ipynb`: análisis, preprocesamiento, modelos y conclusiones.
- `src/`: preprocesamiento, extracción de características e inferencia.
- `data/processed/`: particiones y matrices derivadas.
- `models/`: modelos entrenados.
- `reports/`: tablas de métricas, figuras e informe.
- `docs/`: página estática lista para GitHub Pages.
- `api/`: servicio FastAPI de demostración.
- `dashboard/`: panel Streamlit de resultados.

## Reproducir en Windows

Desde la carpeta raíz del proyecto, con Python 3.10 o superior:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Coloca el dataset bajo `data/raw/Training/` conservando sus carpetas por clase. Abre la carpeta raíz en Antigravity o Jupyter y ejecuta, en orden, `01_EDA.ipynb`, `02_Preprocesamiento.ipynb`, `03_Baseline.ipynb`, `04_Modelos_Avanzados.ipynb` y `05_Resultados.ipynb`. El notebook 02 puede tardar al extraer descriptores de las 1.650 radiografías.

## Ejecutar la demostración local

Terminal 1, API:

```powershell
python -m uvicorn api.app:app --reload
```

La documentación interactiva queda en `http://127.0.0.1:8000/docs`. Terminal 2, dashboard:

```powershell
streamlit run dashboard/app.py
```

El dashboard consulta la API local para la predicción. La carga de una imagen solo debe hacerse con datos autorizados y no destinados a decisiones clínicas.

Para previsualizar la página web sin desplegarla:

```powershell
python -m http.server 8080 --directory docs
```

Luego abre `http://127.0.0.1:8080`. Para GitHub Pages, sube esta carpeta `docs/` al repositorio del proyecto y selecciona `main` / `docs` en Settings → Pages.

## Referencias

- Dataset: [Annotated Dataset for Knee Arthritis Detection](https://www.kaggle.com/datasets/hafiznouman786/annotated-dataset-for-knee-arthritis-detection).
- Repositorio de referencia de la materia: [Machine-Learning](https://github.com/carlobeni/Machine-Learning).
- Ejemplo web del curso: [ML1-ECA: Next.js y despliegue](https://www.youtube.com/watch?v=isgBodVA2C4).

Autor: **Jorge Javier Palma Martínez** · jpalma@fiuna.edu.py · [J-Palma](https://github.com/J-Palma). Repositorio oficial: https://github.com/J-Palma/ML1_2026-2.

## Entregables preparados

- Informe: `reports/informe_final.md` y `output/pdf/informe_final.pdf`.
- Presentación de 13 diapositivas: `output/presentation/presentacion_proyecto_rodilla.pptx` y `output/pdf/presentacion_proyecto_rodilla.pdf`.
- Página estática: carpeta `docs/`, lista para configurar en GitHub Pages.
- API FastAPI y panel Streamlit: `api/` y `dashboard/`.

Autor: **Jorge Javier Palma Martínez** · jpalma@fiuna.edu.py · [J-Palma](https://github.com/J-Palma).

Repositorio: https://github.com/J-Palma/ML1_2026-2. Sitio previsto en GitHub Pages: https://j-palma.github.io/ML1_2026-2/
