# Clasificación de artrosis de rodilla · ML1 FIUNA

**Jorge Javier Palma Martínez** · jpalma@fiuna.edu.py · [J-Palma](https://github.com/J-Palma)

Machine Learning 1 (23433) · Ingeniería Electrónica · FIUNA · Semestre 2026

[Portal y demo](https://j-palma.github.io/ML1_2026-2/) · [Repositorio](https://github.com/J-Palma/ML1_2026-2)

## Versión vigente

Protocolo `rectangular-masked-v3-2026-11-22`. Entrada rectangular de **256×144** por vista, proporción conservada, CLAHE y estadísticas que excluyen el relleno añadido. Se extraen **288 variables** (256 HOG + 32 estadísticas). SVD usa 64×36 píxeles. No se recorta la anatomía ni se estira la imagen.

La validación eligió **SVM RBF C=1**, F1 macro **0.6300**. Se reentrenó con 1.339 imágenes (train+validation). En las 237 imágenes de test: **accuracy 71.31 %, balanced accuracy 72.09 %, F1 macro 0.7155**, kappa cuadrático 0.8567.

**Test reutilizado:** las mismas 237 imágenes ya fueron evaluadas en v2. Esta reevaluación es descriptiva y no constituye un nuevo test independiente. La representación se eligió usando train/validation; no se afirma que la nueva versión mejore todas las métricas. `reports/revision_comparison.json` conserva la comparación entre versiones. El primer protocolo, que compartía duplicados entre particiones, queda reemplazado.

Prototipo académico sin validación clínica o externa. No usar para diagnóstico. El modelo no comprueba que una imagen arbitraria sea una rodilla. En radiografías bilaterales predice una etiqueta por estudio promediando ambas vistas.

## Limpieza reproducible del dataset

`01_EDA.ipynb` muestra inventario, hashes, ejemplos repetidos, conflictos, exclusiones, copia limpia y particiones. La implementación está en `src/data_cleaning.py`.

- `data/raw/Training/`: 1.650 imágenes originales, sin modificaciones.
- `data/clean/Training/`: copia materializada de **1.576 imágenes únicas**.
- Se excluyen 6 imágenes de 3 grupos con etiquetas contradictorias y 68 copias exactas.
- `data/image_inventory.csv`: todas las decisiones y la copia canónica.
- `data/excluded_images.csv`: 74 archivos excluidos y su motivo.
- `data/clean_manifest.csv`: correspondencia entre archivo original y copia limpia.
- `data/dataset_manifest.csv`: rutas originales/limpias, hashes y particiones.
- Train 1.103, validación 236, test 237: mismas particiones depuradas, estratificadas por clase y resolución, semilla 42.

La identidad usa forma y píxeles decodificados, no el nombre del archivo. Se excluye el grupo completo cuando los mismos píxeles tienen etiquetas distintas. No hay IDs de paciente: no descarta recortes, vistas relacionadas ni duplicados aproximados.

## Notebooks

1. `01_EDA.ipynb`: limpieza explícita y EDA.
2. `02_Preprocesamiento.ipynb`: máscara, representación rectangular y experimento previo.
3. `03_Baseline.ipynb`: Dummy y logística regularizada.
4. `04_Modelos_Avanzados.ipynb`: SVM, Random Forest, XGBoost, SVD.
5. `05_Feature_Engineering.ipynb`: comparación ANOVA y PCA.
6. `06_Comparacion_Final.ipynb`: selección, reentrenamiento, métricas, límites y despliegue.

Cada etapa contiene código completo, explicación antes de cada celda y salidas guardadas. Para leer en GitHub no hace falta ejecutar nada. Para reconstruir, reiniciar el kernel y ejecutar de arriba hacia abajo los notebooks 01 a 06; las celdas de entrenamiento ajustan los modelos y las curvas pueden tardar varios minutos. El primer notebook regenera la copia limpia sin modificar originales. No hay interruptor RECOMPUTE ni llamadas resumidas a workflow para ocultar los pasos.

## Instalación y reproducción

Abrir en Antigravity la carpeta completa `proyecto-rodilla` (Python 3.12):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Descargar [Annotated Dataset for Knee Arthritis Detection](https://www.kaggle.com/datasets/hafiznouman786/annotated-dataset-for-knee-arthritis-detection). Colocar las cinco carpetas de grados en `data/raw/Training/`.

```powershell
.\.venv\Scripts\python.exe scripts/build_notebooks.py
.\.venv\Scripts\python.exe scripts/execute_notebooks.py
.\.venv\Scripts\python.exe scripts/build_demo.py
.\.venv\Scripts\python.exe scripts/build_site.py
.\.venv\Scripts\python.exe scripts/build_documents.py
.\.venv\Scripts\python.exe scripts/update_delivery_text.py
```

Las curvas de aprendizaje pueden tardar varios minutos. Los modelos se ajustan en train y se seleccionan por F1 macro de validación; después se reentrenan con train+validation. El selector/escalador se ajusta dentro de cada fold de CV. El test ya examinado se declara como reevaluación, no validación independiente.

## Demo local, API y web

No se necesita descargar el dataset para inferir: el modelo final está incluido. Usar `iniciar_dashboard.cmd` y, para el modo HTTP, `iniciar_api.cmd`; o ejecutar en dos terminales:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Dashboard: `http://127.0.0.1:8501`. API: `http://127.0.0.1:8000/docs`, salud `/health`, predicción `POST /predict`, formulario multipart con campo `file`.

El grado usa `predict()` de forma consistente con la evaluación. Las probabilidades de SVC pueden tener un orden diferente; no son certezas clínicas. Modelo y extractor verifican que comparten la misma configuración.

GitHub Pages publica desde `main/docs` en la raíz del repositorio. La demo web explora diez imágenes de test con resultados precalculados, identificados como tales; Streamlit/API procesan imágenes nuevas. La copia `proyecto-rodilla/docs` se genera con el mismo contenido.

## Verificación

Con las imágenes y matrices presentes:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/check_dashboard.py
```

Las pruebas verifican duplicados con compresiones distintas, conflictos entre etiquetas, conservación de originales, copia limpia idempotente, máscara/estadísticas, hashes entre particiones, modelo vs imagen vs API y rechazo de archivos inválidos. `reports/verification.json` registra las comprobaciones de esta revisión.

## Documentos y modelo

- `reports/informe_final.pdf`: informe vigente.
- `reports/informe_parcial1.pdf`: resumen retrospectivo; no acredita entrega anterior.
- `presentation/slides.pdf`: 14 diapositivas vigentes.
- `presentation/guia_defensa.md`: guion y preguntas para la defensa.
- `models/best_model.joblib`: pipeline final serializado.
- `models/hyperparameters.json`: configuración del modelo y extractor.
- `reports/`: métricas, gráficos, auditoría y comparación de versiones.

Las carpetas de imágenes, entorno virtual y matrices no se versionan. Los manifiestos permiten reconstruir la copia limpia descargando los originales de Kaggle. El ZIP de entrega incluye diez muestras para la demo.

## Docker

Se incluye `api/Dockerfile`. Docker no está instalado en el entorno revisado; la imagen no se ha construido. La API sí se prueba con Python.

```powershell
docker build -f api/Dockerfile -t rodilla-api .
docker run --rm -p 8000:8000 rodilla-api
```

## Fuentes

- [Dataset de Kaggle](https://www.kaggle.com/datasets/hafiznouman786/annotated-dataset-for-knee-arthritis-detection).
- [Referencia de la cátedra](https://github.com/carlobeni/Machine-Learning).
- [Video ML1-ECA](https://www.youtube.com/watch?v=isgBodVA2C4), guía de presentación web; no es fuente de métricas.
- Plan de clases ML1 FIUNA, segundo ciclo 2026, apartados 2.3 a 2.5.

Los documentos y la implementación no acreditan por sí solos envío en eAula, defensa oral ni aprobación de otros TPs/cuestionarios.
