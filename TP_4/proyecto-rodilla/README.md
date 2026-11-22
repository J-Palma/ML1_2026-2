# Clasificación de artrosis de rodilla · Entrega final ML1

**Jorge Javier Palma Martínez** · jpalma@fiuna.edu.py · [J-Palma](https://github.com/J-Palma)
Machine Learning 1 (23433) · Ingeniería Electrónica · FIUNA · Semestre 2026

[Portal y demo](https://j-palma.github.io/ML1_2026-2/) · [Repositorio](https://github.com/J-Palma/ML1_2026-2)

## Resultado vigente

Protocolo `deduplicated-v2-2026-11-22`. La auditoría halló 63 grupos de píxeles repetidos y 3 con etiquetas contradictorias. Se excluyeron 6 imágenes contradictorias y 68 copias: **1.576 imágenes únicas**. Las nuevas particiones contienen 1.103 train, 236 validación y 237 test, estratificadas por clase y resolución con semilla 42. No comparten hashes exactos; no hay IDs que permitan garantizar separación por paciente.

Las métricas de la versión anterior quedan reemplazadas: 30 grupos de duplicados cruzaban particiones. No se comparan directamente protocolos distintos. Los originales permanecen intactos.

La selección por **F1 macro en validación** eligió **SVM RBF, C=10, ANOVA de 160 variables** de las 320 iniciales. Se reentrenó con train + validation. En test: **accuracy 73,00 %, balanced accuracy 71,85 %, F1 macro 0,7207**, kappa cuadrático 0,8485.

Prototipo académico, sin validación clínica ni externa. No usar para diagnóstico. El modelo no detecta que una imagen arbitraria sea una rodilla. En imágenes bilaterales predice una etiqueta de estudio, promediando ambas vistas.

## Archivos de entrega

- `notebooks/`: seis notebooks numerados, ejecutados y con figuras.
- `src/workflow.py`: auditoría, extracción, ajuste, selección, evaluación y gráficos.
- `models/best_model.joblib`: Pipeline final (selector + escalador + SVM).
- `models/hyperparameters.json`: configuración y nombres de las 160 variables.
- `data/image_inventory.csv`: decisiones de inclusión y hashes de todas las imágenes.
- `data/dataset_manifest.csv`: particiones vigentes con rutas relativas.
- `reports/informe_final.pdf`: informe completo.
- `reports/informe_parcial1.pdf`: resumen de etapas iniciales, compilado retrospectivamente; no acredita un envío previo.
- `presentation/slides.pdf`: defensa de 14 diapositivas.
- `reports/`: métricas, auditoría, gráficos y registro de comprobaciones.
- `api/`: FastAPI y Dockerfile.
- `dashboard/`: Streamlit con resultados y carga de imágenes.
- `tests/test_integration.py`: pruebas de consistencia y errores HTTP.

## Instalación (Windows, Python 3.12)

Abrir la carpeta raíz `proyecto-rodilla` en Antigravity y su terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No se necesita el dataset para ejecutar la API o la predicción local del dashboard: el modelo final está incluido.

## Probar la demo

Dashboard con inferencia local, en una terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Abrir `http://127.0.0.1:8501`, pestaña **Probar radiografía**. Seleccionar una imagen y pulsar **Clasificar imagen**. Las probabilidades no son certezas clínicas. SVC puede producir una decisión distinta al argmax de sus probabilidades; se usa `predict()` de forma consistente con la evaluación.

API, en otra terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Documentación: `http://127.0.0.1:8000/docs`; salud: `/health`; predicción: `POST /predict` con formulario multipart, campo `file`. La API devuelve etiqueta, modelo, protocolo y puntuaciones. En el dashboard se puede seleccionar **API HTTP** para usarla. `KNEE_API_URL` cambia la dirección si hace falta. Ambos servicios son demostraciones locales; GitHub Pages no aloja Python.

## Reconstruir el experimento

1. Descargar [Annotated Dataset for Knee Arthritis Detection](https://www.kaggle.com/datasets/hafiznouman786/annotated-dataset-for-knee-arthritis-detection). Colocar las carpetas `0Normal` a `4Severe` en `data/raw/Training/`.
2. Ejecutar:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py
.\.venv\Scripts\python.exe scripts/build_notebooks.py
.\.venv\Scripts\python.exe scripts/execute_notebooks.py
.\.venv\Scripts\python.exe scripts/build_demo.py
.\.venv\Scripts\python.exe scripts/build_site.py
.\.venv\Scripts\python.exe scripts/build_documents.py
```

El pipeline puede tardar varios minutos en las curvas de aprendizaje. No reutiliza las particiones antiguas; construye las actuales de forma determinista. Los notebooks muestran por defecto exportaciones ya calculadas. Para recalcular desde los notebooks, poner `RECOMPUTE=True` y ejecutar en orden del 01 al 06. El test solo se evalúa después de congelar la selección por validación.

## Verificar

Con el dataset y características presentes:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/check_dashboard.py
```

Las pruebas comparan un ejemplo por clase contra la matriz de test serializada y la API, comprueban hashes y rechazan cargas inválidas. `scripts/check_dashboard.py` verifica que Streamlit renderiza métricas y cambia la métrica sin excepciones. `reports/verification.json` registra la revisión de esta entrega.

## Docker de la API (opcional)

Desde esta carpeta, con Docker instalado:

```powershell
docker build -f api/Dockerfile -t rodilla-api .
docker run --rm -p 8000:8000 rodilla-api
```

El Dockerfile acompaña la entrega; si no se dispone de Docker, usar los comandos Python anteriores. No se afirma haber validado una imagen de Docker sin construirla.

## Sitio estático y publicación

La página está en `/docs` de la raíz del repositorio ML1_2026-2; la copia local `proyecto-rodilla/docs` se usa para generar y revisar. GitHub Pages publica desde `main/docs`. La demo web explora diez muestras de test y sus predicciones precalculadas, identificadas como tales. No permite diagnóstico ni ejecuta el modelo con archivos nuevos; esa función está en Streamlit/API.

```powershell
python -m http.server 8080 --directory docs
```

## Fuentes y alcance

- Dataset público de Kaggle citado arriba.
- [Repositorio de referencia de la cátedra](https://github.com/carlobeni/Machine-Learning).
- [Video de ejemplo de reportería web](https://www.youtube.com/watch?v=isgBodVA2C4).
- Plan de clases ML1, FIUNA, segundo ciclo 2026, apartados 2.3–2.5.

Quedan fuera del alcance: validación clínica, evaluación externa e identificación de pacientes. La defensa oral y la entrega en eAula deben realizarlas el estudiante; el repositorio no acredita esos trámites ni los cuestionarios/TPs de exoneración.
