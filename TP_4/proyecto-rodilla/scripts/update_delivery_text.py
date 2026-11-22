"""Update human-facing documentation from the current experiment exports."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]
m=json.loads((R/'reports/modelos_avanzados_metrics.json').read_text(encoding='utf-8'))
f=m['final_test'];h=json.loads((R/'models/hyperparameters.json').read_text(encoding='utf-8'))
old=json.loads((R/'scratch/antes_rectangular_v3_2026-11-22/reports/modelos_avanzados_metrics.json').read_text(encoding='utf-8')) if (R/'scratch/antes_rectangular_v3_2026-11-22').exists() else None
if old:
 history=dict(note='Comparación descriptiva del mismo test, ya evaluado; no es evidencia independiente.',
  previous=dict(protocol=old['protocol'],model=old['selected_model'],validation_f1=old['final_test']['validation_f1'],test_f1=old['final_test']['F1 Macro'],test_accuracy=old['final_test']['Accuracy (%)']),
  current=dict(protocol=m['protocol'],model=m['selected_model'],validation_f1=f['validation_f1'],test_f1=f['F1 Macro'],test_accuracy=f['Accuracy (%)']))
 (R/'reports/revision_comparison.json').write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf-8')
readme=f"""# Clasificación de artrosis de rodilla · ML1 FIUNA

**Jorge Javier Palma Martínez** · jpalma@fiuna.edu.py · [J-Palma](https://github.com/J-Palma)

Machine Learning 1 (23433) · Ingeniería Electrónica · FIUNA · Semestre 2026

[Portal y demo](https://j-palma.github.io/ML1_2026-2/) · [Repositorio](https://github.com/J-Palma/ML1_2026-2)

## Versión vigente

Protocolo `{m['protocol']}`. Entrada rectangular de **256×144** por vista, proporción conservada, CLAHE y estadísticas que excluyen el relleno añadido. Se extraen **288 variables** (256 HOG + 32 estadísticas). SVD usa 64×36 píxeles. No se recorta la anatomía ni se estira la imagen.

La validación eligió **{m['selected_model']}**, F1 macro **{f['validation_f1']:.4f}**. Se reentrenó con 1.339 imágenes (train+validation). En las 237 imágenes de test: **accuracy {f['Accuracy (%)']:.2f} %, balanced accuracy {f['Balanced Acc (%)']:.2f} %, F1 macro {f['F1 Macro']:.4f}**, kappa cuadrático {m['quadratic_kappa']:.4f}.

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
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

Descargar [Annotated Dataset for Knee Arthritis Detection](https://www.kaggle.com/datasets/hafiznouman786/annotated-dataset-for-knee-arthritis-detection). Colocar las cinco carpetas de grados en `data/raw/Training/`.

```powershell
.\\.venv\\Scripts\\python.exe scripts/build_notebooks.py
.\\.venv\\Scripts\\python.exe scripts/execute_notebooks.py
.\\.venv\\Scripts\\python.exe scripts/build_demo.py
.\\.venv\\Scripts\\python.exe scripts/build_site.py
.\\.venv\\Scripts\\python.exe scripts/build_documents.py
.\\.venv\\Scripts\\python.exe scripts/update_delivery_text.py
```

Las curvas de aprendizaje pueden tardar varios minutos. Los modelos se ajustan en train y se seleccionan por F1 macro de validación; después se reentrenan con train+validation. El selector/escalador se ajusta dentro de cada fold de CV. El test ya examinado se declara como reevaluación, no validación independiente.

## Demo local, API y web

No se necesita descargar el dataset para inferir: el modelo final está incluido. Usar `iniciar_dashboard.cmd` y, para el modo HTTP, `iniciar_api.cmd`; o ejecutar en dos terminales:

```powershell
.\\.venv\\Scripts\\python.exe -m streamlit run dashboard/app.py
.\\.venv\\Scripts\\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Dashboard: `http://127.0.0.1:8501`. API: `http://127.0.0.1:8000/docs`, salud `/health`, predicción `POST /predict`, formulario multipart con campo `file`.

El grado usa `predict()` de forma consistente con la evaluación. Las probabilidades de SVC pueden tener un orden diferente; no son certezas clínicas. Modelo y extractor verifican que comparten la misma configuración.

GitHub Pages publica desde `main/docs` en la raíz del repositorio. La demo web explora diez imágenes de test con resultados precalculados, identificados como tales; Streamlit/API procesan imágenes nuevas. La copia `proyecto-rodilla/docs` se genera con el mismo contenido.

## Verificación

Con las imágenes y matrices presentes:

```powershell
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv\\Scripts\\python.exe scripts/check_dashboard.py
```

Las pruebas verifican duplicados con compresiones distintas, conflictos entre etiquetas, conservación de originales, copia limpia idempotente, máscara/estadísticas, hashes entre particiones, modelo vs imagen vs API y rechazo de archivos inválidos. `reports/verification.json` registra las comprobaciones de esta revisión.

## Documentos y modelo

- `reports/informe_final.pdf`: informe vigente.
- `reports/informe_parcial1.pdf`: resumen retrospectivo; no acredita entrega anterior.
- `presentation/slides.pdf`: 14 diapositivas vigentes.
- `presentation/guia_defensa.md`: guion y preguntas para la defensa.
- `presentation/preparacion_defensa_completa.pdf`: guía ampliada con 112 preguntas y anexo SVD.
- `models/best_model.joblib`: pipeline final serializado.
- `models/hyperparameters.json`: configuración del modelo y extractor.
- `reports/`: métricas, gráficos, auditoría y comparación de versiones.

Las carpetas de imágenes, entorno virtual y matrices no se versionan. Los manifiestos permiten reconstruir la copia limpia descargando los originales de Kaggle. El ZIP de entrega incluye diez muestras para la demo.

## Análisis SVD de referencia

El notebook 04 incluye código explicado y ejecutado de valores singulares, suma normalizada, varianza acumulada, F1 de validación frente a k, umbral de Gavish-Donoho por grado y medias/modos/reconstrucciones. Las bases se estiman solo con train y la selección conserva k=10, F1 macro de validación 0,4629. Los resultados están en `reports/svd_analysis.json`, `reports/svd_diagnostics.csv` y tres figuras `svd_*.png`.

Gavish-Donoho es un criterio de reconstrucción bajo hipótesis de ruido no verificadas, y no reemplaza la selección del clasificador. El pipeline final continúa con StandardScaler y SVM RBF C=1, sin SVD. La guía de defensa incluye cómo explicar esta diferencia y doce preguntas adicionales.

```powershell
.\\.venv\\Scripts\\python.exe scripts/update_svd_defense.py
.\\.venv\\Scripts\\python.exe scripts/build_defense_guide.py
```

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
"""
(R/'README.md').write_text(readme,encoding='utf-8')
(R/'reports/informe_final.md').write_text(f"# Informe final vigente\n\nEl documento completo está en [informe_final.pdf](informe_final.pdf).\n\nProtocolo `{m['protocol']}`; modelo {m['selected_model']}; F1 macro validación {f['validation_f1']:.4f}; F1 macro test {f['F1 Macro']:.4f}; accuracy {f['Accuracy (%)']:.2f} %. El test ya fue evaluado en v2: reevaluación descriptiva.\n\nEl EDA materializa 1.576 archivos limpios sin modificar los 1.650 originales.\n",encoding='utf-8')
# Preserve the expanded oral-defense guide instead of replacing it with a summary.
print('README y resumen actualizados; la guía de defensa se conserva.')
