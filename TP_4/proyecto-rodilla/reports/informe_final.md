# Informe final · Clasificación de artrosis de rodilla

Jorge Javier Palma Martínez · jpalma@fiuna.edu.py · ML1 FIUNA · 2026

El documento completo, con métodos, tablas, figuras, discusión y referencias, está en [informe_final.pdf](informe_final.pdf). El generador reproducible es `scripts/build_documents.py`.

## Protocolo vigente

`deduplicated-v2-2026-11-22`. Tras auditar 1.650 imágenes, se excluyeron 6 con etiquetas contradictorias y 68 copias exactas. Quedaron 1.576 únicas: 1.103 train, 236 validación y 237 test. Las particiones no comparten hashes exactos; no hay IDs que permitan garantizar independencia por paciente. Las métricas anteriores quedan reemplazadas.

## Modelo final

Selección por F1 macro de validación, congelada antes del test: **SVM ANOVA160 C=10**. Pipeline con ANOVA de 160 variables, StandardScaler y SVM RBF C=10. Reentrenamiento con 1.339 imágenes. Test: accuracy 73.00%, balanced accuracy 71.85%, F1 macro 0.7207, kappa cuadrático 0.8485.

## Alcance

La auditoría elimina duplicados exactos, pero no todas las vistas relacionadas. Hay sobreajuste, desbalance, pérdida de lateralidad en el promedio bilateral y ausencia de evaluación externa/clínica. Probabilidades sin calibración clínica. No usar para diagnóstico.

## Entregables

Seis notebooks ejecutados; manifiestos; modelo y parámetros; informe PDF; 14 diapositivas; API FastAPI; dashboard Streamlit; portal con demo precalculada; pruebas y registro `verification.json`.

Fuentes: [Kaggle](https://www.kaggle.com/datasets/hafiznouman786/annotated-dataset-for-knee-arthritis-detection), [cátedra](https://github.com/carlobeni/Machine-Learning), plan ML1 FIUNA 2026.
