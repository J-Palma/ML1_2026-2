# Proyecto final · Clasificación de artrosis de rodilla

**Jorge Javier Palma Martínez** · jpalma@fiuna.edu.py · [J-Palma](https://github.com/J-Palma)

Machine Learning 1 (23433) · Ingeniería Electrónica · FIUNA · 2026

- [Portal y demo](https://j-palma.github.io/ML1_2026-2/)
- [Informe final PDF](proyecto-rodilla/reports/informe_final.pdf)
- [Defensa: 14 diapositivas PDF](proyecto-rodilla/presentation/slides.pdf)
- [Seis notebooks ejecutados](proyecto-rodilla/notebooks)
- [Ejecutar API y dashboard](proyecto-rodilla/README.md)
- [Registro de comprobaciones](proyecto-rodilla/reports/verification.json)

## Resultados vigentes

Protocolo `deduplicated-v2-2026-11-22`. Tras excluir conflictos de etiqueta y copias exactas, quedan 1.576 imágenes únicas. Validación elige SVM RBF con 160 variables ANOVA; test de 237 imágenes: **accuracy 73,00 %, F1 macro 0,7207**. Las métricas antiguas quedan reemplazadas debido a duplicados compartidos entre particiones. El informe documenta límites y riesgo de sobreajuste.

La demo web explora resultados precalculados; la inferencia de nuevas imágenes está en Streamlit y FastAPI. Prototipo académico; no usar para diagnóstico.
