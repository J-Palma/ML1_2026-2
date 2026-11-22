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

Protocolo `rectangular-masked-v3-2026-11-22`. El EDA materializa una copia limpia de 1.576 imágenes, excluyendo 6 imágenes contradictorias y 68 copias, sin modificar originales. La entrada rectangular 256×144 usa estadísticas con máscara y 288 descriptores. Validación elige **SVM RBF C=1**: F1 validación 0.6300; test de 237 imágenes: **accuracy 71.31 %, F1 macro 0.7155**. El test ya se evaluó en v2; esta reevaluación no constituye confirmación independiente.

La demo web explora resultados precalculados; Streamlit/FastAPI procesan imágenes nuevas. Prototipo académico; no usar para diagnóstico.
