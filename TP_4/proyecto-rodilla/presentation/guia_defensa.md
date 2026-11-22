# Guía de defensa · 8 a 10 minutos

1. Problema: clasificar cinco grados de artrosis; motivación desde procesamiento de imágenes biomédicas.
2. Datos: 1.650 originales; hashes de píxeles; 6 imágenes contradictorias y 68 copias excluidas; 1.576 únicas.
3. Mostrar `01_EDA`: ejemplos de duplicados, motivos y copia `data/clean/Training` sin modificar originales.
4. Preprocesamiento: entrada 256×144, proporción conservada, máscara para estadísticas; 288 variables y SVD sobre 64×36.
5. Comparación en validación: SVM, Random Forest, XGBoost, logística y SVD; ANOVA y PCA evaluados, no obligatoriamente elegidos.
6. Elegido: **SVM RBF C=1**, F1 validación **0.6300**. Reentrenamiento con 1.339 imágenes.
7. Test descriptivo: accuracy **71.31 %**, balanced accuracy **72.09 %**, F1 **0.7155**. El mismo test ya se evaluó en v2; no hay confirmación independiente.
8. Mostrar matriz, curvas y errores. Menos relleno no garantiza que mejoren todas las métricas; no seleccionar usando test.
9. Demo: dashboard local o API. El sitio público muestra diez predicciones precalculadas.
10. Limitaciones: sin IDs ni cohorte externa, formato asociado al grado, promedio bilateral y probabilidades sin calibración clínica.

## Preguntas probables

**¿Se borraron originales?** No: se generó una copia limpia y un registro de exclusiones.

**¿Detectaron todos los duplicados?** Solo coincidencias exactas de píxeles; recortes y casi duplicados pueden persistir.

**¿Por qué rectangular?** Reduce el relleno artificial y conserva geometría; el cambio de HOG también influye.

**¿Por qué no ANOVA/PCA en el final?** Se comparan por validación, no se fuerzan. La configuración elegida está registrada antes de reevaluar test.

**¿Se puede diagnosticar?** No; es un prototipo académico sin validación clínica o externa.
