# Guía breve para la defensa

## Recorrido sugerido (8–10 minutos)

1. **Problema:** clasificación supervisada multiclase de cinco grados. No diagnostica pacientes.
2. **Auditoría:** explicar por qué se reemplazó el protocolo inicial: 30 grupos exactos cruzaban particiones; 3 grupos tenían etiquetas contradictorias. Excluir 6 imágenes conflictivas y 68 copias deja 1.576 imágenes únicas.
3. **Representación:** conservar proporción, CLAHE, 320 números; HOG describe orientación de bordes. El padding afecta estadísticas y es una limitación.
4. **Selección:** ANOVA elige variables individuales, PCA combina variables en componentes. Ambos se ajustan en entrenamiento. SVD construye subespacios por clase.
5. **Validación:** parámetros, representación y modelo elegidos con validación, no test. Reentrenar con train+validation y después medir test.
6. **Resultado:** SVM RBF C=10 con 160 variables. Accuracy 73,00%, F1 macro 0,7207, balanced accuracy 71,85%, kappa cuadrático 0,8485 sobre 237 imágenes.
7. **Errores:** mostrar matriz y reconocer que Leve es más difícil; discutir la brecha train/CV y posible sobreajuste.
8. **Demo:** Streamlit, cargar una muestra y ejecutar primero Modelo local, luego API HTTP. La web tiene muestras precalculadas e incluye ejemplos fallidos.
9. **Límites:** sin IDs de paciente, recortes posiblemente relacionados, formato asociado al grado, promedio bilateral, sin cohorte externa.

## Preguntas que conviene saber responder

- **¿Por qué F1 macro?** Da igual peso a los cinco grados pese al desbalance.
- **¿Por qué no basta accuracy?** Puede ocultar una clase con recall bajo.
- **¿Por qué cambió de XGBoost a SVM?** Se corrigieron datos y particiones, se añadió selección de variables y se eligió por validación. No se deben comparar directamente los protocolos.
- **¿Qué aprendió SVM?** Una frontera de separación no lineal, con kernel RBF, sobre descriptores numéricos seleccionados y estandarizados.
- **¿Qué significa C?** Controla el compromiso entre margen y penalización de errores; C=10 se eligió entre candidatos de validación.
- **¿Hay fuga de datos?** No quedan hashes exactos compartidos; sin IDs no se puede garantizar independencia por paciente ni descartar recortes similares.
- **¿La probabilidad es certeza médica?** No. Son puntuaciones estimadas del modelo, sin calibración clínica. SVC puede decidir una clase distinta al argmax de probabilidades.
- **¿Usa la imagen entera?** Se calculan descriptores de vistas ajustadas; si es bilateral se promedian y se pierde lateralidad.
- **¿Por qué XGBoost tiene 1.500 árboles?** Son 300 rondas y cinco clases, un árbol por clase en cada ronda.
- **¿Qué falta para una aplicación clínica?** Revisión especializada, separación por paciente, validación externa, control de calidad y calibración.

La defensa oral y el envío en eAula deben realizarlos el estudiante. El proyecto no acredita cuestionarios o TPs de exoneración.
