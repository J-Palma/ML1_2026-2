"""Refresh the SVD appendix from computed results, retaining the reviewed guide."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'presentation/guia_defensa.md'
text = path.read_text(encoding='utf-8-sig').split('\n## 8. Anexo SVD')[0].rstrip()
analysis = json.loads((ROOT / 'reports/svd_analysis.json').read_text(encoding='utf-8'))
text = text.replace('14 diapositivas y 100 preguntas', '14 diapositivas y 112 preguntas')
text = text.replace('reports/padding_experiment/results.json.',
                    'reports/padding_experiment/results.json y reports/svd_analysis.json.')
text = text.replace('| Límite central |',
    '| SVD comparado | k=10; F1 macro de validación 0,4629; no integra el SVM final |\n| Límite central |') if '| SVD comparado |' not in text else text
old = ('Comparé selección ANOVA de 40, 80 y 160 variables y PCA con 32, 64 y 128 componentes. '
       'Las transformaciones se ajustan únicamente con entrenamiento y las alternativas se comparan en validación. '
       'En esta revisión, el mayor F1 macro de validación lo obtuvo el SVM con las 288 variables completas. '
       'Por eso evalué selección y reducción dimensional, pero no las forcé dentro del modelo final. '
       'La alternativa más compleja no tiene que ser necesariamente la elegida.')
new = ('Comparé ANOVA de 40, 80 y 160 variables y PCA de 32, 64 y 128 componentes. '
       'También estudié espectros, varianza y reconstrucción de cinco subespacios SVD. '
       'Para SVD, k igual a diez obtuvo el mejor F1 macro de validación entre cinco rangos. '
       'Retiene entre 68,7 y 78,2 por ciento de varianza según el grado: más varianza no garantiza mejor clasificación. '
       'Gavish-Donoho aporta otro criterio, orientado a reconstrucción con ruido, cuyas hipótesis no verifiqué. '
       'El modelo final conserva las 288 variables: StandardScaler seguido de SVM RBF C igual a uno.')
text = text.replace(old, new).replace('Diapositiva 6. Selección y reducción dimensional · 40 segundos',
                                      'Diapositiva 6. Selección y reducción dimensional · 50 segundos')
if 'T10. Gavish' not in text:
    text = text.replace('Estas referencias explican APIs y conceptos;',
        'T10. Gavish y Donoho, umbral de valores singulares: https://arxiv.org/abs/1305.5870\n\n'
        'T11. NumPy SVD: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html\n\n'
        'T12. Ejemplo de la cátedra: https://github.com/carlobeni/Machine-Learning/tree/project-web/models\n\n'
        'Estas referencias explican APIs y conceptos;')

rows = '\n'.join(f"| {c['label']} | Varianza k=10: {100*c['variance_at_selected_k']:.2f} %; "
                 f"k90={c['k_90']}; k95={c['k_95']}; rango GD={c['gd_rank']} |"
                 for c in analysis['classes'])
appendix = f'''

## 8. Anexo SVD: cómo explicar los nuevos gráficos

El notebook 04 contiene el código, las explicaciones y las salidas de este análisis. Se aprendieron el escalador, la media y la base de cada grado exclusivamente con 1.103 imágenes de train. Se reutilizó el barrido de validación, sin consultar test ni modificar el modelo final.

### Espectro, acumulación y elección de k

Los valores singulares describen la importancia de las direcciones ortogonales de cada matriz centrada. Su suma acumulada normalizada, suma(sigma_i)/suma(sigma), es la curva que el ejemplo llama energía. La energía cuadrática y la varianza acumulada se calculan con suma(sigma_i²)/suma(sigma_i² total). Hay que distinguir las dos fórmulas. Al centrar n imágenes el rango no supera n-1: la caída final a casi cero no demuestra un umbral anatómico ni de ruido.

![Espectros calculados en train; F1 macro de validación frente a k.](../reports/figures/svd_seleccion_dimension.png)

Se evaluaron k=5,10,20,30,50. El mejor fue k={analysis['selected_k']} con F1 macro de validación {analysis['selected_validation_f1']:.4f}. No se exigió 90 % de varianza. Es el mejor de los candidatos probados, no un óptimo universal. Las cifras de la tabla se refieren al ajuste de train para este diagnóstico, no al reajuste de train más validación.

| Grado | Resultado de train |
| --- | --- |
{rows}

### Gavish-Donoho: diagnóstico teórico complementario

El umbral tau=omega(beta)·mediana(sigma) retiene los valores singulares superiores a tau. Beta=min(m,n)/max(m,n) se calcula para cada grado porque sus tamaños son distintos. Omega se obtiene a partir de la mediana de Marchenko-Pastur. La implementación usa integración numérica y búsqueda de raíz, como referencia matemática de la clase.

![Umbrales distintos por grado; el rango teórico no reemplaza la elección con validación.](../reports/figures/svd_gavish_donoho.png)

Los rangos calculados son 99, 96, 43, 40 y 38. El método se estudia para recuperar matrices de bajo rango contaminadas con ruido blanco. No verifiqué esas hipótesis en radiografías ya estandarizadas y centradas. Por eso no afirmo que estos rangos sean óptimos para clasificar ni los incorporo a la búsqueda. Una recomendación de reconstrucción puede diferir del k que separa mejor las clases.

### Medias, modos y reconstrucción

![Un ejemplo fijo de validación por grado; reconstrucción ilustrativa con la base de la etiqueta conocida.](../reports/figures/svd_componentes_reconstruccion.png)

Cada fila muestra media de train, primer modo con signo, entrada de validación, reconstrucción con k=10 y error absoluto. Se tomó el primer ejemplo de cada grado, sin buscar el mejor. La media y la reconstrucción se desestandarizan para volver a la escala de píxeles. Los modos se muestran en coordenadas estandarizadas y su signo es arbitrario. No son lesiones ni explicaciones clínicas.

Para esta figura se conoce la etiqueta y se usa su base, únicamente para mostrar geometría. Para clasificar una imagen nueva no conozco esa etiqueta: calculo los cinco residuos y elijo el menor. Los píxeles representan vistas a 64×36, y en estudios bilaterales corresponden al promedio de las vistas. Esto también limita la interpretación visual.

### Respuesta oral de aproximadamente 45 segundos

“Agregué un análisis de SVD como referencia de la cátedra. Construí cinco subespacios usando solo entrenamiento y mostré sus valores singulares, varianza acumulada y reconstrucciones. Probé cinco rangos y elegí diez componentes por F1 macro de validación. Ese valor no conserva el noventa por ciento de la varianza; conservar más información no garantiza separar mejor los grados. También calculé Gavish-Donoho, que es un criterio de reconstrucción bajo hipótesis de ruido y no una garantía para mi clasificación. Estos análisis explican el SVD comparado; mi modelo final sigue siendo un SVM RBF con 288 características.”

## 9. Doce preguntas adicionales sobre SVD

**101. ¿Qué representan U, Sigma y V transpuesta?**

Para la matriz centrada B de píxeles por imágenes, B=U·Sigma·V transpuesta. Las columnas de U son direcciones ortonormales en el espacio de píxeles; Sigma contiene magnitudes singulares ordenadas. V describe cómo participan las muestras. Uso las primeras k columnas de U para la proyección.

**102. ¿La energía acumulada es igual a la varianza explicada?**

Depende de la fórmula. El ejemplo de la cátedra llama energía a la suma normalizada de sigma. La proporción de energía cuadrática usa sigma al cuadrado y, para matrices centradas, coincide con la varianza acumulada. Mis gráficos nombran cada fórmula de forma explícita.

**103. ¿Por qué k=10 si no alcanza 90 % de varianza?**

Porque seleccioné por clasificación: obtuvo F1 macro de validación 0,4629, el mayor entre k=5,10,20,30,50. El rango diez conserva entre 68,7 y 78,2 % según el grado. El 90 % sería otro criterio de reconstrucción, no una condición de mi búsqueda.

**104. ¿Más componentes siempre mejoran la predicción?**

No. Disminuyen el error de reconstrucción de la misma base, pero también pueden modelar variación que no separa grados. Con bases amplias, más clases pueden reconstruir bien una misma entrada y perder discriminación. Por eso comparé F1 en validación.

**105. ¿Qué optimiza Gavish-Donoho?**

Un umbral duro de valores singulares para recuperar una matriz de bajo rango con ruido, bajo las hipótesis del estudio. No optimiza directamente F1, accuracy o diagnóstico. Lo presento como complemento teórico, no como prueba de superioridad de un clasificador. Fuente T10.

**106. ¿Por qué el umbral es diferente para cada grado?**

Las matrices tienen cantidades distintas de imágenes y espectros distintos. Calculo beta por grado, obtengo omega(beta) y multiplico por su propia mediana singular. Reutilizar el beta de una sola clase sería incorrecto cuando sus tamaños difieren.

**107. ¿Por qué no reemplazaste k=10 por el rango teórico?**

Porque responden a objetivos diferentes y no verifiqué el modelo de ruido blanco. Los rangos teóricos son 99,96,43,40,38. No los evalué como nuevos candidatos de clasificación ni afirmo que mejoren F1. La configuración congelada mantiene k=10 para SVD y SVM como modelo final.

**108. ¿Las reconstrucciones usaron test o lo ajustaron?**

No. Escalador, medias y bases usan train; las imágenes mostradas son el primer ejemplo de validación de cada grado. Se reconstruyen con su etiqueta conocida solo para ilustrar el subespacio. Eso no equivale a medir clasificación ni demuestra independencia adicional.

**109. ¿La imagen media o el primer modo indican dónde está la lesión?**

No. Resumen media y variación de intensidades, con posible influencia de formato y fondo. El modo tiene signo arbitrario y está estandarizado. No es una segmentación, un mapa causal ni una explicación clínica del SVM.

**110. ¿Por qué deshacés la estandarización antes de mostrar la reconstrucción?**

Porque la proyección se calcula en coordenadas estandarizadas. Para comparar con intensidades originales uso inverse_transform del mismo escalador entrenado en train. Los paneles de media, entrada y reconstrucción comparten escala gris de cero a uno; el error absoluto usa esa misma unidad.

**111. ¿Por qué reconstruir con la clase conocida no da la predicción?**

Porque esa figura es una demostración geométrica condicionada a la etiqueta. Para predecir no se conoce la clase: el algoritmo calcula el residuo respecto a las cinco bases y elige el menor. Usar la etiqueta real para decidir la predicción sería fuga de información.

**112. ¿Estos gráficos cambian el SVM o sus resultados publicados?**

No. El SVM sigue usando 288 descriptores y StandardScaler, sin SVD ni PCA. Se comprobó que el archivo del modelo, las métricas finales y la decisión de selección conservan sus hashes. El análisis nuevo documenta la alternativa SVD.
'''
path.write_text(text + appendix, encoding='utf-8')
print('Guía actualizada: 112 preguntas y anexo SVD con resultados y figuras.')
