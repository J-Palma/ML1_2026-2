"""Build fully explained notebooks with executable stage code visibly in cells.

Source functions/statements are copied from production modules at build time,
formatted for reading, and checked for complete stage coverage. No stage wrapper
or RECOMPUTE switch hides the actual training/cleaning implementation.
"""
from pathlib import Path
import ast
import hashlib
import json
import nbformat
try:
    import black
except ImportError:
    black=None

ROOT=Path(__file__).resolve().parents[1]
MODULES={name:(ROOT/f'src/{name}.py').read_text(encoding='utf-8')
         for name in ['workflow','data_cleaning','preprocessing','features','models']}
TREES={name:ast.parse(source) for name,source in MODULES.items()}
NODES={name:{node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
       for name,tree in TREES.items()}
USED={name:set() for name in ['prepare_data','extract','baseline','advanced','feature_engineering','evaluate']}
M=json.loads((ROOT/'reports/modelos_avanzados_metrics.json').read_text(encoding='utf-8'))


def formatted(source):
    source=source.strip()+'\n'
    ast.parse(source)
    return black.format_str(source,mode=black.Mode(line_length=88)) if black else source


def definition(module,name):
    return formatted(ast.get_source_segment(MODULES[module],NODES[module][name]))


def pieces(stage,start,end):
    USED[stage].update(range(start,end))
    return formatted('\n\n'.join(ast.unparse(n) for n in NODES['workflow'][stage].body[start:end]))


def last_fit(stage):
    return formatted('table = '+ast.unparse(NODES['workflow'][stage].body[-1].value)+'\ndisplay(table)')


def section(cells,title,explanation,source):
    cells.extend([nbformat.v4.new_markdown_cell('## '+title+'\n\n'+explanation),
                  nbformat.v4.new_code_cell(formatted(source))])


def start(title,summary,extra_imports=''):
    cells=[nbformat.v4.new_markdown_cell('# '+title+'\n\n**Jorge Javier Palma Martínez · ML1 FIUNA · 2026**\n\n'
        +'Protocolo: `'+M['protocol']+'`.\n\n'+summary+'\n\n'
        +'Este cuaderno contiene el código ejecutable de la etapa, dividido en celdas con explicación. '
        +'Ejecutar de arriba hacia abajo. Los notebooks se recorren del 01 al 06; los ajustes y las curvas pueden tardar varios minutos.')]
    section(cells,'1. Librerías y ubicación del proyecto',
        'Se cargan las bibliotecas utilizadas y se localiza la carpeta raíz. La misma celda funciona al abrir el notebook desde la raíz o desde `notebooks`. No entrena ningún modelo.',
        '''from pathlib import Path
import sys, json, hashlib, shutil, time, warnings, copy
import numpy as np
import pandas as pd
import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display, Image
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src/workflow.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
'''+extra_imports)
    section(cells,'2. Configuración y carpetas de salida',
        'Se fija la semilla, el orden de las cinco etiquetas y las rutas de matrices, informes y modelos. Crear estas carpetas no modifica las radiografías originales.',
        f'''DATA=ROOT/'data/processed'
REPORTS=ROOT/'reports'
FIGURES=REPORTS/'figures'
MODELS=ROOT/'models'
LABELS=['Normal','Dudosa','Leve','Moderada','Severa']
SEED=42
PROTOCOL={M['protocol']!r}
for folder in [DATA, REPORTS, FIGURES, MODELS]:
    folder.mkdir(parents=True,exist_ok=True)
print('Proyecto:',ROOT)
print('Protocolo:',PROTOCOL)
''')
    section(cells,'3. Guardar resultados y mostrar las figuras',
        'La primera función exporta JSON evitando NaN/Infinity. La segunda guarda cada gráfico y lo muestra dentro del notebook, para que el documento conserve código y salida al verse en GitHub.',
        definition('workflow','save_json')+'''
def figure(name):
    path=FIGURES/name
    plt.savefig(path,dpi=180,bbox_inches='tight')
    plt.close()
    display(Image(filename=str(path)))
''')
    return cells


TRAIN_IMPORTS='''from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,f1_score,precision_score,
    recall_score,classification_report,confusion_matrix,ConfusionMatrixDisplay,roc_curve,auc,cohen_kappa_score)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.model_selection import StratifiedKFold,learning_curve
from sklearn.base import clone
from xgboost import XGBClassifier
from src.features import FEATURE_COUNT, PREPROCESSING_SPEC
from src.models import SVDSubspaceClassifier
'''


def training_helpers(cells):
    section(cells,'4. Leer las matrices y definir las métricas',
        'El cargador rechaza matrices de otra versión. La función de evaluación calcula accuracy, balanced accuracy, F1 y precisión/recall macro. F1 macro es el criterio de selección; no se ajustan transformaciones con test.',
        definition('workflow','arrays')+definition('workflow','score')+"\na=arrays()\ndisplay({k:v.shape for k,v in a.items() if k.startswith('X_')})")
    section(cells,'5. Bucle completo de entrenamiento y validación',
        'Este es el código que recorre candidatos: toma train, ejecuta `fit`, predice validación, calcula métricas y guarda modelos y resultados. El escalador/selector de cada Pipeline se ajusta solo con train. La función no accede a las entradas de test.',
        definition('workflow','fit_candidates'))


def write(filename,cells):
    for i,c in enumerate(cells):
        if c.cell_type=='code':
            assert i>0 and cells[i-1].cell_type=='markdown',filename
            ast.parse(c.source)
            assert 'RECOMPUTE' not in c.source and 'workflow.' not in c.source.replace('src/workflow.py','')
    metadata=dict(kernelspec={'display_name':'Python (proyecto-rodilla)','language':'python','name':'python3'},
        notebook_role='full_executable_walkthrough',source_fingerprints={
            f'src/{name}.py':hashlib.sha256(value.encode()).hexdigest() for name,value in MODULES.items()})
    n=nbformat.v4.new_notebook(cells=cells,metadata=metadata)
    nbformat.validate(n);nbformat.write(n,ROOT/'notebooks'/filename)
    print(filename,':',len(cells),'celdas;',sum(c.cell_type=='code' for c in cells),'con código')


# 01: exact pixel audit, non-destructive clean copy, split and EDA.
c=start('01 · Limpieza completa y EDA',
    'Se parte de 1.650 imágenes. Se excluyen 68 copias exactas y 6 imágenes contradictorias; quedan 1.576. '
    'La limpieza conserva `data/raw/Training` y materializa `data/clean/Training` antes de dividir.',
    'from sklearn.model_selection import train_test_split\n')
section(c,'4. Identidad de una imagen: hash de forma y píxeles',
    'SHA-256 se calcula sobre la imagen decodificada, no sobre el nombre o la compresión del archivo. Esto detecta igualdad exacta; no detecta todos los recortes o casi duplicados.',definition('data_cleaning','pixel_digest'))
section(c,'5. Recorrer originales y decidir inclusiones',
    'Se muestra el algoritmo completo: lectura en gris, inventario, grupos con más de una etiqueta, exclusión completa de conflictos y conservación de una sola copia cuando las etiquetas coinciden.',definition('data_cleaning','audit_dataset'))
section(c,'6. Construir una copia limpia sin borrar originales',
    'La función copia únicamente las imágenes aceptadas, usa nombres basados en hashes y comprueba que una segunda ejecución no genere copias adicionales. Guarda la relación entre archivo original y copia limpia.',definition('data_cleaning','materialize_clean_dataset'))
section(c,'7. Ejecutar la auditoría y la copia limpia',
    'Estas instrucciones ejecutan de verdad las funciones anteriores, generan el registro de exclusiones y dejan las rutas limpias disponibles para el entrenamiento. Se muestran motivos y ejemplos.',pieces('prepare_data',1,8)+'''
display(df.audit_status.value_counts().rename('cantidad').to_frame())
display(df[df.audit_status=='conflicting_label'][['path','class_label','pixel_sha256']])
display(df[df.audit_status=='duplicate_copy'][['path','canonical_path']].head())
assert len(df)==1650 and len(clean)==1576
''')
section(c,'8. Separación estratificada después de limpiar',
    'Se forman estratos por clase y resolución y se hacen dos separaciones reproducibles para obtener 70/15/15. Se comprueba que no haya hashes repetidos y se guardan los manifiestos.',pieces('prepare_data',8,19)+"\ndisplay(pd.crosstab(splits.class_label,splits.split_final,margins=True))")
section(c,'9. Resumen verificable de calidad',
    'Se guardan cantidades de originales, grupos repetidos, conflictos y copias limpias. La falta de IDs de paciente se conserva como limitación explícita.',pieces('prepare_data',19,23)+"\ndisplay(audit)")
section(c,'10. Balance de clases',
    'El gráfico utiliza solo la muestra limpia. Permite ver el desbalance que motiva usar F1 macro y pesos de clase.',pieces('prepare_data',23,29))
section(c,'11. Distribución de resoluciones',
    'Se cuenta cada formato después de depurar. No se eliminan imágenes por ser de un grado o una resolución particular.',pieces('prepare_data',29,35))
section(c,'12. Resolución dentro de cada grado',
    'El mapa de calor muestra el porcentaje de formatos por clase. La asociación entre formato y etiqueta puede actuar como atajo, aun después de estratificar.',pieces('prepare_data',35,45))
c.append(nbformat.v4.new_markdown_cell('## Interpretación\n\nHay 1.576 imágenes únicas y particiones de 1.103/236/237. Cero hashes compartidos no garantiza separación por paciente. Los 30 grupos que cruzaban particiones corresponden al protocolo inicial, anterior a v2.'))
write('01_EDA.ipynb',c)


# 02: full preprocessing and extractor definitions, then all image processing.
c=start('02 · Preprocesamiento y extracción de características',
    'Cada vista ocupa un lienzo de 256 de ancho × 144 de alto conservando proporción. La máscara excluye relleno de estadísticas. '
    'El vector tiene 288 variables y la representación SVD, 2.304 píxeles.')
section(c,'4. Tamaños y configuración del extractor',
    'Los tamaños se expresan como (alto, ancho). La entrada HOG es 96×160 y la representación de píxeles 36×64. Esta configuración también acompaña el modelo de la API.',
    "DEFAULT_VIEW_SIZE=(144,256)\n"+'\n'.join(ast.unparse(n) for n in TREES['features'].body if isinstance(n,ast.Assign)))
section(c,'5. Redimensionamiento proporcional con relleno',
    'Se usa una única escala para ambos ejes, se centra la imagen y se rellena la zona restante. Esto conserva la geometría; no recorta la articulación.',definition('preprocessing','resize_with_padding'))
section(c,'6. Separar panoramas bilaterales',
    'El ancho ≥500 es una heurística del dataset, no un detector anatómico. Produce dos vistas por archivo panorámico, pero la muestra seguirá teniendo una sola etiqueta de estudio.',definition('preprocessing','split_bilateral'))
section(c,'7. CLAHE y máscara geométrica',
    'El código construye la máscara del área válida, aplica CLAHE con clip limit 1,5 y 8×8 divisiones y devuelve el relleno a cero. La máscara no segmenta hueso y CLAHE/HOG aún pueden verse afectados por el fondo.',definition('preprocessing','preprocess_view'))
section(c,'8. HOG: cálculo completo',
    'Se calculan gradientes Sobel, magnitud, ángulo sin signo y ocho bins. La entrada forma 5×3 celdas, ocho bloques de 2×2 y 32 valores por bloque: 256 descriptores. Es una implementación simplificada con normalización L2.',definition('features','compute_hog_features'))
section(c,'9. Intensidad, histogramas, cuadrantes y HOG por vista',
    'Se muestran las 32 estadísticas y los 256 HOG. Las estadísticas usan la máscara; los gradientes excluyen su borde. La salida SVD se redimensiona proporcionalmente a 64×36.',definition('features','extract_single_knee_features'))
section(c,'10. Combinar vistas de un estudio',
    'Se extrae cada vista por separado y se promedian los vectores. Así se conserva una fila por archivo, aunque se pierde lateralidad. El preview concatena vistas únicamente para visualizar.',definition('features','extract_image_features'))
section(c,'11. Verificar geometría y compatibilidad con la API',
    'Se comprueba una imagen de train, se mide el relleno y se compara el extractor visible aquí con el módulo utilizado por la API. Esta comprobación evita que la documentación describa otro preprocesamiento.',
    '''manifest=pd.read_csv(ROOT/'data/dataset_manifest.csv')
sample=manifest[manifest.split_final=='train'].iloc[0]
original=cv2.imread(str(ROOT/sample.clean_path),0)
conditioned,mask=preprocess_view(split_bilateral(original)[0],return_mask=True)
assert conditioned.shape==(144,256) and np.all(conditioned[mask==0]==0)
print(f'Relleno añadido de esta vista: {100*(1-mask.mean()):.2f}%')
local,_,local_flat=extract_image_features(original)
from src.features import extract_image_features as api_extractor
canonical,_,canonical_flat=api_extractor(original)
np.testing.assert_allclose(list(local.values()),list(canonical.values()),rtol=0,atol=0)
np.testing.assert_array_equal(local_flat,canonical_flat)
print('288 variables y extractor coincidente con la API.')
''')
section(c,'12. Recorrer las imágenes limpias',
    'Se leen todas las rutas aceptadas y se extraen descriptores y píxeles. Todavía no se ajusta ningún clasificador. Esta celda puede tardar por el procesamiento de imágenes.',pieces('extract',0,4))
section(c,'13. Construir matrices y comprobar valores',
    'Las listas se convierten a arrays float32. Se verifica dimensión y ausencia de NaN/Infinity antes de utilizarlas en cualquier modelo.',pieces('extract',4,9)+"\nprint('Variables:',X.shape,'| Píxeles:',flat.shape)")
section(c,'14. Guardar train, validación y test',
    'El manifiesto determina a qué partición pertenece cada fila. Se guardan matrices, nombres de variables y configuración; no se aprende ningún escalado de test.',pieces('extract',9,16)+"\ndisplay({k:v.shape for k,v in payload.items() if k.startswith('X_')})")
section(c,'15. Comparación visual con la entrada real',
    'Esta figura se genera con el extractor anterior. El formato rectangular reduce padding y mantiene el contenido completo de las imágenes mostradas.',pieces('extract',16,21))
section(c,'16. Experimento previo de relleno',
    'La tabla histórica compara cuatro representaciones con SVM C=10 y ANOVA160 sobre train/validación. El 4,83 % es relleno añadido medio, no todo el fondo original. Ese experimento no es el modelo final ni prueba que mejoren todas las métricas.',
    "display(pd.read_csv(ROOT/'reports/padding_experiment/validation_comparison.csv'))")
write('02_Preprocesamiento.ipynb',c)


# 03: complete candidate loop and actual baseline fits.
c=start('03 · Baseline y regularización',
    'Se comparan Dummy y logística L2. Se ajustan realmente seis candidatos y se seleccionan usando validación, sin consultar test.',TRAIN_IMPORTS)
training_helpers(c)
section(c,'6. Definir Dummy y los candidatos de logística',
    'Dummy predice la clase mayoritaria. Logística combina StandardScaler y regularización L2, compara cinco valores de C y utiliza pesos balanceados.',pieces('baseline',0,2)+"\nprint('Candidatos:',[item[0] for item in specs])")
section(c,'7. Ajustar y medir el baseline',
    'Se ejecuta el bucle completo mostrado arriba. Cada fila contiene métricas de validación y tiempo; los modelos ajustados se guardan para seleccionar posteriormente.',last_fit('baseline'))
section(c,'8. Interpretar la regularización',
    'Se grafica F1 de validación frente a C. Mayor C significa menor regularización en logística, pero el mejor valor se decide mediante validación y no por el ajuste a train.',
    "linear=table[table.family=='Logística'].copy()\nlinear['C']=linear.Modelo.str.extract(r'C=(.*)').astype(float)\nlinear=linear.sort_values('C')\nplt.figure(figsize=(7,4))\nplt.semilogx(linear.C,linear['F1 Macro'],'o-')\nplt.xlabel('C');plt.ylabel('F1 macro · validación');plt.grid(alpha=.3)\nfigure('baseline_regularizacion_validacion.png')")
write('03_Baseline.ipynb',c)


# 04: source and fits of every advanced family.
c=start('04 · SVM, Random Forest, XGBoost y SVD',
    'Se muestran candidatos, ajustes y algoritmo SVD. La comparación utiliza train/validación; test no participa en estas decisiones.',TRAIN_IMPORTS)
training_helpers(c)
section(c,'6. Configuración de SVM y XGBoost',
    'SVM usa RBF, gamma scale, pesos balanceados y semilla 42. XGBoost usa profundidad 4, learning rate 0,05 y submuestreo 0,8. En cinco clases hay cinco árboles por ronda.',definition('workflow','svc')+definition('workflow','xgb'))
section(c,'7. Candidatos SVM',
    'Se crean tres Pipelines con C=1,10,50. Cada uno contiene su propio StandardScaler, que se ajustará con train.',pieces('advanced',0,2))
section(c,'8. Candidatos de árboles y ensambles',
    'Se agregan 100/200/300 rondas de XGBoost y cuatro configuraciones de Random Forest. El código explicita cantidad de árboles, hoja mínima y pesos del bosque.',pieces('advanced',2,4)+"\nprint('Candidatos:',[item[0] for item in specs])")
section(c,'9. Entrenar SVM y ensambles',
    'Se ajusta cada candidato y se mide F1 en validación. Esta es una celda de entrenamiento real y puede tardar.',pieces('advanced',4,5)+"\ndisplay(table)")
c.append(nbformat.v4.new_markdown_cell('## 10. Algoritmo completo del clasificador SVD\n\n'
    'La clase centra por grado y obtiene una base SVD; decide por menor residuo de proyección. Se incluye completa para estudiar el algoritmo. '
    'La celda de entrenamiento importa esta misma implementación desde `src.models`, de modo que joblib puede cargarla después desde la API sin depender de una clase definida en un notebook.\n\n```python\n'+definition('models','SVDSubspaceClassifier')+'```'))
section(c,'11. Ajustar el escalado y las bases SVD',
    'Se aprende el escalador con píxeles de train y solo se transforma validación. Se calculan bases de hasta 50 componentes por clase para reutilizarlas al probar rangos menores.',pieces('advanced',5,10))
section(c,'12. Comparar rangos del subespacio',
    'Se evalúan k=5,10,20,30,50 sin ajustar con validación. La clase predicha minimiza la norma residual; las probabilidades softmax de este clasificador no son una calibración clínica.',pieces('advanced',10,13))
section(c,'13. Guardar la comparación completa',
    'Se reúnen los quince candidatos avanzados, se ordenan por F1 de validación y se conservan los objetos ajustados para la selección posterior.',pieces('advanced',13,16)+"\ndisplay(table)")
write('04_Modelos_Avanzados.ipynb',c)


# 05: visible ANOVA/PCA pipelines and their actual fits.
c=start('05 · Selección de variables y PCA',
    'Se prueban doce representaciones adicionales. ANOVA usa las etiquetas y PCA conserva variación; ninguno se fuerza en el modelo final.',TRAIN_IMPORTS)
training_helpers(c)
section(c,'6. Estimadores base para comparar representaciones',
    'Se mantienen los mismos parámetros de SVM y XGBoost que en el notebook anterior para que los candidatos tengan una definición reproducible.',definition('workflow','svc')+definition('workflow','xgb'))
section(c,'7. Pipelines con selección ANOVA',
    'SelectKBest conserva 40,80,160 variables según el estadístico F. El selector se ajusta dentro del Pipeline con train; SVM agrega escalado y compara C=10/50.',pieces('feature_engineering',0,2))
section(c,'8. Pipelines con PCA',
    'Se estandariza antes de PCA y se comparan 32,64,128 componentes con SVM C=10. PCA no utiliza etiquetas y varianza retenida no equivale necesariamente a poder predictivo.',pieces('feature_engineering',2,3))
section(c,'9. Ajustar y evaluar representaciones',
    'Se entrena cada Pipeline con train y se mide en validación. No se utiliza test para elegir k ni la cantidad de componentes.',last_fit('feature_engineering'))
section(c,'10. Comparar con las entradas completas',
    'Se agregan los SVM y XGBoost sin reducción del notebook 04. Esto permite comprobar si seleccionar o proyectar supera a conservar las 288 variables.',
    "advanced=pd.DataFrame(json.loads((REPORTS/'advanced_validation.json').read_text(encoding='utf-8')))\ncomparison=pd.concat([table,advanced[advanced.family.isin(['SVM','XGBoost'])]],ignore_index=True).sort_values('F1 Macro',ascending=False)\ndisplay(comparison[['Modelo','F1 Macro','Accuracy (%)']])\nplt.figure(figsize=(10,7))\nplt.barh(comparison.Modelo[::-1],comparison['F1 Macro'][::-1],color='#087f8c')\nplt.xlim(0,1);plt.xlabel('F1 macro · validación');plt.title('Selección ANOVA y reducción PCA')\nfigure('feature_engineering_validation.png')")
c.append(nbformat.v4.new_markdown_cell('## Interpretación\n\nEn la revisión vigente ganó SVM RBF C=1 con las 288 variables. ANOVA y PCA se implementaron y evaluaron; no se eligieron por el criterio de validación.'))
write('05_Feature_Engineering.ipynb',c)


# 06: full selection, re-fit, test metrics, plots and learning-curve code.
c=start('06 · Selección final, evaluación e interpretación',
    'El criterio se congela antes de reevaluar test. El test de 237 imágenes ya se examinó en v2; '
    'esta reevaluación es descriptiva y no una nueva confirmación independiente.',TRAIN_IMPORTS)
section(c,'4. Lectura de matrices y función de métricas',
    'Se comprueba la versión y se definen las mismas métricas utilizadas en los candidatos. El formato tabular tiene 288 entradas y el de píxeles, 2.304.',definition('workflow','arrays')+definition('workflow','score'))
section(c,'5. Reunir los 33 candidatos de validación',
    'Se leen las puntuaciones de los tres notebooks anteriores y sus modelos ajustados. El orden se define por F1 macro de validación, no por test.',pieces('evaluate',0,6)+"\ndisplay(validation[['Modelo','family','F1 Macro']])")
section(c,'6. Congelar la decisión elegida',
    'Se elige la mejor fila global y la mejor por familia, y se guarda la decisión antes de calcular nuevas predicciones de test. Se declara expresamente que el test ya se había examinado en v2.',pieces('evaluate',6,10)+"\nprint('Elegido por validación:',chosen)")
section(c,'7. Reentrenar con train más validación y predecir test',
    'Se clona cada ganador de familia, se reajusta con 1.339 imágenes y se evalúa en las 237 de test. El escalador/selector se reajusta con train+validación y no con test.',pieces('evaluate',10,14))
section(c,'8. Serializar el pipeline final y su configuración',
    'Se guarda el ganador elegido por validación, junto con configuración del extractor y parámetros. El objeto incluye las transformaciones aprendidas; la API comparte el extractor de imágenes.',pieces('evaluate',14,21))
section(c,'9. Métricas globales, por clase y trazabilidad',
    'Se calculan tablas, matriz absoluta, kappa cuadrático y predicciones por imagen. Ordenar la tabla por test es descriptivo: no redefine el modelo seleccionado.',pieces('evaluate',21,28)+"\ndisplay(table)\ndisplay(pd.DataFrame(report).T)")
section(c,'10. Comparación de F1 de test',
    'Se compara la configuración elegida de cada familia. La selección se mantiene por validación aunque otra familia tenga mayor accuracy de test.',pieces('evaluate',28,34))
section(c,'11. Matrices de confusión de todas las familias',
    'Las filas representan clases reales y las columnas, predichas. Los conteos muestran el número de errores en cada grado.',pieces('evaluate',34,39))
section(c,'12. Matriz normalizada del modelo final',
    'Normalizar por fila permite comparar recall entre clases pese al desbalance. La diagonal es la fracción recuperada de cada clase real.',pieces('evaluate',39,43))
section(c,'13. Curvas ROC One-vs-Rest',
    'Se binariza cada grado frente al resto y se recorren umbrales de sus puntuaciones. AUC no sustituye F1 multiclase ni demuestra calibración clínica.',pieces('evaluate',43,52))
section(c,'14. Importancia del XGBoost comparado',
    'La figura corresponde al árbol evaluado y no es una explicación causal del SVM final. Si el árbol utiliza ANOVA, se conserva la correspondencia con sus variables seleccionadas.',pieces('evaluate',52,63))
section(c,'15. Selección ANOVA y PCA: resultados de validación',
    'Estas figuras vuelven a mostrar la comparación de representación usando exclusivamente validación; no se elige la dimensión a partir del resultado de test.',pieces('evaluate',63,77))
section(c,'16. Curvas de aprendizaje: código completo',
    'Se usa StratifiedKFold de tres folds dentro de train, cuatro tamaños y clonación del Pipeline en cada ajuste. Es una celda costosa porque vuelve a entrenar varias veces. La banda es desviación entre folds, no un intervalo de confianza de generalización.',pieces('evaluate',77,84)+"\ndisplay(pd.DataFrame(curves))")
section(c,'17. Comprobar las predicciones publicadas',
    'Se contrasta el resultado reproducido con el protocolo vigente. La comprobación detecta cambios inesperados; no hace una nueva búsqueda de parámetros usando test.',
    f"assert chosen == {M['selected_model']!r}\nassert abs(final_row['F1 Macro']-{M['final_test']['F1 Macro']!r}) < 1e-6\nprint('Reproducido:',chosen,'F1 macro test:',final_row['F1 Macro'])")
c.append(nbformat.v4.new_markdown_cell('## Conclusiones y límites\n\n'
    'El resultado vigente es SVM RBF C=1 con las 288 variables. Menos relleno no mejoró todas las métricas de test. '
    'Hay sobreajuste, ausencia de IDs de paciente y de evaluación externa, posible sesgo de formato y pérdida de lateralidad. '
    'GitHub Pages presenta muestras precalculadas; Streamlit/FastAPI permiten inferencia. No es un sistema validado para diagnóstico.'))
write('06_Comparacion_Final.ipynb',c)


for name,used in USED.items():
    expected={i for i,node in enumerate(NODES['workflow'][name].body)
        if not isinstance(node,ast.Return) and not (isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str))}
    assert used==expected,(name,sorted(expected-used),sorted(used-expected))
print('Cobertura completa: todas las instrucciones de las seis etapas están en celdas visibles.')
