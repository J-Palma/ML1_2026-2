"""Generate six self-contained narrative notebooks, backed by shared source code."""
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[1]
setup='''from pathlib import Path
import sys,json
import pandas as pd
from IPython.display import display,Image
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'src/workflow.py').exists())
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src import workflow
# False: show the executed experiment exports; True: recompute this stage.
RECOMPUTE=False
'''
def img(name):return f"display(Image(filename=str(ROOT/'reports/figures/{name}')))"
def load(name):return f"pd.DataFrame(json.loads((ROOT/'reports/{name}').read_text(encoding='utf-8')))"
specs=[
('01_EDA.ipynb','01 · Datos, auditoría y EDA','''Se parte de 1.650 radiografías. La auditoría SHA-256 de píxeles encontró 63 grupos repetidos y 3 grupos con etiquetas contradictorias. Se excluyen todas las 6 imágenes contradictorias y 68 copias; quedan 1.576 imágenes únicas. Se conserva una copia por hash y los originales no se modifican. No hay IDs de pacientes, por lo que este control no detecta todos los recortes o vistas relacionadas.

Las nuevas particiones 70/15/15 se estratifican por clase y resolución con semilla 42: 1.103 train, 236 validation, 237 test. Las cifras del protocolo anterior quedan reemplazadas.''','prepare_data',"display(pd.read_csv(ROOT/'data/dataset_manifest.csv').groupby(['class_label','split_final']).size().unstack())\ndisplay(json.loads((ROOT/'reports/data_quality_audit.json').read_text(encoding='utf-8')))",['distribucion_clases.png','eda_resoluciones.png','eda_resolucion_por_grado.png']),
('02_Preprocesamiento.ipynb','02 · Preprocesamiento y 320 descriptores','''Cada vista pasa por escala de grises, ajuste proporcional con padding a 256 × 256 y CLAHE (clip limit 1,5). La regla de ancho ≥500 divide estudios bilaterales por la mitad; es específica del dataset y no detecta anatomía. Se promedian los descriptores de ambas vistas, manteniendo una fila y una etiqueta por estudio.

Vector: 5 estadísticas de intensidad, 3 de gradiente/Laplaciano, 16 bins de histograma, 8 estadísticas por cuadrantes y 288 HOG. El percentil 25 es constante en train por el fondo negro; los descriptores globales incluyen padding. SVD usa además 64 × 64 píxeles. No se imputan radiografías ilegibles: la extracción detiene la ejecución. No se eliminan atípicos por grado o resolución.''','extract',"import numpy as np\na=np.load(ROOT/'data/processed/features_arrays.npz')\ndisplay({k:v.shape for k,v in a.items() if k.startswith('X_')})\ndisplay(pd.read_csv(ROOT/'data/processed/features_dataset.csv').iloc[:5,:12])",['acondicionamiento_radiografias.png']),
('03_Baseline.ipynb','03 · Baseline y regularización','''Dummy mayoritaria aporta la referencia trivial. Regresión logística con L2 y class_weight balanced se ajusta en un Pipeline con StandardScaler. C se barre en {0.001,0.01,0.1,1,10}. El escalador y clasificador se ajustan únicamente con train; la puntuación de selección es F1 macro en validación fija. No se consulta test en esta etapa.''','baseline','display('+load('baseline_validation.json')+')',[]),
('04_Modelos_Avanzados.ipynb','04 · SVM, Random Forest, XGBoost y SVD','''Se comparan SVM RBF (C=1,10,50), Random Forest (100/200/300 árboles, hoja 1/2), XGBoost (100/200/300 rondas, profundidad 4, learning rate 0.05) y SVD por clase (k=5,10,20,30,50). SVM y SVD incluyen estandarización. Los parámetros se seleccionan por F1 macro de validación.

XGBoost multiclase utiliza cinco árboles por ronda: 300 rondas no significan 300 árboles, sino 1.500. SVD centra por clase y proyecta sobre su base truncada; predice por menor norma residual. Sus probabilidades softmax son aproximadas y no calibradas.''','advanced','display('+load('advanced_validation.json')+')',[]),
('05_Feature_Engineering.ipynb','05 · Selección ANOVA y reducción PCA','''SelectKBest con ANOVA compara 40,80,160 descriptores. PCA compara 32,64,128 componentes después de StandardScaler. Cada transformación se ajusta con train dentro del Pipeline; validación elige la alternativa. La columna constante no aporta discriminación y sklearn la descarta del ranking útil. Esta selección no utiliza importancias calculadas con test.

Se prueban SVM con selección (C=10/50), XGBoost con selección (200 rondas) y SVM con PCA (C=10). Las representaciones se comparan en validación; no se afirma que reducir siempre mejore. La selección final fue SVM ANOVA160 C=10, frente a 320 entradas originales.''','feature_engineering','display('+load('feature_validation.json')+')',['feature_engineering_validation.png']),
('06_Comparacion_Final.ipynb','06 · Modelo final, test, interpretación y despliegue','''La decisión final se congela en selection_decision.json usando todos los candidatos de validación. Se reentrena la configuración elegida de cada familia con train+validation, y después se calculan las métricas del test reservado. El modelo final serializa selección, escalado y clasificador juntos, evitando diferencias entre notebook y API.

Las curvas de aprendizaje usan 3 folds estratificados dentro de train: son diagnósticas posteriores a la búsqueda y no validación anidada independiente. La importancia mostrada pertenece al XGBoost comparado y no explica causalmente al SVM final. Se informa kappa cuadrático como métrica ordinal adicional.

Los límites incluyen falta de IDs de paciente, posibles recortes relacionados, formato asociado al grado, promedio bilateral y ausencia de validación externa/clínica. Las métricas viejas no se comparan directamente con las actuales. La API y el dashboard son una demostración académica.''','evaluate',"display(pd.read_csv(ROOT/'reports/comparacion_modelos.csv'))\ndisplay(json.loads((ROOT/'reports/modelos_avanzados_metrics.json').read_text(encoding='utf-8'))['classification_report'])",['comparacion_modelos.png','final_confusion_normalizada.png','modelos_avanzados_roc_curves.png','modelos_avanzados_learning_curves.png','modelos_avanzados_feature_importance.png'])]
for filename,title,body,stage,view,images in specs:
 cells=[nbformat.v4.new_markdown_cell('# '+title+'\n\n**Jorge Javier Palma Martínez · ML1 FIUNA · 2026**\n\nProtocolo: `deduplicated-v2-2026-11-22`.\n\n'+body),nbformat.v4.new_code_cell(setup),nbformat.v4.new_markdown_cell('## Ejecutar o consultar la etapa\n\nLos resultados mostrados se cargan de los artefactos calculados por `python scripts/run_pipeline.py`. Para reconstruir, ejecutar los seis notebooks en orden con `RECOMPUTE=True`; para leer la entrega, dejarlo en `False`. Las implementaciones completas están en `src/workflow.py`.'),nbformat.v4.new_code_cell(f'if RECOMPUTE:\n    workflow.{stage}()\n'+view)]
 for name in images:cells +=[nbformat.v4.new_code_cell(img(name))]
 n=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (proyecto-rodilla)','language':'python','name':'python3'}});nbformat.write(n,ROOT/'notebooks'/filename)
old=ROOT/'notebooks/05_Resultados.ipynb'
if old.exists() and (ROOT/'scratch/antes_correccion_2026-11-22/notebooks/05_Resultados.ipynb').exists():old.unlink()
print('Seis notebooks estructurados, con fuente compartida y ejecución reproducible.')
