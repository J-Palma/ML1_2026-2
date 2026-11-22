"""Build the complete executed narrative, including explicit data cleaning."""
from pathlib import Path
import json
import nbformat
ROOT=Path(__file__).resolve().parents[1]
M=json.loads((ROOT/'reports/modelos_avanzados_metrics.json').read_text(encoding='utf-8'))
F=M['final_test'];P=M['protocol'];N=M['input_features'];K=M['feature_count']
setup="""from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from IPython.display import display,Image
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'src/workflow.py').exists())
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src import workflow
# True reconstruye esta etapa; False permite leer los resultados exportados.
RECOMPUTE=False
"""
md=nbformat.v4.new_markdown_cell
code=nbformat.v4.new_code_cell

def image(name):return code(f"display(Image(filename=str(ROOT/'reports/figures/{name}')))")
def header(title,body):return [md('# '+title+'\n\n**Jorge Javier Palma Martínez · ML1 FIUNA · 2026**\n\nProtocolo: `'+P+'`.\n\n'+body),code(setup)]
def write(name,cells):
 n=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (proyecto-rodilla)','language':'python','name':'python3'}})
 nbformat.write(n,ROOT/'notebooks'/name)

def stage(title,body,method,view,images):
 cells=header(title,body)
 cells += [md('## Ejecutar o consultar\n\nPara recalcular, poner `RECOMPUTE=True` y ejecutar los notebooks en orden. Cada etapa usa funciones compartidas con la API; los resultados exportados corresponden al pipeline real.'),code(f'if RECOMPUTE:\n    workflow.{method}()\n'+view)]
 cells += [image(name) for name in images]
 return cells

cells=header('01 · Limpieza del dataset, auditoría y EDA',
 'Partimos de 1.650 radiografías del dataset público de Kaggle, distribuidas en cinco grados. Antes de dividir los datos hay que eliminar copias exactas de la muestra de trabajo y separar etiquetas contradictorias. La carpeta original `data/raw/Training` se conserva intacta. El resultado materializado se guarda en `data/clean/Training`.')
cells += [md('## 1. Inventariar las imágenes originales\n\nLa identidad se calcula con SHA-256 de la **forma y los píxeles decodificados en escala de grises**. Así se detecta la misma radiografía aunque tenga otro nombre o compresión PNG. No se decide por el nombre del archivo ni por la resolución. La implementación está en `src/data_cleaning.py`.'),code("""from src.data_cleaning import audit_dataset, materialize_clean_dataset
inventory = audit_dataset(ROOT) if RECOMPUTE else pd.read_csv(ROOT/'data/image_inventory.csv', keep_default_na=False)
print('Imágenes originales:',len(inventory))
display(inventory.groupby('class_label').size().rename('originales').to_frame())
display(inventory[['path','class_label','width','height','pixel_sha256']].head())""")]
cells += [md('## 2. Detectar repeticiones y conflictos de etiqueta\n\nUna repetición con la misma etiqueta permite conservar una sola copia. Si los mismos píxeles aparecen con grados diferentes, se excluye el grupo completo: no tenemos una revisión clínica que permita elegir la etiqueta correcta.'),code("""groups=inventory.groupby('pixel_sha256').agg(copies=('path','size'),different_labels=('target','nunique'))
repeated=groups[groups.copies>1]
conflicts=groups[groups.different_labels>1]
print('Grupos repetidos:',len(repeated))
print('Grupos con etiquetas contradictorias:',len(conflicts))
display(inventory[inventory.pixel_sha256.isin(conflicts.index)][['path','class_label','pixel_sha256','audit_status']])
display(inventory[inventory.audit_status=='duplicate_copy'][['path','canonical_path','class_label']].head(8))""")]
cells += [md('## 3. Aplicar la limpieza y generar una copia sin repetidos\n\nSe excluyen **6 imágenes** de 3 grupos contradictorios y **68 copias** adicionales de grupos consistentes. Quedan **1.576 imágenes únicas**. La copia limpia usa nombres basados en el hash; `data/clean_manifest.csv` relaciona cada copia con su archivo original. `data/excluded_images.csv` registra cada exclusión y su motivo. No se borra ninguna radiografía original.'),code("""if RECOMPUTE:
    workflow.prepare_data()  # auditoría + copia limpia + particiones deterministas
accepted=inventory[inventory.audit_status=='included'].copy()
excluded=inventory[inventory.audit_status!='included'].copy()
assert not accepted.pixel_sha256.duplicated().any()
assert len(accepted)+len(excluded)==len(inventory)
clean_manifest=pd.read_csv(ROOT/'data/clean_manifest.csv')
assert set(clean_manifest.pixel_sha256)==set(accepted.pixel_sha256)
assert not clean_manifest.pixel_sha256.duplicated().any()
print('Originales:',len(inventory),'| Aceptadas:',len(accepted),'| Excluidas:',len(excluded))
display(inventory.audit_status.value_counts().rename('cantidad').to_frame())
display(clean_manifest[['original_path','clean_path','class_name']].head())
if (ROOT/'data/clean/Training').exists():
    files=[p for p in (ROOT/'data/clean/Training').rglob('*') if p.is_file()]
    assert len(files)==len(accepted)
    print('Archivos presentes en data/clean/Training:',len(files))
else:
    print('Las imágenes no se versionan; ejecutar con RECOMPUTE=True y el dataset descargado para generarlas.')""")]
cells += [md('## 4. Particionar después de limpiar\n\nPartición 70/15/15 con semilla 42, estratificada conjuntamente por grado y resolución. Mantiene las mismas 1.103 imágenes de train, 236 de validación y 237 de test del protocolo depurado anterior. El entrenamiento lee `clean_path`, no toda la carpeta original. Verificamos que ningún hash exacto esté repetido entre particiones.'),code("""manifest=pd.read_csv(ROOT/'data/dataset_manifest.csv')
assert not manifest.pixel_sha256.duplicated().any()
assert set(manifest.pixel_sha256)==set(accepted.pixel_sha256)
assert manifest.groupby('pixel_sha256').split_final.nunique().max()==1
display(pd.crosstab(manifest.class_label,manifest.split_final,margins=True))
display(pd.crosstab(manifest.resolution,manifest.split_final,margins=True))
display(json.loads((ROOT/'reports/data_quality_audit.json').read_text(encoding='utf-8')))""")]
cells += [md('## 5. EDA del conjunto limpio\n\nSe conserva el desbalance de clases y la asociación entre resolución y grado. La limpieza exacta no identifica pacientes, recortes ni imágenes casi iguales; no garantiza independencia por persona. Las 30 coincidencias que cruzaban particiones históricas justificaron abandonar aquella partición inicial.')]
cells += [image(x) for x in ['distribucion_clases.png','eda_resoluciones.png','eda_resolucion_por_grado.png']]
write('01_EDA.ipynb',cells)

cells=stage('02 · Preprocesamiento rectangular y máscara',
 'Cada vista se ajusta proporcionalmente a **256 de ancho × 144 de alto**, sin recortar ni estirar. CLAHE usa clip limit 1,5 y cuadrícula 8 × 8. El ancho original ≥500 divide las vistas bilaterales; es una regla específica del dataset, no una detección anatómica. Se promedian los descriptores por estudio.\n\nLa máscara identifica únicamente el relleno añadido y lo excluye de intensidad, histogramas, cuadrantes y estadísticas de gradientes. Se excluye también el borde artificial del cálculo de gradientes. El fondo presente en la radiografía original permanece. HOG conserva proporción con una entrada 160×96 y aporta 256 descriptores; se suman 32 estadísticas: **288 variables**. SVD usa 64×36 píxeles (2.304 valores), también con proporción conservada.',
 'extract',"""a=np.load(ROOT/'data/processed/features_arrays.npz')
assert str(a['protocol'])==workflow.PROTOCOL
display({k:v.shape for k,v in a.items() if k.startswith('X_')})
display(json.loads((ROOT/'reports/preprocessing_spec.json').read_text(encoding='utf-8')))
display(pd.read_csv(ROOT/'data/processed/features_dataset.csv').iloc[:5,:12])""",['acondicionamiento_radiografias.png'])
cells += [md('## Verificar la máscara sobre una imagen de entrenamiento\n\nLa máscara conserva los píxeles fuente, incluso si son negros. No es una segmentación del hueso.'),code("""import cv2
from src.preprocessing import preprocess_view,split_bilateral
from src.features import extract_image_features
manifest=pd.read_csv(ROOT/'data/dataset_manifest.csv')
row=manifest[manifest.split_final=='train'].iloc[0]
original=cv2.imread(str(ROOT/row.clean_path),0)
view=split_bilateral(original)[0]
enhanced,mask=preprocess_view(view,return_mask=True)
features,_,_=extract_image_features(original)
assert enhanced.shape==(144,256)
assert np.all(enhanced[mask==0]==0)
assert len(features)==288
print(f'Relleno añadido en esta vista: {100*(1-mask.mean()):.2f}%')
print('Variables:',len(features),'| Media sin relleno:',features['intensity_mean'])""")]
cells += [md('## Experimento que motivó la elección\n\nAntes de reentrenar se compararon cuatro representaciones con SVM C=10 y ANOVA160, usando únicamente train/validación: el relleno medio bajó de 46,47 % a 4,83 % y F1 de validación pasó de 0,5881 a 0,6031. La cuadrícula HOG también cambia (320 a 288 variables), por lo que no se atribuye el efecto solo al fondo. Es una exploración sobre validación reutilizada, no confirmación independiente.'),code("display(pd.read_csv(ROOT/'reports/padding_experiment/validation_comparison.csv'))")]
write('02_Preprocesamiento.ipynb',cells)
write('03_Baseline.ipynb',stage('03 · Baseline y regularización',
 'Dummy mayoritaria es la referencia trivial. Logística L2 compara C={0.001,0.01,0.1,1,10}, con pesos balanceados y StandardScaler dentro del Pipeline. El ajuste usa train y la elección usa F1 macro de validación; no se elige por test.',
 'baseline',"display(pd.DataFrame(json.loads((ROOT/'reports/baseline_validation.json').read_text(encoding='utf-8'))))",[]))
write('04_Modelos_Avanzados.ipynb',stage('04 · Modelos avanzados',
 'SVM RBF (C=1,10,50), Random Forest (100/200/300 árboles, hoja mínima 1/2), XGBoost (100/200/300 rondas, profundidad 4) y SVD por subespacios (k=5,10,20,30,50). Las representaciones y parámetros se ajustan con train y se comparan por validación. XGBoost multiclase crea cinco árboles por ronda: 300 rondas equivalen a 1.500 árboles. SVD predice por menor norma residual y sus probabilidades son aproximadas.',
 'advanced',"display(pd.DataFrame(json.loads((ROOT/'reports/advanced_validation.json').read_text(encoding='utf-8'))))",[]))
write('05_Feature_Engineering.ipynb',stage('05 · Selección ANOVA y reducción PCA',
 f'ANOVA compara 40, 80 y 160 variables de las {N} iniciales; PCA compara 32, 64 y 128 componentes estandarizados. Cada transformación se ajusta dentro del Pipeline, solo con train. Se compara SVM (C=10/50) y XGBoost con ANOVA, y SVM C=10 con PCA. La configuración final elegida por validación es **{M["selected_model"]}**. No se presupone que reducir siempre mejore.',
 'feature_engineering',"display(pd.DataFrame(json.loads((ROOT/'reports/feature_validation.json').read_text(encoding='utf-8'))))",['feature_engineering_validation.png']))
write('06_Comparacion_Final.ipynb',stage('06 · Reentrenamiento, evaluación e inferencia',
 f'La configuración elegida es **{M["selected_model"]}**, con F1 macro de validación {F["validation_f1"]:.4f}. Se reentrena con 1.339 imágenes (train+validation). Modelo, extractor, API y dashboard comparten la misma versión. Accuracy de test {F["Accuracy (%)"]:.2f} %, F1 macro {F["F1 Macro"]:.4f}.\n\n**Test reutilizado:** las mismas 237 imágenes ya fueron evaluadas en la versión anterior. Esta reevaluación es descriptiva; no equivale a un nuevo test independiente ni permite confirmar una mejora generalizable. La elección de la representación y los parámetros de esta revisión usa validación. Las curvas de aprendizaje de 3 folds son diagnósticas posteriores a la búsqueda.\n\nLa importancia mostrada corresponde al XGBoost comparado; no demuestra causalidad ni explica automáticamente otro modelo final. Persisten la falta de IDs de paciente, posibles recortes relacionados, sesgo de formato y ausencia de validación externa o clínica.',
 'evaluate',"""display(json.loads((ROOT/'reports/selection_decision.json').read_text(encoding='utf-8')))
display(pd.read_csv(ROOT/'reports/comparacion_modelos.csv'))
display(json.loads((ROOT/'reports/modelos_avanzados_metrics.json').read_text(encoding='utf-8'))['classification_report'])""",['comparacion_modelos.png','final_confusion_normalizada.png','modelos_avanzados_roc_curves.png','modelos_avanzados_learning_curves.png','modelos_avanzados_feature_importance.png']))
print('Seis notebooks generados con limpieza explícita y preprocesamiento rectangular.')
