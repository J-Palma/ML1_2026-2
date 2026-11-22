"""Interactive report and local/API prediction demonstration."""
from pathlib import Path
import sys,json,os
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2,numpy as np,pandas as pd,plotly.express as px,requests,streamlit as st
from src.inference import predict_image
st.set_page_config(page_title='Artrosis de rodilla · ML1 FIUNA',page_icon='🦴',layout='wide')
@st.cache_data
def read_json(name):return json.loads((ROOT/'reports'/name).read_text(encoding='utf-8'))
m=read_json('modelos_avanzados_metrics.json');d=read_json('dataset_summary.json')
st.title('Clasificación de artrosis de rodilla')
st.caption('Jorge Javier Palma Martínez · jpalma@fiuna.edu.py · Machine Learning 1 · FIUNA · 2026')
st.info('Prototipo académico. No usar para diagnóstico ni decisiones clínicas.')
cols=st.columns(4)
for c,label,value in zip(cols,['Imágenes únicas','Test reservado','F1 macro final','Accuracy de test'],[d['total_images'],m['test_size'],f"{m['final_test']['F1 Macro']:.3f}",f"{m['final_test']['Accuracy (%)']:.2f} %"]):c.metric(label,value)
st.write('**Modelo elegido por validación:**',m['selected_model'])
a,b,c,e=st.tabs(['Comparación','Características y auditoría','Probar radiografía','Método y límites'])
with a:
 frame=pd.DataFrame(m['comparison_table']);key=st.selectbox('Métrica',['F1 Macro','Accuracy (%)','Balanced Acc (%)'])
 st.plotly_chart(px.bar(frame.sort_values(key,ascending=False),x='Modelo',y=key,color='family',text=key),width='stretch')
 st.dataframe(frame,hide_index=True,width='stretch')
 st.caption('Selección por validación. Reevaluación del mismo test de v2; no es una nueva confirmación independiente.')
 for name,caption in [('final_confusion_normalizada.png','Matriz normalizada del modelo final'),('modelos_avanzados_learning_curves.png','Entrenamiento y CV estratificada de 3 folds'),('modelos_avanzados_roc_curves.png','ROC multiclase')]:st.image(str(ROOT/'reports/figures'/name),caption=caption,width='stretch')
with b:
 audit=d['audit'];st.subheader('Duplicados y etiquetas')
 st.write(f"De {d['raw_images']} originales se excluyeron {audit['excluded_conflicting_images']} imágenes con etiquetas contradictorias y {audit['removed_duplicate_copies']} copias exactas. Quedaron {d['total_images']} imágenes únicas.")
 st.warning('Sin IDs de paciente: no garantiza independencia por persona ni elimina duplicados aproximados.')
 st.dataframe(pd.DataFrame(d['classes']),hide_index=True,width='stretch')
 st.subheader('Selección ANOVA y reducción PCA')
 st.dataframe(pd.read_csv(ROOT/'reports/validation_comparison.csv'),hide_index=True,width='stretch')
 st.image(str(ROOT/'reports/figures/feature_engineering_validation.png'),width='stretch')
 st.image(str(ROOT/'reports/figures/modelos_avanzados_feature_importance.png'),caption='Importancia del XGBoost evaluado; no es una explicación causal.',width='stretch')
with c:
 st.subheader('Predicción sobre una imagen')
 st.write('Usá una radiografía del mismo tipo que el dataset. El modelo siempre asigna un grado: no comprueba que sea una rodilla.')
 mode=st.radio('Ejecutar mediante',['Modelo local','API HTTP'],horizontal=True)
 uploaded=st.file_uploader('Seleccionar radiografía',type=['png','jpg','jpeg','bmp','tif','tiff'])
 if uploaded:
  st.image(uploaded,caption='Imagen seleccionada',width=500)
  if st.button('Clasificar imagen',type='primary'):
   try:
    content=uploaded.getvalue()
    if mode=='Modelo local':result=predict_image(cv2.imdecode(np.frombuffer(content,dtype=np.uint8),cv2.IMREAD_GRAYSCALE))
    else:
     url=os.getenv('KNEE_API_URL','http://127.0.0.1:8000').rstrip('/')
     response=requests.post(url+'/predict',files={'file':(uploaded.name,content,uploaded.type)},timeout=90);response.raise_for_status();result=response.json()
    st.success(f"Grado estimado: {result['class_index']} · {result['class_name']}")
    st.plotly_chart(px.bar(x=list(result['probabilities']),y=list(result['probabilities'].values()),labels={'x':'Grado','y':'Probabilidad del modelo'},range_y=[0,1]),width='stretch')
    st.caption(result['probability_note'])
   except (ValueError,requests.RequestException,cv2.error) as exc:st.error(f'No se pudo procesar: {exc}')
with e:
 st.markdown('''1. Auditoría de píxeles por SHA-256: excluir conflictos de etiqueta y copias exactas.
2. Partición 70/15/15 estratificada por grado y resolución, semilla 42.
3. Extracción de 288 descriptores; entrada 256×144, estadísticas sin relleno, CLAHE y promedio bilateral.
4. Ajuste, selección ANOVA y PCA exclusivamente con train/validation.
5. Reentrenamiento con train + validation y evaluación en test reservado.

**Límites:** mismo test ya evaluado en v2; esta reevaluación es descriptiva. Conjunto pequeño, una etiqueta por estudio bilateral, sin IDs de pacientes ni validación externa. Kappa cuadrático describe errores ordinales, pero los clasificadores entrenan como multiclase nominal. Probabilidades sin calibración clínica.''')
 st.link_button('Repositorio y documentación','https://github.com/J-Palma/ML1_2026-2/tree/main/TP_4')
