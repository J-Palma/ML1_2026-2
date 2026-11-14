"""Export TP4 figures from existing partitions and trained model; no retraining."""
from pathlib import Path
import argparse
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, accuracy_score

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--processed',type=Path,default=root/'data/processed');parser.add_argument('--site',type=Path,default=root.parents[1]/'docs');args=parser.parse_args()
assets=args.site/'assets';assets.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(args.processed/'dataset_splits.csv')
labels=['Normal','Dudosa','Leve','Moderada','Severa']
fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
counts=df['resolution'].value_counts().sort_index(); ax.bar(counts.index,counts.values,color=['#007c83','#d68a22'])
for i,c in enumerate(counts.values):ax.text(i,c+15,str(c),ha='center')
ax.set(title='Resoluciones originales del dataset',ylabel='Cantidad de imágenes');ax.set_ylim(0,1650)
fig.savefig(assets/'eda_resoluciones.png',dpi=180);plt.close(fig)
pct=pd.crosstab(df.target,df.resolution,normalize='index').sort_index()*100
fig,ax=plt.subplots(figsize=(8,4.7),layout='constrained');im=ax.imshow(pct.values,cmap='Blues',vmin=0,vmax=100,aspect='auto');fig.colorbar(im,ax=ax,label='% dentro de cada grado')
ax.set_xticks(range(len(pct.columns)),pct.columns);ax.set_yticks(range(5),labels);ax.set_title('Resolución por grado de artrosis')
for i in range(5):
 for j in range(len(pct.columns)):ax.text(j,i,f'{pct.iloc[i,j]:.1f} %',ha='center',va='center',color='white' if pct.iloc[i,j]>55 else '#102a43')
fig.savefig(assets/'eda_resolucion_por_grado.png',dpi=180);plt.close(fig)
arrays=np.load(args.processed/'features_arrays.npz');model=joblib.load(root/'models/xgboost_forest.joblib');pred=model.predict(arrays['X_test_tabular']);y=arrays['y_test']
fig,ax=plt.subplots(figsize=(7,6),layout='constrained');ConfusionMatrixDisplay.from_predictions(y,pred,display_labels=labels,normalize='true',values_format='.2f',cmap='Blues',ax=ax,colorbar=False);ax.set(title='XGBoost: matriz normalizada por grado real (test)',xlabel='Grado predicho',ylabel='Grado real');fig.savefig(assets/'xgb_confusion_normalizada.png',dpi=180);plt.close(fig)
print('EDA:',counts.to_dict(),'F1 XGBoost:',round(f1_score(y,pred,average='macro'),4),'accuracy',round(accuracy_score(y,pred)*100,2))
