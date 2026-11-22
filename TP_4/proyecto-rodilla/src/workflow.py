"""Six-stage academic ML experiment. Fit/select without consulting test labels."""
from pathlib import Path
import hashlib
import json
import time
import warnings
import copy
import cv2
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc, cohen_kappa_score)
from xgboost import XGBClassifier
from src.features import extract_image_features
from src.models import SVDSubspaceClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed'
REPORTS = ROOT/'reports'
FIGURES = REPORTS/'figures'
MODELS = ROOT/'models'
LABELS = ['Normal', 'Dudosa', 'Leve', 'Moderada', 'Severa']
SEED = 42
PROTOCOL = 'deduplicated-v2-2026-11-22'
for folder in [DATA, REPORTS, FIGURES, MODELS]: folder.mkdir(parents=True, exist_ok=True)


def save_json(path, value):
    def finite(item):
        if isinstance(item, dict): return {k: finite(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)): return [finite(v) for v in item]
        if isinstance(item, np.generic): item = item.item()
        if isinstance(item, float) and not np.isfinite(item): return None
        return item
    Path(path).write_text(json.dumps(finite(value), ensure_ascii=False, indent=2,
        allow_nan=False, default=str), encoding='utf-8')


def figure(name):
    plt.savefig(FIGURES/name, dpi=180, bbox_inches='tight')
    plt.close()


def prepare_data():
    """Audit every raw image, quarantine conflicts, deduplicate and split anew."""
    rows = []
    for p in sorted((ROOT/'data/raw/Training').rglob('*')):
        if p.suffix.lower() not in {'.png', '.jpg', '.jpeg'}: continue
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None: raise ValueError(f'Imagen ilegible: {p}')
        target = int(p.parent.name[0]); h,w = im.shape
        digest = hashlib.sha256(str(im.shape).encode()+im.tobytes()).hexdigest()
        rows.append(dict(path=p.relative_to(ROOT).as_posix(), class_name=p.parent.name,
            class_label=LABELS[target], target=target, width=w, height=h,
            resolution=f'{w} x {h}', pixel_sha256=digest))
    df = pd.DataFrame(rows)
    if df.empty: raise FileNotFoundError('Colocá el dataset en data/raw/Training/<clase>/*.png')
    conflicts = df.groupby('pixel_sha256').target.nunique()
    bad = set(conflicts[conflicts>1].index)
    df['audit_status'] = 'included'
    df.loc[df.pixel_sha256.isin(bad),'audit_status']='conflicting_label'
    eligible = df[~df.pixel_sha256.isin(bad)]
    repeated = eligible.index[eligible.duplicated('pixel_sha256', keep='first')]
    df.loc[repeated,'audit_status']='duplicate_copy'
    clean = df[df.audit_status=='included'].copy()
    strata = clean.class_name.astype(str)+'_'+clean.resolution
    train,temp = train_test_split(clean,test_size=.30,stratify=strata,random_state=SEED)
    validation,test = train_test_split(temp,test_size=.50,stratify=temp.class_name.astype(str)+'_'+temp.resolution,random_state=SEED)
    parts=[]
    for name,part in [('train',train),('validation',validation),('test',test)]:
        part=part.copy();part['split_final']=name;parts.append(part)
    splits = pd.concat(parts,ignore_index=True)
    assert not splits.pixel_sha256.duplicated().any()
    assert set(splits.target)==set(range(5))
    df.to_csv(ROOT/'data/image_inventory.csv',index=False)
    splits.to_csv(DATA/'dataset_splits.csv',index=False)
    # Portable, public manifest contains relative paths and fingerprints, not pixels.
    splits.to_csv(ROOT/'data/dataset_manifest.csv',index=False)
    audit=dict(protocol=PROTOCOL,raw_images=len(df),unique_eligible_images=len(clean),
        duplicate_groups=int((df.groupby('pixel_sha256').size()>1).sum()),
        conflicting_label_groups=len(bad),excluded_conflicting_images=int((df.audit_status=='conflicting_label').sum()),
        removed_duplicate_copies=int((df.audit_status=='duplicate_copy').sum()),
        cross_split_exact_duplicates=0,patient_identifiers_available=False,
        historical_cross_split_duplicate_groups=30,
        limitation='Sin identificadores de paciente; no descarta duplicados aproximados, recortes o vistas del mismo paciente.')
    save_json(REPORTS/'data_quality_audit.json',audit)
    summary=dict(protocol=PROTOCOL,total_images=len(clean),raw_images=len(df),
        splits=splits.split_final.value_counts().to_dict(),resolutions=clean.resolution.value_counts().to_dict(),
        classes=[dict(target=i,name=LABELS[i],count=int((clean.target==i).sum())) for i in range(5)],audit=audit)
    save_json(REPORTS/'dataset_summary.json',summary)
    plt.figure(figsize=(8,4));counts=clean.target.value_counts().sort_index()
    plt.bar(LABELS,counts,color='#087f8c');plt.ylabel('Imágenes únicas');plt.title('Dataset depurado: distribución por grado');figure('distribucion_clases.png')
    plt.figure(figsize=(8,4));counts=clean.resolution.value_counts().sort_index()
    plt.bar(counts.index,counts,color=['#087f8c','#df9d3e']);plt.ylabel('Imágenes únicas');plt.title('Resoluciones originales tras depuración');figure('eda_resoluciones.png')
    matrix=pd.crosstab(clean.target,clean.resolution,normalize='index')*100
    fig,ax=plt.subplots(figsize=(7,4));ax.imshow(matrix,cmap='Blues',vmin=0,vmax=100,aspect='auto');ax.set_xticks(range(len(matrix.columns)),matrix.columns);ax.set_yticks(range(5),LABELS)
    for i in range(5):
        for j in range(len(matrix.columns)):
            v=matrix.iloc[i,j];ax.text(j,i,f'{v:.1f} %',ha='center',va='center',color='white' if v>55 else '#102a43')
    ax.set_title('Resolución por grado: % dentro de cada clase');figure('eda_resolucion_por_grado.png')
    print(json.dumps(audit,ensure_ascii=False));print(pd.crosstab(splits.class_label,splits.split_final))
    return splits


def extract():
    manifest=pd.read_csv(DATA/'dataset_splits.csv');features=[];flats=[]
    for r in manifest.itertuples():
        f,_,flat=extract_image_features(cv2.imread(str(ROOT/r.path),0));features.append(list(f.values()));flats.append(flat)
    X=np.asarray(features,dtype=np.float32);flat=np.asarray(flats,dtype=np.float32);y=manifest.target.to_numpy()
    names=list(f)
    assert X.shape==(len(manifest),320) and np.isfinite(X).all() and np.isfinite(flat).all()
    payload={'feature_names':np.array(names)}
    for label,key in [('train','train'),('validation','val'),('test','test')]:
        mask=(manifest.split_final==label).to_numpy();payload[f'X_{key}_tabular']=X[mask];payload[f'X_{key}_flat']=flat[mask];payload[f'y_{key}']=y[mask]
    np.savez_compressed(DATA/'features_arrays.npz',**payload)
    save_json(DATA/'feature_names.json',names)
    out=pd.DataFrame(X,columns=names);pd.concat([manifest,out],axis=1).to_csv(DATA/'features_dataset.csv',index=False)
    fig,axes=plt.subplots(3,2,figsize=(10,7))
    for i,row in enumerate(manifest.groupby('target').first().reset_index().iloc[[0,2,4]].itertuples()):
        image=cv2.imread(str(ROOT/row.path),0);_,preview,_=extract_image_features(image)
        axes[i,0].imshow(image,cmap='gray',vmin=0,vmax=255);axes[i,1].imshow(preview,cmap='gray',vmin=0,vmax=255)
        axes[i,0].set_title(f'Original: {LABELS[row.target]}');axes[i,1].set_title('Entrada del extractor: proporción + CLAHE')
        axes[i,0].axis('off');axes[i,1].axis('off')
    plt.tight_layout();figure('acondicionamiento_radiografias.png')
    print('Matrices:',{k:list(v.shape) for k,v in payload.items() if k.startswith('X_')})
    return pd.DataFrame(X,columns=names)


def arrays(): return np.load(DATA/'features_arrays.npz')


def svc(C=10): return SVC(C=C,kernel='rbf',class_weight='balanced',probability=True,random_state=SEED)


def xgb(n=200): return XGBClassifier(n_estimators=n,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,eval_metric='mlogloss',random_state=SEED,n_jobs=2)


def score(y,pred):
    return {'Accuracy (%)':round(accuracy_score(y,pred)*100,4),'Balanced Acc (%)':round(balanced_accuracy_score(y,pred)*100,4),
        'F1 Macro':round(f1_score(y,pred,average='macro',zero_division=0),6),
        'F1 Weighted':round(f1_score(y,pred,average='weighted',zero_division=0),6),
        'Precision Macro':round(precision_score(y,pred,average='macro',zero_division=0),6),
        'Recall Macro':round(recall_score(y,pred,average='macro',zero_division=0),6)}


def fit_candidates(specs,filename):
    a=arrays();rows=[];fitted={}
    for name,family,kind,model in specs:
        X=a[f'X_train_{kind}'];V=a[f'X_val_{kind}'];t=time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=FutureWarning);model.fit(X,a['y_train'])
        metrics=score(a['y_val'],model.predict(V))
        rows.append(dict(Modelo=name,family=family,input_kind=kind,**metrics,seconds=round(time.perf_counter()-t,2)))
        fitted[name]=model
        print(name,metrics['F1 Macro'],flush=True)
    table=pd.DataFrame(rows).sort_values('F1 Macro',ascending=False,kind='stable')
    save_json(REPORTS/filename,table.to_dict(orient='records'))
    joblib.dump(fitted,MODELS/(filename.replace('.json','_candidates.joblib')))
    return table


def baseline():
    specs=[('Dummy mayoritaria','Dummy','tabular',DummyClassifier(strategy='most_frequent'))]
    for C in [.001,.01,.1,1.,10.]:
        specs.append((f'Logística C={C:g}','Logística','tabular',Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(C=C,class_weight='balanced',max_iter=3000,random_state=SEED))])))
    return fit_candidates(specs,'baseline_validation.json')


def advanced():
    specs=[]
    for C in [1.,10.,50.]:specs.append((f'SVM RBF C={C:g}','SVM','tabular',Pipeline([('scale',StandardScaler()),('clf',svc(C))])))
    for n in [100,200,300]:specs.append((f'XGBoost {n} rondas','XGBoost','tabular',xgb(n)))
    for n,leaf in [(100,1),(200,1),(200,2),(300,2)]:
        specs.append((f'Random Forest n={n} hoja={leaf}','Random Forest','tabular',RandomForestClassifier(n_estimators=n,min_samples_leaf=leaf,class_weight='balanced_subsample',n_jobs=2,random_state=SEED)))
    table=fit_candidates(specs,'advanced_validation.json')
    # Reuse the largest bases for rank validation; fit on train only.
    a=arrays(); scaler=StandardScaler();X=scaler.fit_transform(a['X_train_flat']);V=scaler.transform(a['X_val_flat']);model=SVDSubspaceClassifier(50).fit(X,a['y_train'])
    rows=table.to_dict(orient='records');saved=joblib.load(MODELS/'advanced_validation_candidates.joblib')
    for k in [5,10,20,30,50]:
        model.k_components=k;name=f'SVD k={k}'
        rows.append(dict(Modelo=name,family='SVD',input_kind='flat',**score(a['y_val'],model.predict(V))))
        saved[name]=Pipeline([('scale',copy.deepcopy(scaler)),('clf',copy.deepcopy(model))])
    table=pd.DataFrame(rows).sort_values('F1 Macro',ascending=False,kind='stable')
    save_json(REPORTS/'advanced_validation.json',table.to_dict(orient='records'));joblib.dump(saved,MODELS/'advanced_validation_candidates.joblib')
    return table


def feature_engineering():
    specs=[]
    for k in [40,80,160]:
        for C in [10.,50.]:
            specs.append((f'SVM ANOVA{k} C={C:g}','SVM','tabular',Pipeline([('select',SelectKBest(f_classif,k=k)),('scale',StandardScaler()),('clf',svc(C))])))
        specs.append((f'XGBoost ANOVA{k}','XGBoost','tabular',Pipeline([('select',SelectKBest(f_classif,k=k)),('clf',xgb(200))])))
    for n in [32,64,128]:
        specs.append((f'SVM PCA{n} C=10','SVM','tabular',Pipeline([('scale',StandardScaler()),('pca',PCA(n_components=n,random_state=SEED)),('clf',svc(10))])))
    return fit_candidates(specs,'feature_validation.json')


def evaluate():
    a=arrays();rows=[];models={}
    for file in ['baseline_validation','advanced_validation','feature_validation']:
        rows+=json.loads((REPORTS/(file+'.json')).read_text(encoding='utf-8'))
        models.update(joblib.load(MODELS/(file+'_candidates.joblib')))
    validation=pd.DataFrame(rows).sort_values('F1 Macro',ascending=False,kind='stable')
    validation.to_csv(REPORTS/'validation_comparison.csv',index=False)
    winner=validation.iloc[0];chosen=winner.Modelo
    # Freeze this decision BEFORE reading test predictions/metrics.
    save_json(REPORTS/'selection_decision.json',dict(protocol=PROTOCOL,selected_model=chosen,selection_metric='F1 macro validation',validation_f1=float(winner['F1 Macro']),test_used_for_selection=False))
    family_winners=validation.drop_duplicates('family')
    test_rows=[];predictions={};final_models={}
    for row in family_winners.itertuples():
        model=clone(models[row.Modelo]);kind=row.input_kind
        X=np.vstack([a[f'X_train_{kind}'],a[f'X_val_{kind}']]);y=np.r_[a['y_train'],a['y_val']]
        with warnings.catch_warnings():warnings.filterwarnings('ignore',category=FutureWarning);model.fit(X,y)
        pred=model.predict(a[f'X_test_{kind}']);prob=model.predict_proba(a[f'X_test_{kind}'])
        entry=dict(Modelo=row.Modelo,family=row.family,validation_f1=float(getattr(row,'_3',0)) if False else float(validation[validation.Modelo==row.Modelo]['F1 Macro'].iloc[0]),**score(a['y_test'],pred))
        test_rows.append(entry);predictions[row.Modelo]=(pred,prob);final_models[row.Modelo]=model
        print('TEST',row.Modelo,entry['F1 Macro'],flush=True)
    table=pd.DataFrame(test_rows).sort_values('F1 Macro',ascending=False)
    final=final_models[chosen]
    joblib.dump(final,MODELS/'best_model.joblib')
    selected_features=a['feature_names'].tolist()
    if isinstance(final,Pipeline) and 'select' in final.named_steps:selected_features=np.array(selected_features)[final.named_steps['select'].get_support()].tolist()
    hyper=dict(protocol=PROTOCOL,name=chosen,input_features=320,selected_features=selected_features,
        output_features=int(final.named_steps['pca'].n_components_) if isinstance(final,Pipeline) and 'pca' in final.named_steps else len(selected_features),
        parameters={k:v for k,v in final.get_params().items() if isinstance(v,(str,int,float,bool,type(None)))},selection='F1 macro en validación fija; test no usado para elegir')
    save_json(MODELS/'hyperparameters.json',hyper)
    final_row=next(x for x in test_rows if x['Modelo']==chosen);final_pred,final_prob=predictions[chosen]
    report=classification_report(a['y_test'],final_pred,labels=range(5),target_names=LABELS,output_dict=True,zero_division=0)
    table.to_csv(REPORTS/'comparacion_modelos.csv',index=False)
    metrics=dict(project='Clasificación de artrosis de rodilla · ML1 FIUNA',protocol=PROTOCOL,
        selected_model=chosen,best_model=chosen,best_model_f1_macro=final_row['F1 Macro'],
        selection_rule='Mayor F1 macro en validación; selección congelada antes del test',comparison_table=test_rows,
        final_test=final_row,classification_report=report,confusion_matrix=confusion_matrix(a['y_test'],final_pred).tolist(),
        quadratic_kappa=float(cohen_kappa_score(a['y_test'],final_pred,weights='quadratic')),test_size=len(a['y_test']),
        best_observed_test_model=str(table.iloc[0].Modelo),feature_count=hyper['output_features'],input_features=320,
        audit=json.loads((REPORTS/'data_quality_audit.json').read_text(encoding='utf-8')))
    save_json(REPORTS/'modelos_avanzados_metrics.json',metrics)
    save_json(REPORTS/'per_sample_predictions.json',[dict(index=int(i),actual=int(y),predicted=int(p),probabilities=probs.tolist()) for i,(y,p,probs) in enumerate(zip(a['y_test'],final_pred,final_prob))])
    plt.figure(figsize=(9,4));plt.barh(table.Modelo[::-1],table['F1 Macro'][::-1],color='#087f8c');plt.xlim(0,1);plt.xlabel('F1 macro · test');plt.title('Modelos elegidos por familia usando validación');figure('comparacion_modelos.png')
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,(name,(pred,_)) in zip(axes.flat,predictions.items()):
        ConfusionMatrixDisplay.from_predictions(a['y_test'],pred,display_labels=LABELS,cmap='Blues',ax=ax,colorbar=False);ax.set_title(name,fontsize=9)
    for ax in list(axes.flat)[len(predictions):]:ax.axis('off')
    plt.tight_layout();figure('modelos_avanzados_confusion_matrices.png')
    fig,ax=plt.subplots(figsize=(7,6));ConfusionMatrixDisplay.from_predictions(a['y_test'],final_pred,display_labels=LABELS,normalize='true',values_format='.2f',cmap='Blues',ax=ax,colorbar=False);ax.set_title('Modelo final: proporciones por grado real');figure('final_confusion_normalizada.png')
    binary=label_binarize(a['y_test'],classes=range(5));plt.figure(figsize=(8,5))
    for i,label in enumerate(LABELS):
        fpr,tpr,_=roc_curve(binary[:,i],final_prob[:,i]);plt.plot(fpr,tpr,label=f'{label}: AUC={auc(fpr,tpr):.3f}')
    plt.plot([0,1],[0,1],'k--');plt.xlabel('Tasa de falsos positivos');plt.ylabel('Tasa de verdaderos positivos');plt.title('Modelo final: ROC One-vs-Rest');plt.legend();figure('modelos_avanzados_roc_curves.png')
    # Explain an evaluated tree model even when the validation-selected final is SVM.
    xgb_name=family_winners[family_winners.family=='XGBoost'].iloc[0].Modelo
    tree=final_models[xgb_name];names=a['feature_names']
    if isinstance(tree,Pipeline):
        if 'select' in tree.named_steps:names=names[tree.named_steps['select'].get_support()]
        tree=tree.named_steps['clf']
    importance=tree.feature_importances_;order=np.argsort(importance)[-20:]
    plt.figure(figsize=(9,6));plt.barh(np.array(names)[order],importance[order],color='#087f8c');plt.xlabel('Importancia interna');plt.title(f'Interpretación del XGBoost evaluado: {xgb_name}');figure('modelos_avanzados_feature_importance.png')
    # Feature engineering comparison is VALIDATION ONLY, never selected by test.
    feat=validation[validation.family.isin(['SVM','XGBoost'])]
    plt.figure(figsize=(10,7));plt.barh(feat.Modelo[::-1],feat['F1 Macro'][::-1],color='#087f8c');plt.xlabel('F1 macro · validación');plt.xlim(0,1);plt.title('Selección ANOVA y reducción PCA');figure('feature_engineering_validation.png')
    curves=[];fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    curve_names=[chosen,xgb_name] if chosen!=xgb_name else [chosen,family_winners[family_winners.family=='SVM'].iloc[0].Modelo]
    for ax,name in zip(axes,curve_names):
        kind=validation[validation.Modelo==name].iloc[0].input_kind
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=FutureWarning)
            n,tr,va=learning_curve(clone(models[name]),a[f'X_train_{kind}'],a['y_train'],train_sizes=[.25,.5,.75,1.],cv=StratifiedKFold(3,shuffle=True,random_state=SEED),scoring='f1_macro',n_jobs=1)
        for values,label,color in [(tr,'Entrenamiento','#087f8c'),(va,'CV estratificada','#df9d3e')]:
            mean=values.mean(axis=1);std=values.std(axis=1);ax.plot(n,mean,'o-',label=label,color=color);ax.fill_between(n,mean-std,mean+std,alpha=.15,color=color)
        ax.set(title=name,xlabel='Muestras de entrenamiento',ylabel='F1 macro',ylim=(0,1));ax.legend(fontsize=8);ax.grid(alpha=.2)
        curves.append(dict(model=name,train_f1=float(tr.mean(axis=1)[-1]),cv_f1=float(va.mean(axis=1)[-1])))
    plt.tight_layout();figure('modelos_avanzados_learning_curves.png');save_json(REPORTS/'learning_curves_summary.json',curves)
    return table
