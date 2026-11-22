"""Audit decoded pixels and materialize a clean copy without altering raw data."""
from pathlib import Path
import hashlib
import shutil
import cv2
import pandas as pd

LABELS=['Normal','Dudosa','Leve','Moderada','Severa']


def pixel_digest(image):
    return hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()


def audit_dataset(project_root):
    root=Path(project_root);rows=[]
    for p in sorted((root/'data/raw/Training').rglob('*')):
        if p.suffix.lower() not in {'.png','.jpg','.jpeg'}: continue
        image=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
        if image is None: raise ValueError(f'Imagen ilegible: {p}')
        target=int(p.parent.name[0]);h,w=image.shape
        rows.append(dict(path=p.relative_to(root).as_posix(),class_name=p.parent.name,
            class_label=LABELS[target],target=target,width=w,height=h,resolution=f'{w} x {h}',
            pixel_sha256=pixel_digest(image)))
    df=pd.DataFrame(rows)
    if df.empty: raise FileNotFoundError('Colocá el dataset en data/raw/Training/<clase>/*.png')
    conflicts=df.groupby('pixel_sha256').target.nunique()
    bad=set(conflicts[conflicts>1].index)
    df['audit_status']='included'
    df.loc[df.pixel_sha256.isin(bad),'audit_status']='conflicting_label'
    eligible=df[~df.pixel_sha256.isin(bad)]
    df.loc[eligible.index[eligible.duplicated('pixel_sha256',keep='first')],'audit_status']='duplicate_copy'
    canonical=df[df.audit_status=='included'].set_index('pixel_sha256').path.to_dict()
    df['canonical_path']=df.pixel_sha256.map(canonical).fillna('')
    return df


def materialize_clean_dataset(inventory, project_root):
    """Create only accepted files, named by pixel hash. Raw files are read-only."""
    root=Path(project_root).resolve();destination=root/'data/clean/Training'
    rows=[]
    for record in inventory[inventory.audit_status=='included'].itertuples():
        source=root/record.path
        target=destination/record.class_name/(record.pixel_sha256+source.suffix.lower())
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            existing=cv2.imread(str(target),0)
            if existing is None or pixel_digest(existing)!=record.pixel_sha256:
                raise ValueError(f'La copia limpia no coincide: {target}')
        else: shutil.copy2(source,target)
        rows.append(dict(original_path=record.path,clean_path=target.relative_to(root).as_posix(),
            target=record.target,class_name=record.class_name,pixel_sha256=record.pixel_sha256))
    clean=pd.DataFrame(rows)
    expected=set(clean.clean_path)
    actual={p.relative_to(root).as_posix() for p in destination.rglob('*') if p.is_file()}
    if actual!=expected: raise ValueError('data/clean contiene archivos inesperados; revisar antes de usarla.')
    assert not clean.pixel_sha256.duplicated().any()
    clean.to_csv(root/'data/clean_manifest.csv',index=False)
    return clean
