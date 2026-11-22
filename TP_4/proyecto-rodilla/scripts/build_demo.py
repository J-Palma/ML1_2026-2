from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,pandas as pd
from src.inference import predict_image
R=Path(__file__).resolve().parents[1];df=pd.read_csv(R/'data/dataset_manifest.csv');out=[];folder=R/'docs/assets/demo';folder.mkdir(parents=True,exist_ok=True)
for target in range(5):
 for num,row in enumerate(df[(df.split_final=='test')&(df.target==target)].head(2).itertuples()):
  image=cv2.imread(str(R/row.path),0);name=f'grado-{target}-ejemplo-{num+1}.png';cv2.imwrite(str(folder/name),image)
  out.append(dict(id=f'{target}-{num}',image='assets/demo/'+name,actual=int(target),actual_name=row.class_label,**predict_image(image)))
(R/'docs/data/demo_predictions.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print('Demo:',len(out),'muestras de test.')
