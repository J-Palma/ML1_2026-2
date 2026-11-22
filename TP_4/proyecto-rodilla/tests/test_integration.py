"""Integration checks: clean splits, serialized prediction parity and HTTP errors."""
from pathlib import Path
import sys,json,unittest
import cv2,numpy as np,pandas as pd,joblib
from fastapi.testclient import TestClient
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from api.app import app
from src.inference import predict_image,load_model
class IntegrationTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.client=TestClient(app);cls.manifest=pd.read_csv(ROOT/'data/dataset_manifest.csv');cls.samples=cls.manifest[cls.manifest.split_final=='test'].groupby('target').first().reset_index()
 def test_clean_partition_and_model_size(self):
  self.assertFalse(self.manifest.pixel_sha256.duplicated().any())
  self.assertEqual(len(self.manifest),1576)
  a=np.load(ROOT/'data/processed/features_arrays.npz');self.assertTrue(np.isfinite(a['X_test_tabular']).all())
  metadata=json.loads((ROOT/'models/hyperparameters.json').read_text(encoding='utf-8'))
  model=load_model();self.assertEqual(model.n_features_in_,288);self.assertEqual(metadata['input_features'],288)
  self.assertEqual(str(a['protocol']),metadata['protocol'])
  clean=pd.read_csv(ROOT/'data/clean_manifest.csv');self.assertEqual(len(clean),1576)
  self.assertFalse(clean.pixel_sha256.duplicated().any())
  self.assertEqual(set(clean.pixel_sha256),set(self.manifest.pixel_sha256))
 def test_uploaded_images_match_serialized_evaluation(self):
  a=np.load(ROOT/'data/processed/features_arrays.npz');pred=load_model().predict(a['X_test_tabular']);test=self.manifest[self.manifest.split_final=='test'].reset_index(drop=True)
  for row in self.samples.itertuples():
   p=ROOT/row.path;im=cv2.imread(str(p),0);local=predict_image(im);response=self.client.post('/predict',files={'file':(p.name,p.read_bytes(),'image/png')})
   self.assertEqual(response.status_code,200,response.text);self.assertEqual(response.json()['class_index'],local['class_index']);index=int(test.index[test.path==row.path][0]);self.assertEqual(local['class_index'],int(pred[index]));self.assertAlmostEqual(sum(local['probabilities'].values()),1,places=5)
 def test_invalid_requests_and_health(self):
  self.assertEqual(self.client.get('/health').status_code,200)
  self.assertEqual(self.client.post('/predict').status_code,422)
  self.assertEqual(self.client.post('/predict',files={'file':('x.txt',b'abc','text/plain')}).status_code,415)
  self.assertEqual(self.client.post('/predict',files={'file':('x.png',b'garbage','image/png')}).status_code,400)
  self.assertEqual(self.client.post('/predict',files={'file':('x.png',b'','image/png')}).status_code,413)
  self.assertEqual(self.client.post('/predict',files={'file':('x.png',b'x'*(15*1024*1024+1),'image/png')}).status_code,413)
if __name__=='__main__':unittest.main()
