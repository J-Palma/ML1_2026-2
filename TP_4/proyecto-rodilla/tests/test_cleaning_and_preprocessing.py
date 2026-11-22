"""Check label conflict quarantine, actual clean files, and padding exclusion."""
from pathlib import Path
import hashlib
import sys
import tempfile
import unittest
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data_cleaning import audit_dataset,materialize_clean_dataset
from src.preprocessing import preprocess_view
from src.features import extract_single_knee_features


class CleaningAndPreprocessingTests(unittest.TestCase):
    def test_decoded_duplicates_conflicts_and_non_destructive_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            examples=[('0Normal/a.png',50,0),('0Normal/b.png',50,9),
                      ('0Normal/c.png',90,0),('1Doubtful/a.png',90,9),('2Mild/a.png',130,0)]
            for name,value,level in examples:
                path=root/'data/raw/Training'/name;path.parent.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(path),np.full((10,16),value,np.uint8),[cv2.IMWRITE_PNG_COMPRESSION,level])
            before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'data/raw').rglob('*.png')}
            inventory=audit_dataset(root)
            self.assertEqual(inventory.audit_status.value_counts().to_dict(),
                {'included':2,'conflicting_label':2,'duplicate_copy':1})
            clean=materialize_clean_dataset(inventory,root)
            self.assertEqual(len(clean),2)
            self.assertEqual(len(list((root/'data/clean').rglob('*.png'))),2)
            self.assertEqual(before,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before})
            self.assertEqual(set(clean.target),{0,2})
            # Re-running is idempotent; no extra copies appear.
            self.assertEqual(len(materialize_clean_dataset(inventory,root)),2)

    def test_rectangular_field_and_statistics_exclude_only_added_padding(self):
        image=np.full((162,300),100,np.uint8)
        # Preserve genuine black source pixels; do not confuse them with padding.
        image[:,:20]=0
        enhanced,mask=preprocess_view(image,return_mask=True)
        self.assertEqual(enhanced.shape,(144,256))
        self.assertLess(1-mask.mean(),.05)
        self.assertGreater(mask.mean(),.95)
        self.assertTrue(np.all(enhanced[mask==0]==0))
        features,preview,flat=extract_single_knee_features(image)
        self.assertEqual(len(features),288)
        self.assertEqual(flat.shape,(2304,))
        self.assertTrue(np.isfinite(list(features.values())).all())
        self.assertAlmostEqual(features['intensity_mean'],float(enhanced[mask>0].mean()),places=6)
        self.assertGreater(features['intensity_mean'],float(enhanced.mean()))


if __name__=='__main__':unittest.main()
