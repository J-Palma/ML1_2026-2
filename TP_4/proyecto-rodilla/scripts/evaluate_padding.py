"""Isolated padding ablation. Never fits, selects, or evaluates with test rows.

Keeps the published extractor and model unchanged. Run from project root.
"""
from pathlib import Path
import sys
import json
import hashlib
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from src.preprocessing import preprocess_view, resize_with_padding, split_bilateral
from src.features import compute_hog_features

OUT = ROOT / 'reports/padding_experiment'
CONFIGS = [
    ('Cuadrado v2', (256, 256), False, (128, 128)),
    ('Cuadrado con máscara', (256, 256), True, (128, 128)),
    ('Rectangular', (144, 256), False, (96, 160)),
    ('Rectangular con máscara', (144, 256), True, (96, 160)),
]


def view_features(view, target, masked, hog_size):
    enhanced = preprocess_view(view, target)
    h, w = view.shape
    scale = min(target[0] / h, target[1] / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    y, x = (target[0] - nh) // 2, (target[1] - nw) // 2
    mask = np.zeros(target, dtype=np.uint8)
    mask[y:y + nh, x:x + nw] = 1
    pixels = enhanced[mask > 0] if masked else enhanced.flatten()
    values = [np.mean(pixels), np.std(pixels), np.median(pixels),
              np.percentile(pixels, 25), np.percentile(pixels, 75)]
    sx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.hypot(sx, sy)
    lap = cv2.Laplacian(enhanced, cv2.CV_64F)
    # Discard the artificial border from gradient statistics in masked arms.
    inner = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
    g, l = (grad[inner], lap[inner]) if masked else (grad, lap)
    values += [np.mean(g), np.std(g), np.var(l)]
    hist = cv2.calcHist([enhanced], [0], mask if masked else None,
                        [16], [0, 256]).flatten()
    values.extend(hist / (hist.sum() + 1e-6))
    hh, hw = target[0] // 2, target[1] // 2
    for r, c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        q = enhanced[r*hh:(r+1)*hh, c*hw:(c+1)*hw]
        qm = mask[r*hh:(r+1)*hh, c*hw:(c+1)*hw]
        qp = q[qm > 0] if masked else q.flatten()
        if not qp.size:
            raise ValueError('Empty quadrant: review target geometry.')
        values += [np.mean(qp), np.std(qp)]
    if target == (256, 256):
        hog_image = cv2.resize(enhanced, (128, 128), interpolation=cv2.INTER_AREA)
    else:
        # Rectangular HOG also preserves proportions; no square stretching.
        hog_image = resize_with_padding(enhanced, hog_size)
    values.extend(compute_hog_features(hog_image))
    return np.asarray(values, dtype=np.float32), enhanced, 1 - mask.mean()


def extract(image, config):
    _, target, masked, hog_size = config
    items = [view_features(v, target, masked, hog_size)
             for v in split_bilateral(image)]
    return (np.mean(np.stack([v[0] for v in items]), axis=0),
            np.hstack([v[1] for v in items]),
            float(np.mean([v[2] for v in items])))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    production_model = ROOT / 'models/best_model.joblib'
    before = hashlib.sha256(production_model.read_bytes()).hexdigest()
    manifest = pd.read_csv(ROOT / 'data/dataset_manifest.csv')
    # Filter before even reading images or labels for this experiment.
    development = manifest[manifest.split_final.isin(['train', 'validation'])].copy()
    assert len(development) == 1339
    assert set(development.split_final) == {'train', 'validation'}
    images = [cv2.imread(str(ROOT / p), 0) for p in development.path]
    if any(im is None for im in images):
        raise ValueError('Unreadable development image')
    train = development.split_final.eq('train').to_numpy()
    y = development.target.to_numpy()
    rows = []
    # Four fixed representations, one unchanged estimator configuration.
    for config in CONFIGS:
        extracted = [extract(im, config) for im in images]
        X = np.stack([v[0] for v in extracted])
        assert np.isfinite(X).all()
        model = Pipeline([
            ('select', SelectKBest(f_classif, k=160)),
            ('scale', StandardScaler()),
            ('clf', SVC(C=10, kernel='rbf', class_weight='balanced',
                        probability=True, random_state=42)),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            warnings.simplefilter('ignore', RuntimeWarning)
            model.fit(X[train], y[train])
        pred = model.predict(X[~train])
        row = dict(method=config[0], target_height=config[1][0],
                   target_width=config[1][1], masked_statistics=config[2],
                   raw_features=X.shape[1], selected_features=160,
                   artificial_padding_pct=100*np.mean([v[2] for v in extracted]),
                   validation_f1_macro=f1_score(y[~train], pred, average='macro'),
                   validation_accuracy=accuracy_score(y[~train], pred),
                   validation_balanced_accuracy=balanced_accuracy_score(y[~train], pred))
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    # Baseline parity check ensures the experiment reproduces the frozen setup.
    expected = 0.5881108567077029  # Frozen v2 baseline; do not compare to the v3 winner.
    assert abs(rows[0]['validation_f1_macro'] - expected) < 1e-6
    assert hashlib.sha256(production_model.read_bytes()).hexdigest() == before
    result = dict(experiment='padding-ablation-v1', seed=42,
                  train_images=int(train.sum()), validation_images=int((~train).sum()),
                  test_images_read=0, production_model_unchanged=True,
                  configurations=rows,
                  limitations=[
                      'Exploratory reuse of validation; improvement is not independent confirmation.',
                      'Rectangle also changes HOG grid: 320 to 288 descriptors before ANOVA.',
                      'Mask excludes added padding, not the original radiographic background.',
                      'No cropping or anatomical segmentation; no claim of clinical improvement.',
                  ])
    (OUT / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(OUT / 'validation_comparison.csv', index=False)

    selected = development[development.split_final.eq('train')].groupby('target').first().iloc[[0,2,4]]
    fig, axes = plt.subplots(3, 3, figsize=(12, 7), dpi=150)
    names = ['Normal', 'Leve', 'Severa']
    for i, row in enumerate(selected.itertuples()):
        im = cv2.imread(str(ROOT / row.path), 0)
        _, square, p1 = extract(im, CONFIGS[0])
        _, rect, p2 = extract(im, CONFIGS[3])
        for j, (img, title) in enumerate([
                (im, f'Original: {names[i]}'),
                (square, f'Cuadrado: {100*p1:.1f}% relleno añadido'),
                (rect, f'Rectángulo: {100*p2:.1f}% relleno añadido')]):
            axes[i,j].imshow(img, cmap='gray', vmin=0, vmax=255)
            axes[i,j].set_title(title, fontsize=10)
            axes[i,j].axis('off')
    fig.suptitle('Comparación experimental: misma imagen completa, sin estirar ni recortar', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / 'comparison.png', bbox_inches='tight')
    plt.close(fig)
    print('Experiment saved to', OUT, flush=True)


if __name__ == '__main__':
    main()
