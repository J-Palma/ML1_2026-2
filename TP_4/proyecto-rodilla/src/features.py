"""Rectangular, geometry-preserving descriptors shared by training and inference."""
from __future__ import annotations
import cv2
import numpy as np
from src.preprocessing import DEFAULT_VIEW_SIZE, preprocess_view, resize_with_padding, split_bilateral

FEATURE_COUNT = 288
HOG_SIZE = (96, 160)
FLAT_SIZE = (36, 64)
PREPROCESSING_SPEC = dict(version='rectangular-masked-v3', view_size=list(DEFAULT_VIEW_SIZE),
    hog_size=list(HOG_SIZE), flat_size=list(FLAT_SIZE), masked_statistics=True,
    clahe_clip_limit=1.5, bilateral_width_threshold=500)


def compute_hog_features(gray_img: np.ndarray, cell_size: int = 32, n_bins: int = 8) -> np.ndarray:
    """Normalized 2x2-cell HOG; 96x160 gives 256 descriptors with 32px cells."""
    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle %= 180.0
    bins = np.floor(angle / (180.0 / n_bins)).astype(np.int32) % n_bins
    ny, nx = gray_img.shape[0] // cell_size, gray_img.shape[1] // cell_size
    hist_grid = np.zeros((ny, nx, n_bins), dtype=np.float32)
    for y in range(ny):
        for x in range(nx):
            ys, xs = slice(y*cell_size, (y+1)*cell_size), slice(x*cell_size, (x+1)*cell_size)
            for b in range(n_bins):
                hist_grid[y,x,b] = np.sum(magnitude[ys,xs][bins[ys,xs] == b])
    blocks=[]
    for y in range(ny-1):
        for x in range(nx-1):
            block=hist_grid[y:y+2,x:x+2,:].flatten()
            blocks.append(block / (np.linalg.norm(block)+1e-6))
    return np.concatenate(blocks) if blocks else hist_grid.flatten()


def extract_single_knee_features(image_gray: np.ndarray, target_size=DEFAULT_VIEW_SIZE):
    """Exclude added padding from statistics; preserve the whole source field."""
    enhanced, mask = preprocess_view(image_gray, target_size, return_mask=True)
    pixels = enhanced[mask > 0]
    features = dict(intensity_mean=float(np.mean(pixels)), intensity_std=float(np.std(pixels)),
        intensity_median=float(np.median(pixels)), intensity_p25=float(np.percentile(pixels,25)),
        intensity_p75=float(np.percentile(pixels,75)))
    sx=cv2.Sobel(enhanced,cv2.CV_64F,1,0,ksize=3)
    sy=cv2.Sobel(enhanced,cv2.CV_64F,0,1,ksize=3)
    inner=cv2.erode(mask,np.ones((3,3),np.uint8)) > 0
    grad=np.hypot(sx,sy)[inner]
    lap=cv2.Laplacian(enhanced,cv2.CV_64F)[inner]
    features.update(grad_mean=float(np.mean(grad)),grad_std=float(np.std(grad)),laplacian_var=float(np.var(lap)))
    hist=cv2.calcHist([enhanced],[0],mask,[16],[0,256]).flatten()
    hist/=hist.sum()+1e-6
    features.update({f'hist_bin_{i}':float(v) for i,v in enumerate(hist)})
    hh, hw=target_size[0]//2,target_size[1]//2
    for name,(r,c) in zip(['top_left','top_right','bottom_left','bottom_right'],[(0,0),(0,1),(1,0),(1,1)]):
        q=enhanced[r*hh:(r+1)*hh,c*hw:(c+1)*hw]
        qm=mask[r*hh:(r+1)*hh,c*hw:(c+1)*hw]
        values=q[qm > 0]
        if not values.size: raise ValueError('El cuadrante no contiene píxeles de la imagen.')
        features[f'{name}_mean']=float(np.mean(values));features[f'{name}_std']=float(np.std(values))
    hog_image=resize_with_padding(enhanced,HOG_SIZE)
    features.update({f'hog_{i}':float(v) for i,v in enumerate(compute_hog_features(hog_image))})
    if len(features)!=FEATURE_COUNT: raise ValueError(f'Expected {FEATURE_COUNT} features, got {len(features)}')
    flat=(resize_with_padding(enhanced,FLAT_SIZE).astype(np.float32)/255.0).flatten()
    return features, enhanced, flat


def extract_image_features(image_gray: np.ndarray, target_size=DEFAULT_VIEW_SIZE):
    """Average descriptors per study; never propagate study labels to new rows."""
    items=[extract_single_knee_features(view,target_size) for view in split_bilateral(image_gray)]
    names=list(items[0][0])
    values=np.mean(np.array([list(x[0].values()) for x in items]),axis=0)
    features={name:float(v) for name,v in zip(names,values)}
    preview=np.hstack([x[1] for x in items])
    flat=np.mean(np.stack([x[2] for x in items]),axis=0,dtype=np.float32)
    return features, preview, flat
