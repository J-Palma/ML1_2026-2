"""Feature extraction shared by notebooks, the prediction API, and dashboard."""

from __future__ import annotations

import cv2
import numpy as np

from src.preprocessing import preprocess_view, split_bilateral


def compute_hog_features(gray_img: np.ndarray, cell_size: int = 32, n_bins: int = 8) -> np.ndarray:
    """Compute the project's normalized 2x2-cell Dalal-Triggs HOG descriptor."""
    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle %= 180.0
    bin_width = 180.0 / n_bins
    bin_index = np.floor(angle / bin_width).astype(np.int32) % n_bins
    n_cells_y = gray_img.shape[0] // cell_size
    n_cells_x = gray_img.shape[1] // cell_size
    hist_grid = np.zeros((n_cells_y, n_cells_x, n_bins), dtype=np.float32)

    for y in range(n_cells_y):
        for x in range(n_cells_x):
            ys = slice(y * cell_size, (y + 1) * cell_size)
            xs = slice(x * cell_size, (x + 1) * cell_size)
            cell_mag = magnitude[ys, xs]
            cell_bin = bin_index[ys, xs]
            for bin_id in range(n_bins):
                hist_grid[y, x, bin_id] = np.sum(cell_mag[cell_bin == bin_id])

    blocks = []
    for y in range(n_cells_y - 1):
        for x in range(n_cells_x - 1):
            block = hist_grid[y:y + 2, x:x + 2, :].flatten()
            blocks.append(block / (np.linalg.norm(block) + 1e-6))
    return np.concatenate(blocks) if blocks else hist_grid.flatten()


def extract_single_knee_features(image_gray: np.ndarray, target_size=(256, 256)) -> tuple[dict, np.ndarray, np.ndarray]:
    """Return one ordered 320-feature dictionary, preview, and flat pixels."""
    enhanced = preprocess_view(image_gray, target_size)
    pixels = enhanced.astype(np.float32)

    features = {
        "intensity_mean": float(np.mean(pixels)),
        "intensity_std": float(np.std(pixels)),
        "intensity_median": float(np.median(pixels)),
        "intensity_p25": float(np.percentile(pixels, 25)),
        "intensity_p75": float(np.percentile(pixels, 75)),
    }

    hist = cv2.calcHist([enhanced], [0], None, [16], [0, 256]).flatten()
    hist /= np.sum(hist) + 1e-6

    sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    features.update({
        "grad_mean": float(np.mean(grad_mag)),
        "grad_std": float(np.std(grad_mag)),
        "laplacian_var": float(cv2.Laplacian(enhanced, cv2.CV_64F).var()),
    })

    features.update({f"hist_bin_{i}": float(hist[i]) for i in range(16)})

    half_h, half_w = target_size[0] // 2, target_size[1] // 2
    quadrant_names = ("top_left", "top_right", "bottom_left", "bottom_right")
    quadrant_coords = ((0, 0), (0, 1), (1, 0), (1, 1))
    for name, (row, col) in zip(quadrant_names, quadrant_coords):
        quadrant = enhanced[row * half_h:(row + 1) * half_h, col * half_w:(col + 1) * half_w]
        features[f"{name}_mean"] = float(np.mean(quadrant))
        features[f"{name}_std"] = float(np.std(quadrant))

    gray_128 = cv2.resize(enhanced, (128, 128), interpolation=cv2.INTER_AREA)
    hog = compute_hog_features(gray_128, cell_size=32, n_bins=8)
    features.update({f"hog_{i}": float(value) for i, value in enumerate(hog)})
    if len(features) != 320:
        raise ValueError(f"Expected 320 features, got {len(features)}")
    img_64 = cv2.resize(enhanced, (64, 64), interpolation=cv2.INTER_AREA)
    flat = (img_64.astype(np.float32) / 255.0).flatten()
    return features, enhanced, flat


def extract_image_features(image_gray: np.ndarray, target_size=(256, 256)) -> tuple[dict, np.ndarray, np.ndarray]:
    """Extract features, preview, and flat pixels; average bilateral views."""
    view_features = []
    flat_features = []
    conditioned_views = []
    for view in split_bilateral(image_gray):
        feature_dict, conditioned, flat = extract_single_knee_features(view, target_size)
        view_features.append(feature_dict)
        flat_features.append(flat)
        conditioned_views.append(conditioned)

    if len(view_features) == 1:
        averaged_features = view_features[0]
    else:
        averaged_features = {
            name: float((view_features[0][name] + view_features[1][name]) / 2.0)
            for name in view_features[0]
        }
    flat_average = np.mean(np.stack(flat_features), axis=0, dtype=np.float32)
    preview = np.hstack(conditioned_views) if len(conditioned_views) == 2 else conditioned_views[0]
    return averaged_features, preview, flat_average
