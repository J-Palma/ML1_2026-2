"""Image preparation helpers for the knee osteoarthritis project."""

import cv2
import numpy as np


def resize_with_padding(image, target_size=(256, 256), interpolation=cv2.INTER_AREA):
    """Resize an image to a fixed canvas without changing its aspect ratio."""
    target_h, target_w = target_size
    height, width = image.shape[:2]
    scale = min(target_w / width, target_h / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    canvas = np.zeros((target_h, target_w), dtype=image.dtype)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def split_bilateral(image):
    """Split panoramas into left and right knee views; leave single views intact."""
    _, width = image.shape[:2]
    if width >= 500:
        middle = width // 2
        return image[:, :middle], image[:, middle:]
    return (image,)


def preprocess_view(image, target_size=(256, 256)):
    """Preserve anatomy on a padded canvas and enhance local contrast."""
    target_h, target_w = target_size
    source_h, source_w = image.shape[:2]
    scale = min(target_w / source_w, target_h / source_h)
    new_w, new_h = max(1, int(round(source_w * scale))), max(1, int(round(source_h * scale)))
    x0, y0 = (target_w - new_w) // 2, (target_h - new_h) // 2
    mask = np.zeros((target_h, target_w), dtype=np.uint8)
    mask[y0:y0 + new_h, x0:x0 + new_w] = 1
    fitted = resize_with_padding(image, target_size)
    # A moderate limit improves local contrast without making bone texture noisy.
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(fitted)
    # Keep padding black after CLAHE, so it does not acquire artificial texture.
    enhanced[mask == 0] = 0
    return enhanced


def preprocess_knee_image(image_gray, target_size=(256, 256)):
    """Return view(s) on a canvas shaped like the source, avoiding square padding."""
    views = split_bilateral(image_gray)
    # Keep the source aspect ratio for the demonstration; bilateral panels are
    # shown as two consecutive views at their natural aspect ratio.
    if target_size is None:
        conditioned = [preprocess_view(view, view.shape[:2]) for view in views]
    else:
        conditioned = [preprocess_view(view, target_size) for view in views]
    return np.hstack(conditioned) if len(conditioned) == 2 else conditioned[0]

