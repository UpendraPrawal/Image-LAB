import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
from PIL import Image


def encode_image(img_array, is_bgr=True):
    """Convert numpy array to base64 PNG string."""
    if is_bgr and len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def encode_figure():
    """Encode current matplotlib figure to base64."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#0d1117', dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')
    return encoded


# ─── 1. IMAGE READING AND DISPLAY ─────────────────────────────────────────────

def read_and_display(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return {
        'original': encode_image(img_bgr),
        'rgb': encode_image(img_rgb, is_bgr=False),
        'info': {
            'shape_bgr': str(img_bgr.shape),
            'shape_rgb': str(img_rgb.shape),
            'dtype': str(img_bgr.dtype),
        }
    }


# ─── 2. IMAGE PROPERTIES ──────────────────────────────────────────────────────

def image_properties(img_bgr):
    rows, cols = img_bgr.shape[:2]
    channels = img_bgr.shape[2] if len(img_bgr.shape) == 3 else 1
    total_pixels = rows * cols
    total_rgb = total_pixels * channels
    sample_pixel = img_bgr[rows // 2, cols // 2].tolist()
    return {
        'original': encode_image(img_bgr),
        'info': {
            'rows': rows,
            'cols': cols,
            'channels': channels,
            'total_pixels': f'{total_pixels:,}',
            'total_values': f'{total_rgb:,}',
            'resolution': f'{cols} × {rows}',
            'sample_pixel_bgr': str(sample_pixel),
            'sample_pixel_rgb': str(list(reversed(sample_pixel))),
            'dtype': str(img_bgr.dtype),
            'size_kb': f'{img_bgr.nbytes / 1024:.1f}',
        }
    }


# ─── 3. COLOR AND INTENSITY ANALYSIS ──────────────────────────────────────────

def color_intensity_analysis(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img_bgr)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor('#0d1117')

    # Grayscale
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title('Grayscale', color='#58d9f5', fontsize=11)
    axes[0].axis('off')

    for ax, channel, cmap, label in zip(
        axes[1:], [r, g, b],
        ['Reds_r', 'Greens_r', 'Blues_r'],
        ['Red Channel', 'Green Channel', 'Blue Channel']
    ):
        ax.imshow(channel, cmap=cmap)
        ax.set_title(label, color='#58d9f5', fontsize=11)
        ax.axis('off')

    for ax in axes:
        ax.set_facecolor('#0d1117')

    fig_encoded = encode_figure()

    return {
        'original': encode_image(img_bgr),
        'analysis_plot': fig_encoded,
        'info': {
            'gray_mean': f'{gray.mean():.2f}',
            'gray_std': f'{gray.std():.2f}',
            'r_mean': f'{r.mean():.2f}',
            'g_mean': f'{g.mean():.2f}',
            'b_mean': f'{b.mean():.2f}',
            'r_max': int(r.max()),
            'g_max': int(g.max()),
            'b_max': int(b.max()),
        }
    }


# ─── 4. HISTOGRAM ANALYSIS ────────────────────────────────────────────────────

def histogram_analysis(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img_bgr)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes.flat:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.spines[:].set_color('#30363d')

    # Grayscale histogram
    axes[0, 0].hist(gray.flatten(), bins=256, color='#58d9f5', alpha=0.8)
    axes[0, 0].set_title('Grayscale Histogram', color='#58d9f5')
    axes[0, 0].set_xlabel('Intensity', color='#8b949e')
    axes[0, 0].set_ylabel('Frequency', color='#8b949e')

    # Binned histogram (32 bins)
    axes[0, 1].hist(gray.flatten(), bins=32, color='#7ee787', alpha=0.8)
    axes[0, 1].set_title('Histogram (32 Bins)', color='#7ee787')
    axes[0, 1].set_xlabel('Intensity', color='#8b949e')
    axes[0, 1].set_ylabel('Frequency', color='#8b949e')

    # RGB histogram
    for channel, color, label in zip([r, g, b], ['#ff7b72', '#7ee787', '#58d9f5'], ['Red', 'Green', 'Blue']):
        axes[1, 0].plot(cv2.calcHist([channel], [0], None, [256], [0, 256]),
                        color=color, alpha=0.8, label=label, linewidth=1.5)
    axes[1, 0].set_title('RGB Color Histogram', color='#c9d1d9')
    axes[1, 0].legend(facecolor='#161b22', labelcolor='#c9d1d9')
    axes[1, 0].set_xlabel('Pixel Value', color='#8b949e')
    axes[1, 0].set_ylabel('Count', color='#8b949e')

    # Histogram comparison (cumulative)
    for channel, color, label in zip([r, g, b], ['#ff7b72', '#7ee787', '#58d9f5'], ['Red', 'Green', 'Blue']):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()
        axes[1, 1].plot(np.cumsum(hist) / np.sum(hist), color=color, alpha=0.8, label=label, linewidth=1.5)
    axes[1, 1].set_title('Cumulative Histogram', color='#c9d1d9')
    axes[1, 1].legend(facecolor='#161b22', labelcolor='#c9d1d9')
    axes[1, 1].set_xlabel('Pixel Value', color='#8b949e')
    axes[1, 1].set_ylabel('CDF', color='#8b949e')

    plt.tight_layout()
    return {
        'original': encode_image(img_bgr),
        'plot': encode_figure(),
        'info': {
            'total_pixels': f'{gray.size:,}',
            'mean_intensity': f'{gray.mean():.2f}',
            'median_intensity': f'{np.median(gray):.2f}',
            'std_dev': f'{gray.std():.2f}',
        }
    }


# ─── 5. IMAGE TRANSFORMATIONS ─────────────────────────────────────────────────

def image_transformations(img_bgr, brightness=50, contrast=1.5, alpha=1.2, beta=30):
    # Negative
    negative = 255 - img_bgr

    # Brightness
    bright = cv2.add(img_bgr, np.ones(img_bgr.shape, dtype=np.uint8) * int(brightness))

    # Contrast (using CLAHE on gray, merged back)
    contrast_img = cv2.convertScaleAbs(img_bgr, alpha=float(contrast), beta=0)

    # Alpha-Beta
    alpha_beta = cv2.convertScaleAbs(img_bgr, alpha=float(alpha), beta=int(beta))

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.patch.set_facecolor('#0d1117')
    titles = ['Original', f'Negative', f'Brightness +{brightness}', f'Contrast ×{contrast}', f'Alpha-Beta (α={alpha}, β={beta})']
    imgs = [img_bgr, negative, bright, contrast_img, alpha_beta]
    colors = ['#c9d1d9', '#ff7b72', '#f0e68c', '#7ee787', '#58d9f5']

    for ax, img, title, color in zip(axes, imgs, titles, colors):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color=color, fontsize=9)
        ax.axis('off')
        ax.set_facecolor('#0d1117')

    plt.tight_layout()
    return {
        'original': encode_image(img_bgr),
        'negative': encode_image(negative),
        'bright': encode_image(bright),
        'contrast': encode_image(contrast_img),
        'alpha_beta': encode_image(alpha_beta),
        'plot': encode_figure(),
        'params': {'brightness': brightness, 'contrast': contrast, 'alpha': alpha, 'beta': beta}
    }


# ─── 6. THRESHOLDING ──────────────────────────────────────────────────────────

def thresholding(img_bgr, threshold=127):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thresh_val = int(threshold)

    # Manual threshold
    _, manual = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # Otsu
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    # Sample 5x5 matrix
    sample = gray[gray.shape[0]//2-2:gray.shape[0]//2+3,
                  gray.shape[1]//2-2:gray.shape[1]//2+3]
    _, sample_thresh = cv2.threshold(sample, thresh_val, 255, cv2.THRESH_BINARY)

    # Histogram with threshold line
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.spines[:].set_color('#30363d')

    axes[0].hist(gray.flatten(), bins=256, color='#58d9f5', alpha=0.7)
    axes[0].axvline(x=thresh_val, color='#ff7b72', linewidth=2, linestyle='--', label=f'Manual ({thresh_val})')
    otsu_val = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    axes[0].axvline(x=otsu_val, color='#7ee787', linewidth=2, linestyle='--', label=f'Otsu ({int(otsu_val)})')
    axes[0].set_title('Histogram with Thresholds', color='#c9d1d9')
    axes[0].legend(facecolor='#161b22', labelcolor='#c9d1d9')

    axes[1].imshow(np.hstack([gray, manual, otsu]), cmap='gray')
    axes[1].set_title('Original | Manual | Otsu', color='#c9d1d9')
    axes[1].axis('off')

    plt.tight_layout()
    return {
        'original': encode_image(img_bgr),
        'gray': encode_image(gray, is_bgr=False),
        'manual': encode_image(manual, is_bgr=False),
        'otsu': encode_image(otsu, is_bgr=False),
        'adaptive': encode_image(adaptive, is_bgr=False),
        'plot': encode_figure(),
        'info': {
            'manual_threshold': thresh_val,
            'otsu_threshold': int(otsu_val),
            'sample_matrix': sample.tolist(),
            'sample_thresholded': sample_thresh.tolist(),
        }
    }


# ─── 7. GEOMETRIC TRANSFORMATIONS ─────────────────────────────────────────────

def geometric_transformations(img_bgr, angle=45, tx=50, ty=30, scale=0.75):
    h, w = img_bgr.shape[:2]

    # Flips
    flip_h = cv2.flip(img_bgr, 1)
    flip_v = cv2.flip(img_bgr, 0)
    flip_both = cv2.flip(img_bgr, -1)

    # Rotation 90°
    rot90 = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)

    # Rotation by angle
    M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), float(angle), 1.0)
    rotated = cv2.warpAffine(img_bgr, M_rot, (w, h))

    # Translation
    M_trans = np.float32([[1, 0, int(tx)], [0, 1, int(ty)]])
    translated = cv2.warpAffine(img_bgr, M_trans, (w, h))

    # Scaling (OpenCV)
    scale_f = float(scale)
    scaled_cv = cv2.resize(img_bgr, None, fx=scale_f, fy=scale_f, interpolation=cv2.INTER_LINEAR)

    # Scaling (manual nearest neighbor)
    new_h, new_w = int(h * scale_f), int(w * scale_f)
    scaled_manual = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    for i in range(new_h):
        for j in range(new_w):
            src_i = min(int(i / scale_f), h - 1)
            src_j = min(int(j / scale_f), w - 1)
            scaled_manual[i, j] = img_bgr[src_i, src_j]

    return {
        'original': encode_image(img_bgr),
        'flip_h': encode_image(flip_h),
        'flip_v': encode_image(flip_v),
        'flip_both': encode_image(flip_both),
        'rot90': encode_image(rot90),
        'rotated': encode_image(rotated),
        'translated': encode_image(translated),
        'scaled_cv': encode_image(scaled_cv),
        'scaled_manual': encode_image(scaled_manual),
        'params': {'angle': angle, 'tx': tx, 'ty': ty, 'scale': scale}
    }


# ─── 8. INTERPOLATION TECHNIQUES ──────────────────────────────────────────────

def interpolation_techniques(img_bgr, scale=2.0):
    h, w = img_bgr.shape[:2]
    scale_f = float(scale)
    new_size = (int(w * scale_f), int(h * scale_f))

    nn = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_NEAREST)
    linear = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_LINEAR)
    cubic = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_CUBIC)
    lanczos = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_LANCZOS4)

    # Comparison figure (center crop for clarity)
    cx, cy = new_size[0] // 2, new_size[1] // 2
    crop_size = min(150, cx, cy)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor('#0d1117')
    titles = ['Nearest Neighbor', 'Bilinear', 'Bicubic', 'Lanczos4']
    imgs = [nn, linear, cubic, lanczos]
    colors = ['#ff7b72', '#f0e68c', '#7ee787', '#58d9f5']

    for ax, img, title, color in zip(axes, imgs, titles, colors):
        crop = img[cy-crop_size:cy+crop_size, cx-crop_size:cx+crop_size]
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color=color, fontsize=10)
        ax.axis('off')
        ax.set_facecolor('#0d1117')

    plt.suptitle(f'Interpolation Comparison (×{scale_f} zoom, center crop)', color='#c9d1d9', fontsize=12)
    plt.tight_layout()

    return {
        'original': encode_image(img_bgr),
        'nearest': encode_image(nn),
        'linear': encode_image(linear),
        'cubic': encode_image(cubic),
        'lanczos': encode_image(lanczos),
        'plot': encode_figure(),
        'info': {
            'original_size': f'{w} × {h}',
            'scaled_size': f'{new_size[0]} × {new_size[1]}',
            'scale_factor': scale_f,
        }
    }


# ─── 9. FACE DETECTION ────────────────────────────────────────────────────────

def face_detection(img_bgr):
    import os
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Try to load cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    annotated = img_bgr.copy()
    face_crops = []

    if len(faces) > 0:
        for (x, y, fw, fh) in faces:
            cv2.rectangle(annotated, (x, y), (x + fw, y + fh), (0, 255, 100), 3)
            # Crop with padding
            pad = 20
            y1 = max(0, y - pad)
            y2 = min(img_bgr.shape[0], y + fh + pad)
            x1 = max(0, x - pad)
            x2 = min(img_bgr.shape[1], x + fw + pad)
            crop = img_bgr[y1:y2, x1:x2]
            face_crops.append(encode_image(crop))

    return {
        'original': encode_image(img_bgr),
        'annotated': encode_image(annotated),
        'faces': face_crops,
        'info': {
            'faces_found': len(faces),
            'face_coords': [{'x': int(x), 'y': int(y), 'w': int(fw), 'h': int(fh)} for x, y, fw, fh in faces] if len(faces) > 0 else [],
        }
    }
