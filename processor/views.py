import json
import numpy as np
import cv2
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from . import utils


OPERATIONS = {
    'read_display': {
        'label': 'Image Reading & Display',
        'icon': '📷',
        'category': 1,
        'desc': 'Read image, convert BGR↔RGB, display channels',
        'params': []
    },
    'properties': {
        'label': 'Image Properties',
        'icon': '📊',
        'category': 2,
        'desc': 'Rows, cols, channels, pixel counts, resolution',
        'params': []
    },
    'color_analysis': {
        'label': 'Color & Intensity Analysis',
        'icon': '🎨',
        'category': 3,
        'desc': 'Grayscale, channel splitting, intensity analysis',
        'params': []
    },
    'histogram': {
        'label': 'Histogram Analysis',
        'icon': '📈',
        'category': 4,
        'desc': 'Grayscale, binned, RGB, and cumulative histograms',
        'params': []
    },
    'transformations': {
        'label': 'Image Transformations',
        'icon': '⚡',
        'category': 5,
        'desc': 'Negative, brightness, contrast, alpha-beta',
        'params': [
            {'name': 'brightness', 'label': 'Brightness (+)', 'type': 'range', 'min': 0, 'max': 150, 'default': 50},
            {'name': 'contrast', 'label': 'Contrast (×)', 'type': 'range', 'min': 0.5, 'max': 4.0, 'step': 0.1, 'default': 1.5},
            {'name': 'alpha', 'label': 'Alpha (α)', 'type': 'range', 'min': 0.1, 'max': 3.0, 'step': 0.1, 'default': 1.2},
            {'name': 'beta', 'label': 'Beta (β)', 'type': 'range', 'min': -100, 'max': 100, 'default': 30},
        ]
    },
    'thresholding': {
        'label': 'Thresholding',
        'icon': '⬛',
        'category': 6,
        'desc': 'Manual, Otsu, adaptive thresholding + comparison',
        'params': [
            {'name': 'threshold', 'label': 'Threshold Value', 'type': 'range', 'min': 0, 'max': 255, 'default': 127},
        ]
    },
    'geometric': {
        'label': 'Geometric Transformations',
        'icon': '🔄',
        'category': 7,
        'desc': 'Flip, rotate, translate, scale',
        'params': [
            {'name': 'angle', 'label': 'Rotation Angle (°)', 'type': 'range', 'min': 1, 'max': 360, 'default': 45},
            {'name': 'tx', 'label': 'Translate X (px)', 'type': 'range', 'min': -200, 'max': 200, 'default': 50},
            {'name': 'ty', 'label': 'Translate Y (px)', 'type': 'range', 'min': -200, 'max': 200, 'default': 30},
            {'name': 'scale', 'label': 'Scale Factor', 'type': 'range', 'min': 0.1, 'max': 2.0, 'step': 0.05, 'default': 0.75},
        ]
    },
    'interpolation': {
        'label': 'Interpolation Techniques',
        'icon': '🔍',
        'category': 8,
        'desc': 'Nearest neighbor, bilinear, bicubic, Lanczos',
        'params': [
            {'name': 'scale', 'label': 'Scale Factor', 'type': 'range', 'min': 1.0, 'max': 4.0, 'step': 0.5, 'default': 2.0},
        ]
    },
    'face_detection': {
        'label': 'Face Detection & Crop',
        'icon': '👤',
        'category': 9,
        'desc': 'Haar Cascade detection and face cropping',
        'params': []
    },
}


def index(request):
    return render(request, 'processor/index.html', {'operations': OPERATIONS})


def read_image_from_request(request):
    if 'image' not in request.FILES:
        return None, 'No image uploaded'
    f = request.FILES['image']
    arr = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, 'Could not decode image'
    # Limit size
    h, w = img.shape[:2]
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img, None


@csrf_exempt
@require_http_methods(["POST"])
def process(request):
    operation = request.POST.get('operation', '')
    if operation not in OPERATIONS:
        return JsonResponse({'error': 'Unknown operation'}, status=400)

    img, err = read_image_from_request(request)
    if err:
        return JsonResponse({'error': err}, status=400)

    try:
        if operation == 'read_display':
            result = utils.read_and_display(img)
            template = 'processor/results/read_display.html'

        elif operation == 'properties':
            result = utils.image_properties(img)
            template = 'processor/results/properties.html'

        elif operation == 'color_analysis':
            result = utils.color_intensity_analysis(img)
            template = 'processor/results/color_analysis.html'

        elif operation == 'histogram':
            result = utils.histogram_analysis(img)
            template = 'processor/results/histogram.html'

        elif operation == 'transformations':
            brightness = float(request.POST.get('brightness', 50))
            contrast = float(request.POST.get('contrast', 1.5))
            alpha = float(request.POST.get('alpha', 1.2))
            beta = float(request.POST.get('beta', 30))
            result = utils.image_transformations(img, brightness, contrast, alpha, beta)
            template = 'processor/results/transformations.html'

        elif operation == 'thresholding':
            threshold = float(request.POST.get('threshold', 127))
            result = utils.thresholding(img, threshold)
            template = 'processor/results/thresholding.html'

        elif operation == 'geometric':
            angle = float(request.POST.get('angle', 45))
            tx = float(request.POST.get('tx', 50))
            ty = float(request.POST.get('ty', 30))
            scale = float(request.POST.get('scale', 0.75))
            result = utils.geometric_transformations(img, angle, tx, ty, scale)
            template = 'processor/results/geometric.html'

        elif operation == 'interpolation':
            scale = float(request.POST.get('scale', 2.0))
            result = utils.interpolation_techniques(img, scale)
            template = 'processor/results/interpolation.html'

        elif operation == 'face_detection':
            result = utils.face_detection(img)
            template = 'processor/results/face_detection.html'

        from django.template.loader import render_to_string
        html = render_to_string(template, {'result': result})
        return JsonResponse({'html': html, 'success': True})

    except Exception as e:
        import traceback
        return JsonResponse({'error': str(e), 'trace': traceback.format_exc()}, status=500)
