<div align="center">

```
██╗███╗   ███╗ █████╗  ██████╗ ███████╗ ██╗      █████╗ ██████╗
██║████╗ ████║██╔══██╗██╔════╝ ██╔════╝ ██║     ██╔══██╗██╔══██╗
██║██╔████╔██║███████║██║  ███╗█████╗   ██║     ███████║██████╔╝
██║██║╚██╔╝██║██╔══██║██║   ██║██╔══╝   ██║     ██╔══██║██╔══██╗
██║██║ ╚═╝ ██║██║  ██║╚██████╔╝███████╗ ███████╗██║  ██║██████╔╝
╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ╚══════╝╚═╝  ╚═╝╚═════╝
```

### *A full-stack Computer Vision Workbench powered by Django & OpenCV*

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Django](https://img.shields.io/badge/Django-4.2+-092E20?style=for-the-badge&logo=django&logoColor=white)](https://djangoproject.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Latest-11557C?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org)

<br/>

![License](https://img.shields.io/badge/License-MIT-f5c842?style=flat-square)
![No DB](https://img.shields.io/badge/Database-None_Required-4dffd2?style=flat-square)
![Zero Deps Frontend](https://img.shields.io/badge/Frontend-Vanilla_JS-ff6b6b?style=flat-square)
![Operations](https://img.shields.io/badge/Operations-30+-c4b5fd?style=flat-square)

<br/>

[**Quick Start**](#-quick-start) · [**Features**](#-features) · [**Operations**](#-operations) · [**Project Structure**](#-project-structure) · [**Tech Stack**](#-tech-stack)

</div>

---
## 🌍 Live Demo

👉 **Try ImageLab Online:**  
https://image-lab-2k6x.onrender.com
<br/>

---
## ✨ Overview

**ImageLab** is a fully software-based, browser-accessible image processing workbench. Upload any image, choose from **9 operation categories** and **30+ processing techniques**, tune parameters with live sliders, and explore rich visual results — histograms, side-by-side comparisons, annotated outputs, and data tables — all in your browser.

No cloud, no GPU, no database. Just Python, Django, and OpenCV running locally.

<br/>

## 🖥️ Interface Highlights

| Feature | Description |
|---|---|
| 🖱️ **Drag & Drop Upload** | Drop any image directly onto the upload zone |
| 🎚️ **Live Parameter Sliders** | Tune brightness, contrast, angle, threshold in real-time |
| 🔬 **Lightbox Zoom** | Click any result image to view full-size |
| 📊 **Plot Rendering** | Histograms and comparison charts rendered via Matplotlib |
| ⚡ **AJAX Processing** | Zero page reloads — results stream in dynamically |
| 🌑 **Premium Dark UI** | Glassmorphism panels, animated backgrounds, custom typography |

<br/>

## 🚀 Quick Start

### Prerequisites

- Python **3.9+**
- `pip` package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/imagelab.git
cd imagelab

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create media directory
mkdir -p media

# 5. Start the development server
python manage.py runserver
```

```
✅ Server running at → http://127.0.0.1:8000
```

> **Or use the one-line run script:**
> ```bash
> bash run.sh
> ```

<br/>

## 🧪 Features

### 9 Operation Categories · 30+ Techniques

```
┌─────────────────────────────────────────────────────────────────────┐
│  📷  Reading & Display     →  BGR/RGB conversion, channel display    │
│  📊  Image Properties      →  Dimensions, pixel counts, resolution   │
│  🎨  Color Analysis        →  Grayscale, channel split, intensity    │
│  📈  Histogram Analysis    →  Grayscale, binned, RGB, cumulative     │
│  ⚡  Transformations       →  Negative, brightness, contrast, α-β    │
│  ⬛  Thresholding          →  Manual, Otsu, adaptive + matrix view   │
│  🔄  Geometric             →  Flip, rotate, translate, scale         │
│  🔍  Interpolation         →  NN, bilinear, bicubic, Lanczos4        │
│  👤  Face Detection        →  Haar Cascade detection + face crop     │
└─────────────────────────────────────────────────────────────────────┘
```

<br/>

## 📋 Operations

<details>
<summary><b>📷 1 · Image Reading and Display</b></summary>

<br/>

| Sub-operation | Description |
|---|---|
| Read with OpenCV | Load image as BGR NumPy array |
| BGR → RGB Conversion | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` |
| Side-by-side Display | View original and converted image together |

</details>

<details>
<summary><b>📊 2 · Image Properties</b></summary>

<br/>

| Property | Method |
|---|---|
| Rows, Cols, Channels | `img.shape` |
| Total Pixels | `rows × cols` |
| Total RGB Values | `rows × cols × channels` |
| Center Pixel Value | `img[h//2, w//2]` |
| Resolution | `{width} × {height}` |
| Memory (uncompressed) | `img.nbytes` |

</details>

<details>
<summary><b>🎨 3 · Color and Intensity Analysis</b></summary>

<br/>

| Operation | Method |
|---|---|
| Grayscale Conversion | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` |
| Channel Splitting | `cv2.split(img)` → B, G, R arrays |
| Intensity Statistics | Mean, std dev per channel |
| Channel Visualization | False-color rendering per channel |

</details>

<details>
<summary><b>📈 4 · Histogram Analysis</b></summary>

<br/>

| Histogram Type | Description |
|---|---|
| Grayscale Histogram | 256-bin intensity frequency chart |
| Binned Histogram | Coarser 32-bin summary |
| RGB Color Histogram | Overlaid per-channel frequency curves |
| Cumulative Histogram | CDF comparison across R, G, B |

</details>

<details>
<summary><b>⚡ 5 · Image Transformations</b></summary>

<br/>

| Transform | Formula | Parameter |
|---|---|---|
| Negative | `output = 255 − input` | — |
| Brightness | `clip(input + β, 0, 255)` | `β` ∈ [0, 150] |
| Contrast | `clip(α × input, 0, 255)` | `α` ∈ [0.5, 4.0] |
| Alpha-Beta | `clip(α × input + β, 0, 255)` | `α` + `β` |

</details>

<details>
<summary><b>⬛ 6 · Thresholding</b></summary>

<br/>

| Method | Description |
|---|---|
| Manual Binary | Fixed user-defined threshold `T` |
| Otsu's Method | Auto-computed optimal threshold |
| Adaptive Gaussian | Local 11×11 neighborhood thresholding |
| Matrix Example | 5×5 center crop before/after view |

</details>

<details>
<summary><b>🔄 7 · Geometric Transformations</b></summary>

<br/>

| Operation | Method | Parameter |
|---|---|---|
| Horizontal Flip | `cv2.flip(img, 1)` | — |
| Vertical Flip | `cv2.flip(img, 0)` | — |
| Flip Both Axes | `cv2.flip(img, -1)` | — |
| Rotate 90° | `cv2.rotate(img, ROTATE_90_CLOCKWISE)` | — |
| Rotate Custom | `cv2.warpAffine` + rotation matrix | `θ` ∈ [1°, 360°] |
| Translate | `cv2.warpAffine` + translation matrix | `tx`, `ty` |
| Scale (OpenCV) | `cv2.resize` with `INTER_LINEAR` | `scale` ∈ [0.1, 2.0] |
| Scale (Manual) | Pixel loop — nearest neighbor | `scale` ∈ [0.1, 2.0] |

</details>

<details>
<summary><b>🔍 8 · Interpolation Techniques</b></summary>

<br/>

| Method | Flag | Quality | Speed |
|---|---|---|---|
| Nearest Neighbor | `INTER_NEAREST` | ★☆☆☆ | ★★★★ |
| Bilinear | `INTER_LINEAR` | ★★☆☆ | ★★★☆ |
| Bicubic | `INTER_CUBIC` | ★★★☆ | ★★☆☆ |
| Lanczos4 | `INTER_LANCZOS4` | ★★★★ | ★☆☆☆ |

</details>

<details>
<summary><b>👤 9 · Face Detection and Cropping</b></summary>

<br/>

| Step | Details |
|---|---|
| Classifier | `haarcascade_frontalface_default.xml` |
| Detection | `cv2.CascadeClassifier.detectMultiScale` |
| Scale Factor | `1.1` (10% size increase per scale) |
| Min Neighbors | `5` |
| Min Face Size | `30 × 30 px` |
| Cropping | Detected ROI + 20px padding, extracted as sub-image |

</details>

<br/>

## 🛠️ Tech Stack

```
┌──────────────────┬──────────────────────────────────────────────────┐
│  Layer           │  Technology                                       │
├──────────────────┼──────────────────────────────────────────────────┤
│  Web Framework   │  Django 4.2                                       │
│  Image Engine    │  OpenCV 4.x (opencv-python-headless)              │
│  Array Math      │  NumPy 1.24+                                      │
│  Visualization   │  Matplotlib 3.7+ (Agg backend, base64 encoded)   │
│  Image I/O       │  Pillow 10.0+                                     │
│  Frontend        │  Vanilla JS · CSS3 · HTML5 (zero frameworks)     │
│  Fonts           │  Instrument Serif · DM Mono · Outfit (Google)    │
│  Database        │  None — fully stateless, in-memory processing    │
└──────────────────┴──────────────────────────────────────────────────┘
```

<br/>

## 📁 Project Structure

```
image_lab_project/
│
├── 📄 manage.py                    # Django management entry point
├── 📄 requirements.txt             # Python dependencies
├── 📄 run.sh                       # One-command setup & launch script
│
├── 📁 image_lab/                   # Django project configuration
│   ├── settings.py                 # App settings, media config, limits
│   ├── urls.py                     # Root URL routing
│   └── wsgi.py                     # WSGI entry point
│
└── 📁 processor/                   # Core application
    ├── views.py                    # Request handling & operation dispatch
    ├── utils.py                    # All OpenCV processing functions (650+ lines)
    ├── urls.py                     # App URL patterns
    │
    └── 📁 templates/processor/
        ├── index.html              # Main UI — sidebar, upload, controls, results
        └── 📁 results/             # Per-operation result partials (AJAX targets)
            ├── read_display.html
            ├── properties.html
            ├── color_analysis.html
            ├── histogram.html
            ├── transformations.html
            ├── thresholding.html
            ├── geometric.html
            ├── interpolation.html
            └── face_detection.html
```

<br/>

## 🎮 How to Use

```
 STEP 1          STEP 2              STEP 3            STEP 4
┌──────────┐    ┌──────────────┐    ┌─────────────┐   ┌────────────┐
│  Upload  │ →  │   Select an  │ →  │   Adjust    │ → │    View    │
│  Image   │    │  Operation   │    │  Parameters │   │  Results   │
│          │    │  (Sidebar)   │    │  (Sliders)  │   │            │
│ Drag &   │    │              │    │             │   │  Images +  │
│  Drop or │    │  9 Categories│    │ Brightness  │   │  Plots +   │
│  Browse  │    │  30+ Methods │    │ Threshold   │   │  Tables    │
└──────────┘    └──────────────┘    │ Angle, etc. │   └────────────┘
                                    └─────────────┘
```

<br/>

## ⚙️ Configuration

Edit **`image_lab/settings.py`** to customize:

| Setting | Default | Description |
|---|---|---|
| `DEBUG` | `True` | Set to `False` for production |
| `ALLOWED_HOSTS` | `['*']` | Restrict to your domain in production |
| `FILE_UPLOAD_MAX_MEMORY_SIZE` | `10 MB` | Maximum uploaded image size |
| Max image dimension | `1200 px` | Images auto-resized above this (hardcoded in `views.py`) |

<br/>

## 📦 Dependencies

```txt
Django>=4.2
opencv-python-headless>=4.8
numpy>=1.24
matplotlib>=3.7
Pillow>=10.0
```

> **Note:** `opencv-python-headless` is used (no GUI window support needed) — lighter than full `opencv-python`.

<br/>

## 🔒 Security Notes for Production

- Set `DEBUG = False` in `settings.py`
- Set `SECRET_KEY` to a strong random value via environment variable
- Set `ALLOWED_HOSTS` to your actual domain
- Consider adding rate limiting to the `/process/` endpoint
- Add authentication if deploying publicly

```python
# settings.py — production example
import os
SECRET_KEY = os.environ['DJANGO_SECRET_KEY']
DEBUG = False
ALLOWED_HOSTS = ['yourdomain.com']
```

<br/>

## 🤝 Contributing

Contributions are welcome! Here's how to add a new operation:

1. **Add processing logic** in `processor/utils.py`
2. **Register the operation** in `OPERATIONS` dict in `processor/views.py`
3. **Add a view branch** in the `process()` view function
4. **Create a result template** in `processor/templates/processor/results/`

<br/>

## 📄 License

```
MIT License — free to use, modify, and distribute.
```

<br/>

---

<div align="center">

**Built with 🧠 Python · 👁️ OpenCV · 🌐 Django**

*ImageLab — See more in every pixel.*

</div>
