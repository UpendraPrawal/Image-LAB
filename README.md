# 🔬 ImageLab — Computer Vision Workbench

A fully software-based Django web application for interactive image processing using OpenCV, NumPy, and Matplotlib.

---

## 📦 Features (9 Categories, 30+ Operations)

| # | Category | Operations |
|---|----------|-----------|
| 1 | **Image Reading & Display** | BGR/RGB reading, channel conversion, display |
| 2 | **Image Properties** | Rows, cols, channels, pixel count, resolution, memory |
| 3 | **Color & Intensity Analysis** | Grayscale, channel split, intensity stats |
| 4 | **Histogram Analysis** | Grayscale, binned, RGB, cumulative histograms |
| 5 | **Image Transformations** | Negative, brightness, contrast, alpha-beta |
| 6 | **Thresholding** | Manual, Otsu, adaptive + 5×5 matrix example |
| 7 | **Geometric Transformations** | Flip, rotate (90° & custom), translate, scale |
| 8 | **Interpolation Techniques** | Nearest neighbor, bilinear, bicubic, Lanczos4 |
| 9 | **Face Detection & Crop** | Haar Cascade detection + face cropping |

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.9+
- pip

### Install & Start
```bash
cd image_lab_project

# Option 1: Use the run script
bash run.sh

# Option 2: Manual setup
pip install -r requirements.txt
mkdir -p media
python manage.py runserver
```

Then open **http://127.0.0.1:8000** in your browser.

---

## 🛠️ Tech Stack

- **Backend**: Django 4.2
- **Image Processing**: OpenCV 4.x, NumPy, Matplotlib
- **Frontend**: Vanilla JS + CSS (no frontend framework)
- **No database required** — fully stateless, processes images in memory

---

## 📁 Project Structure

```
image_lab_project/
├── manage.py
├── requirements.txt
├── run.sh
├── image_lab/              # Django project settings
│   ├── settings.py
│   └── urls.py
└── processor/              # Main app
    ├── views.py            # Request handling & routing
    ├── utils.py            # All OpenCV processing functions
    ├── urls.py
    └── templates/
        └── processor/
            ├── index.html          # Main UI
            └── results/            # Result partials
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

---

## 💡 Usage

1. **Upload** an image (drag & drop or click)
2. **Select** an operation from the sidebar
3. **Adjust** parameters (sliders for brightness, threshold, rotation angle, etc.)
4. **Click** "RUN ANALYSIS" to process
5. **View** results — images + plots + data tables

---

## ⚙️ Configuration

Edit `image_lab/settings.py` to change:
- `DEBUG = False` for production
- `FILE_UPLOAD_MAX_MEMORY_SIZE` for max image size (default: 10MB)
- Images larger than 1200px are auto-resized for performance
