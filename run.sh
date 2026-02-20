#!/bin/bash
# ImageLab — Django Image Processing Application
# Setup & Run Script

echo "=============================================="
echo "  ImageLab — Computer Vision Workbench"
echo "=============================================="

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create media directory
mkdir -p media

# Check Django setup
echo ""
echo "Checking setup..."
python manage.py check

echo ""
echo "=============================================="
echo "  Starting ImageLab server..."
echo "  Open: http://127.0.0.1:8000"
echo "=============================================="
python manage.py runserver 0.0.0.0:8000
