#!/bin/bash
# Startup script for Azure App Service (Linux)

# 1. Update package list and install system dependencies for OpenCV/Torch if needed
# Note: opencv-python-headless should handle most dependencies, but libglib2.0-0 is often needed
apt-get update && apt-get install -y libglib2.0-0

# 2. Run Gunicorn
# --bind 0.0.0.0:8000 specifies the port Azure expects
# --timeout 600 ensures the app doesn't timeout during heavy AI processing
# app:app refers to app.py and the Flask 'app' object
gunicorn --bind 0.0.0.0:8000 --timeout 600 --workers 1 app:app
