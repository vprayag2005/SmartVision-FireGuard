#!/bin/bash
# Startup script for Azure App Service (Linux)

# 1. Update package list and install system dependencies for OpenCV/Torch
# We only install if libGL.so.1 is missing to avoid long startup times on every boot
if [ ! -f "/usr/lib/x86_64-linux-gnu/libGL.so.1" ]; then
    echo "Installing system dependencies..."
    apt-get update && apt-get install -y libglib2.0-0 libgl1
else
    echo "System dependencies already installed."
fi

# 2. Run Gunicorn
# --bind 0.0.0.0:8000 specifies the port Azure expects
# --timeout 600 ensures the app doesn't timeout during heavy AI processing
# app:app refers to app.py and the Flask 'app' object
gunicorn --bind 0.0.0.0:8000 --timeout 600 --workers 1 app:app
