#!/bin/bash
# Startup script for Azure App Service (Linux)

# 1. Update package list and install system dependencies for OpenCV/Torch
# We need libglib2.0-0 and libgl1 for OpenCV to run on Linux servers
apt-get update && apt-get install -y libglib2.0-0 libgl1

# 2. Run Gunicorn
# --bind 0.0.0.0:8000 specifies the port Azure expects
# --timeout 600 ensures the app doesn't timeout during heavy AI processing
# app:app refers to app.py and the Flask 'app' object
gunicorn --bind 0.0.0.0:8000 --timeout 600 --workers 1 app:app
