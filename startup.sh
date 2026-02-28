#!/bin/bash
# Startup script for Azure App Service (Linux)
set -euo pipefail

export PYTHONUNBUFFERED=1

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

# 1. Update package list and install system dependencies for OpenCV/Torch
# We only install if libGL.so.1 is missing to avoid long startup times on every boot
if [ ! -f "/usr/lib/x86_64-linux-gnu/libGL.so.1" ]; then
    echo "Installing system dependencies..."
    apt-get update && apt-get install -y libglib2.0-0 libgl1
else
    echo "System dependencies already installed."
fi

# 2. Run Gunicorn
# --bind 0.0.0.0:${PORT:-8000} uses the port Azure exposes
# --timeout 600 ensures the app doesn't timeout during heavy AI processing
# --access-logfile/-error-logfile emit logs to stdout for App Service logs
# app:app refers to app.py and the Flask 'app' object
if [ -d "$APP_DIR/antenv" ]; then
    echo "Activating virtualenv: $APP_DIR/antenv"
    # shellcheck disable=SC1090
    source "$APP_DIR/antenv/bin/activate"
fi
PORT_TO_USE="${PORT:-${WEBSITES_PORT:-8000}}"
echo "Using PORT: ${PORT_TO_USE}"
exec gunicorn --bind 0.0.0.0:${PORT_TO_USE} --timeout 600 --workers 1 \
  --access-logfile - --error-logfile - --capture-output --log-level info \
  app:app
