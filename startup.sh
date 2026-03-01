#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Ensure system dependencies for OpenCV/Torch are installed
if [ ! -f "/usr/lib/x86_64-linux-gnu/libGL.so.1" ]; then
    apt-get update && apt-get install -y libglib2.0-0 libgl1
fi

# Activate virtualenv if present
[ -d "./antenv" ] && source "./antenv/bin/activate"

# Start Gunicorn
PORT_TO_USE="${PORT:-${WEBSITES_PORT:-8000}}"
echo "Starting SmartVision Guard on port: ${PORT_TO_USE}"

exec gunicorn --bind 0.0.0.0:${PORT_TO_USE} --timeout 600 --workers 1 --threads 12 \
  --access-logfile - --error-logfile - --capture-output --log-level info \
  app:app
