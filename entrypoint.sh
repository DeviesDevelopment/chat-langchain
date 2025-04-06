#!/bin/sh
set -e

# Run your Python script
# python ./backend/ingest.py

# Start your app with Uvicorn
exec uvicorn --app-dir=backend main:app --host 0.0.0.0 --port 8080