#!/bin/sh
set -e

# echo "Running ingest step..."
# Run your Python script
# python ./backend/ingest.py

echo "Starting Uvicorn..."
# Start your app with Uvicorn
exec uvicorn --app-dir=backend main:app --host 0.0.0.0 --port 8080