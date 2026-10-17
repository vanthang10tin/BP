#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "=========================================================="
echo "       QC STUDIO & CVAT 2D - ALL-IN-ONE LAUNCHER          "
echo "=========================================================="

if ! docker info > /dev/null 2>&1; then
    echo "[ERROR] Docker is not running. Please start Docker and retry."
    exit 1
fi

if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
    echo "Created .env from .env.example."
fi

echo "Starting all services via Docker Compose..."
docker compose up -d

echo ""
echo "=========================================================="
echo "                SYSTEM IS READY TO USE!                   "
echo "=========================================================="
echo "  1. CVAT Annotation Tool : http://localhost:8080"
echo "  2. QC Studio Workspace  : http://localhost:5000"
echo "  3. QC Engine API        : http://localhost:8000"
echo "=========================================================="
echo "To stop the system, run: docker compose down"
