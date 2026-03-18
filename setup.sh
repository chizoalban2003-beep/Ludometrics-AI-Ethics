#!/usr/bin/env bash
set -euo pipefail

# Heroku/Render-style platforms provide $PORT; default for local runs.
PORT="${PORT:-8501}"

mkdir -p ~/.streamlit/
cat > ~/.streamlit/config.toml <<EOF
[server]
headless = true
port = $PORT
EOF
