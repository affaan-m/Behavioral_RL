#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p src/static/js

# Create main.js if it doesn't exist
if [ ! -f src/static/js/main.js ]; then
    echo "console.log('IGT Experiment loaded');" > src/static/js/main.js
fi