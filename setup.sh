#!/bin/bash

# ============================================================
# HalluSight - Mac Setup Script
# Run this ONCE to install everything you need
# Usage: bash setup.sh
# ============================================================

echo ""
echo "======================================"
echo "  HalluSight - Setting Up Environment"
echo "======================================"
echo ""

# Check Python version
echo "[1/6] Checking Python..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python3 not found. Please install it first:"
    echo "  brew install python@3.10"
    exit 1
fi
echo "✓ Python found"
echo ""

# Create virtual environment
echo "[2/6] Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate it
echo "[3/6] Activating virtual environment..."
source venv/bin/activate
echo "✓ Activated"
echo ""

# Upgrade pip
echo "[4/6] Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install all libraries
echo "[5/6] Installing all libraries (this may take 5-10 minutes)..."
pip install torch --quiet
pip install transformers accelerate --quiet
pip install scikit-learn numpy scipy --quiet
pip install spacy flask flask-cors joblib --quiet
pip install huggingface_hub --quiet
echo "✓ All libraries installed"
echo ""

# Download spacy language model
echo "[6/6] Downloading spaCy English model..."
python -m spacy download en_core_web_sm --quiet
echo "✓ spaCy model downloaded"
echo ""

echo "======================================"
echo "  ✅ SETUP COMPLETE!"
echo "======================================"
echo ""
echo "NEXT STEPS:"
echo "  1. Every time you open a new Terminal, run:"
echo "     source venv/bin/activate"
echo ""
echo "  2. To test LLaMA loading, run:"
echo "     python test_day1.py"
echo ""
echo "  3. If using LLaMA from HuggingFace, login first:"
echo "     huggingface-cli login"
echo ""
