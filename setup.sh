#!/bin/sh

python -m venv .venv

.venv/bin/python -m pip install seaborn numpy matplotlib scipy pandas ipykernel scikit-learn
