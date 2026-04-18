#!/bin/bash
set -e

echo "Jebi Hackathon 2026 - Grupo 12"
echo "EX-5600 Shovel Productivity Analysis"
echo "Inputs:"
ls -la inputs/

pip install -q -r requirements.txt

python -m solution.main

echo "Outputs generados:"
ls -la outputs/
echo "Done."
