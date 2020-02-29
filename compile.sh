#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

echo "Building nl..."
cd lib/models
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace
