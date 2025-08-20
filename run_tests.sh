#!/bin/bash

for file in tests/*.py; do
    python "$file"
done