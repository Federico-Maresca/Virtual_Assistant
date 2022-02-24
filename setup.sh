#!/bin/bash

if [ $# -eq 0 ]; then
  DATA_DIR="./"
else
  DATA_DIR="$1"
fi

#Install github Python dependencies
python3 -m pip install pip --upgrade
python3 -m pip install -r requirements.txt

echo -e "Dependencies where installed, all ready to go!"