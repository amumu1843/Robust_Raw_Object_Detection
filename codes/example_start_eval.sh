#!/bin/bash
raw_path=$1 # absolute parent directory for input raws

# Please specify the exact version of dependencies,
# to avoid version incompatible errors.
pip install -r requirements.txt

# optional
# self-written libraries compilation
bash your_bash.sh

# start testing
python main.py --path ${raw_path}
