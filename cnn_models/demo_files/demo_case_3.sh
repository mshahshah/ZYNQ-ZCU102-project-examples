#!/bin/bash

testcase=3
echo "Target : Test Full HLS synthesize and run estimation and compare outputs!"
yes | cp -rf demo_files/input_arguments_$testcase.yaml input_arguments.yaml
python dnn_framework.py -mode syn