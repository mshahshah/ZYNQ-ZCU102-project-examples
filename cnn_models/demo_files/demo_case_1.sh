#!/bin/bash

testcase=1
echo "Target : Test normal HLS synthesize on LeNet at 100MHz on Minimal Pragma!"
yes | cp -rf demo_files/input_arguments_$testcase.yaml input_arguments.yaml
python dnn_framework.py -mode syn