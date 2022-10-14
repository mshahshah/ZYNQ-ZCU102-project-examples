#!/bin/bash

testcase=2
echo "Target : Test Full synthesize on LeNet at 100 MHz on Minimal Pragma!"
yes | cp -rf demo_files/input_arguments_$testcase.yaml input_arguments.yaml
python dnn_framework.py -mode syn