#!/bin/bash

testcase=4
echo "Target : Test normal dse_pragma on LeNet at 100MHz on Minimal Pragma!"
yes | cp -rf regression/input_arguments_$testcase.yaml input_arguments.yaml
python dnn_framework.py -mode dse_pragma > regression/case_$testcase.log

#check if the testcase was completed successfully
    b=$(tail -1 regression/case_$testcase.log)
    
    if [[ $b == *"Completed"* ]]; then
    echo "BASH: Case $testcase Passed!"
    echo "BASH: Task Passed!" >> regression/case_$testcase.log
    else
    printf "BASH: Case $testcase FAILED! \n\n"
    printf "BASH: Task Failed!" >> regression/case_$testcase.log
    fi