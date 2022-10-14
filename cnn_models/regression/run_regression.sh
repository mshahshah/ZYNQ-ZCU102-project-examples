#!/bin/bash

# This batch file can be used to quickly test multiple runs of dnn_framework 
# to ensure that functionality has not been compromised
printf "Regression test started!"
source src/setup_python.sh 
rm -rf regression/*.log
clear

test_failed="FALSE"
for testcase in 1 2 3 4 5
do
    printf "Regression testcase $testcase is started!\n"

    #run the next testcase
    "regression/case_$testcase.sh"

    #check if the log file shows that the testcase was completed
    b=$(tail -1 regression/case_$testcase.log)
    
    if [[ $b != *"Passed"* ]]; then
    test_failed="TRUE"
    break
    fi
done

if [ $test_failed = "FALSE" ]
then printf "All tests PASSED!\n\n"
else printf "At least one test FAILED\n\n"
fi
