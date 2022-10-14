#!/bin/bash

start_time="$(date -u +%s)"
arg1=$2
version=$3
sol_num=$4
source src/vivado_path_selector.sh $version

cd $arg1
echo "BASH : ----------------------------------------------------"
echo "BASH : Calling HLS: $arg1, Version $version and  $sol_num are the input arguments"
base_command="$dnn_hls -f run_hls_syn.tcl $sol_num"
echo $base_command
echo "BASH : =============   Synthesising ... ==================="

case $1 in
"show_error")

$base_command 
egrep -i --context=3 "error" synthesis_report$sol_num.txt

end_time="$(date -u +%s)"
duration="$(($end_time-$start_time))"
echo "BASH : ----------------------------------------------------"
echo "BASH : Total $(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
	;;

"skip")
   ;;

"python")
$base_command > synthesis_report$sol_num.txt
   ;;

*)

$base_command
esac

end_time="$(date -u +%s)"
duration="$(($end_time-$start_time))"
echo "BASH : ----------------------------------------------------"
echo "BASH : Total $(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

a=$(grep -c "error" vitis_hls.log)

if [[ $a -eq '0' ]] 
   then
   echo "BASH : =============   Synthesis done ====================="
else
   echo
   echo "BASH : =============   Synthesis faild  ==================="
fi



echo
