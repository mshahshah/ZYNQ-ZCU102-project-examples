

set sol_name "ip_test"
set sol_path "dnn_hls/$sol_name"
set sol_list_path "dnn_hls/${sol_name}_sol_list"
set sol_file "dnn_hls/$sol_name/${sol_name}_data.json"


set tclfiles [glob -directory $sol_list_path *.tcl]
set numberOfSolutions [llength $tclfiles] 
puts "DSE TCL : $numberOfSolutions solutions detected in $sol_list_path"

open_project dnn_hls
set_top dnn_LeNet
add_files dnn_hls/dnn_configs.h
add_files dnn_hls/dnn_layers.cpp
add_files dnn_hls/dnn_layers.h
add_files dnn_hls/top.cpp
add_files dnn_hls/top.h
add_files -tb dnn_hls/monitors.cpp
add_files -tb dnn_hls/monitors.h
add_files -tb dnn_hls/top_tb.cpp

for {set i 0} {$i < $numberOfSolutions} {incr i} {
   open_solution -reset $sol_name
   #set_part {xc7vx485tffg1761-2}
   set_part {xc7vx485t-ffg1761-2}
   create_clock -period 8 -name default
   #config_sdx -optimization_level none -target none
   #config_export -vivado_optimization_level 2
   set_clock_uncertainty 12.5%
   source "$sol_list_path/solution_$i.tcl"
   set TIME_start [clock clicks -milliseconds]

   file delete -force $sol_file
   
   csynth_design
   
   set TIME_taken [expr [clock clicks -milliseconds] - $TIME_start]
   set time_taken_second [expr {$TIME_taken/1000}]
   puts "-------------time taken in second=----------------\n"
   puts $time_taken_second
   
   if {[file exists $sol_file] == 1} {
      puts "DSE TCL : Solution $i file is copied"
      file copy -force $sol_file "$sol_list_path/solution_$i.json"
      }
   
   }
quit
