
puts "CMD : script_syn.tcl is running!"
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
open_solution -reset "ip_test"
#set_part {xc7vx485tffg1761-2}
set_part {xc7vx485t-ffg1761-2}
create_clock -period 8 -name default
#config_sdx -optimization_level none -target none
#config_export -vivado_optimization_level 2
set_clock_uncertainty 12.5%
source "./dnn_hls/ip_test/directives.tcl"

set TIME_start [clock clicks -milliseconds]

csynth_design
quit
set TIME_taken [expr [clock clicks -milliseconds] - $TIME_start]
set time_taken_second [expr {$TIME_taken/1000}]
puts "-------------time taken in second=----------------\n"
puts $time_taken_second