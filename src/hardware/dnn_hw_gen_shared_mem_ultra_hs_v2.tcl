
set top_module LeNet [lindex $argv 0]
set mode new
#set b [lindex $argv 1]

set run_path [pwd]


source $top_module/integration/dnn_hw_gen_cfg.tcl



set prj_path          $run_path/$toplevel_hw
set outputDir         $prj_path/report_hw
set HLS_path          $run_path/$toplevel_hls
set integration_path  $HLS_path/integration

if { ($mode == "update") || ($mode == "new") } {
	puts "Running the hardware in $run_path"
	puts "The mode is $mode"
} else {
	puts "Exiting the process !  Enter a correct input argument: new , update"
	exit
}

file delete -force -- $outputDir
file mkdir $outputDir

proc add_dnn_module {} {
    create_bd_cell -type ip -vlnv xilinx.com:hls:dnn_LeNet:1.0 -name dnn_LeNet_0
	
}

proc add_clock_t1 {} {
    startgroup
    create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz_0
    set_property -dict [list CONFIG.PRIM_IN_FREQ.VALUE_SRC USER] [get_bd_cells clk_wiz_0]
    set_property -dict [list CONFIG.PRIM_IN_FREQ {120.000} CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {120.000} CONFIG.CLKIN1_JITTER_PS {83.33}] [get_bd_cells clk_wiz_0]
    set_property -dict [list CONFIG.MMCM_CLKFBOUT_MULT_F {10.000} CONFIG.MMCM_CLKIN1_PERIOD {8.333}] [get_bd_cells clk_wiz_0]
    set_property -dict [list CONFIG.MMCM_CLKOUT0_DIVIDE_F {10.000} CONFIG.CLKOUT1_JITTER {109.763}] [get_bd_cells clk_wiz_0]
    set_property -dict [list CONFIG.CLKOUT1_PHASE_ERROR {86.077}] [get_bd_cells clk_wiz_0]
    endgroup
}

proc add_clock {} {
    create_ip_run [get_files -of_objects [get_fileset sources_1] ./LeNet_hw_hs/auto_dnn_hw.srcs/sources_1/bd/dnn_top_acc/dnn_top_acc.bd]

    file mkdir $prj_path/auto_dnn_hw.srcs/constrs_1
    file mkdir $prj_path/auto_dnn_hw.srcs/constrs_1/new
    close [ open $prj_path/auto_dnn_hw.srcs/constrs_1/new/dnn_contraints.xdc w ]
    add_files -fileset constrs_1 $prj_path/auto_dnn_hw.srcs/constrs_1/new/dnn_contraints.xdc
    set_property target_constrs_file $prj_path/auto_dnn_hw.srcs/constrs_1/new/dnn_contraints.xdc [current_fileset -constrset]

    create_clock -period 10.000 -name CNN_clk -waveform {0.000 5.000} [get_ports aclk]
    save_constraints
}



if { $mode == "new" } {
puts "************ Building a new BD design  ************ "
    create_project -force auto_dnn_hw  $prj_path

    set_property board_part $FPGA_chip [current_project]
    reset_property board_connections [get_projects auto_dnn_hw]

    create_bd_design "dnn_top_acc"
    update_compile_order -fileset sources_1
    set_property  ip_repo_paths  $integration_path/ip [current_project]
    update_ip_catalog

    catch add_dnn_module

    add_files -norecurse ./src/hs_driver.sv
    #add_files -norecurse ./src/max2u.sv
    #add_files -norecurse ./src/max8.sv

    startgroup
        make_bd_pins_external  [get_bd_pins dnn_LeNet_0/ap_clk]
        make_bd_pins_external  [get_bd_pins dnn_LeNet_0/ap_rst]
        make_bd_intf_pins_external  [get_bd_intf_pins dnn_LeNet_0/ap_ctrl]
    endgroup    
    
    startgroup
    create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 $input_bram_name
    set_property -dict [list CONFIG.Use_RSTA_Pin {true} CONFIG.use_bram_block {Stand_Alone}] [get_bd_cells $input_bram_name]
    set_property -dict [list CONFIG.Enable_32bit_Address {true} CONFIG.Use_Byte_Write_Enable {true}] [get_bd_cells $input_bram_name]
    set_property -dict [list CONFIG.Byte_Size {8} CONFIG.Write_Depth_A {1030}] [get_bd_cells $input_bram_name]
    set_property -dict [list CONFIG.Register_PortA_Output_of_Memory_Primitives {false}] [get_bd_cells $input_bram_name]
    set_property -dict [list CONFIG.Load_Init_File {true} CONFIG.Coe_File $integration_path/test_files/in_data.coe] [get_bd_cells $input_bram_name]
    set_property -dict [list CONFIG.EN_SAFETY_CKT {false}] [get_bd_cells $input_bram_name]
    endgroup
    
    startgroup
    create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 $kernel_bram_name
    set_property -dict [list CONFIG.Use_RSTA_Pin {true} CONFIG.use_bram_block {Stand_Alone}] [get_bd_cells $kernel_bram_name]
    set_property -dict [list CONFIG.Enable_32bit_Address {true} CONFIG.Use_Byte_Write_Enable {true}] [get_bd_cells $kernel_bram_name]
    set_property -dict [list CONFIG.Byte_Size {8} CONFIG.Write_Depth_A {61720}] [get_bd_cells $kernel_bram_name]
    set_property -dict [list CONFIG.Register_PortA_Output_of_Memory_Primitives {false}] [get_bd_cells $kernel_bram_name]
    set_property -dict [list CONFIG.Load_Init_File {true} CONFIG.Coe_File $integration_path/test_files/kernels.coe] [get_bd_cells $kernel_bram_name]
    set_property -dict [list CONFIG.EN_SAFETY_CKT {false}] [get_bd_cells $kernel_bram_name]
    endgroup

    startgroup
    create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 $output_bram_name
    set_property -dict [list CONFIG.Memory_Type {Simple_Dual_Port_RAM} \
        CONFIG.Enable_32bit_Address {true} \
        CONFIG.Use_Byte_Write_Enable {true} \
        CONFIG.Byte_Size {8} CONFIG.Write_Width_A {32} \
        CONFIG.Write_Depth_A {10} \
        CONFIG.Read_Width_A {32} \
        CONFIG.Operating_Mode_A {NO_CHANGE} \
        CONFIG.Write_Width_B {32} \
        CONFIG.Read_Width_B {32} \
        CONFIG.Enable_B {Use_ENB_Pin} \
        CONFIG.Register_PortA_Output_of_Memory_Primitives {false} \
        CONFIG.Register_PortB_Output_of_Memory_Primitives {false} \
        CONFIG.Use_RSTA_Pin {false} CONFIG.Use_RSTB_Pin {true} \
        CONFIG.Port_B_Clock {100} \
        CONFIG.Port_B_Enable_Rate {100} \
        CONFIG.use_bram_block {Stand_Alone} \
        CONFIG.EN_SAFETY_CKT {false} \
        CONFIG.Coe_File {$prj_path/out_data.coe} \
        CONFIG.Fill_Remaining_Memory_Locations {true}] [get_bd_cells $output_bram_name]
    endgroup

    startgroup
    connect_bd_intf_net [get_bd_intf_pins $input_bram_name/BRAM_PORTA] [get_bd_intf_pins dnn_LeNet_0/data_port_PORTA]
    connect_bd_intf_net [get_bd_intf_pins $kernel_bram_name/BRAM_PORTA] [get_bd_intf_pins dnn_LeNet_0/kernel_port_PORTA]
    connect_bd_intf_net [get_bd_intf_pins $output_bram_name/BRAM_PORTA] [get_bd_intf_pins dnn_LeNet_0/output_port_PORTA]
	endgroup

    make_bd_intf_pins_external  [get_bd_intf_pins $output_bram_name/BRAM_PORTB]
    set_property name O_read_port [get_bd_intf_ports BRAM_PORTB_0]

} elseif { $mode == "update" } {
puts "************ Updateing the BD IP  ************ "
	open_project  $prj_path/auto_dnn_hw.xpr
	update_compile_order -fileset sources_1
	open_bd_design $prj_path/auto_dnn_hw.srcs/sources_1/bd/dnn_top_acc/dnn_top_acc.bd
	report_ip_status -name ip_status 
	upgrade_ip -vlnv xilinx.com:hls:dnn_LeNet:1.0 [get_ips  dnn_top_acc_dnn_LeNet_0_0] -log ip_upgrade.log
	export_ip_user_files -of_objects [get_ips dnn_top_acc_dnn_LeNet_0_0] -no_script -sync -force -quiet
	report_ip_status -name ip_status	
}

puts "************ Saving and generating Wrapper  ************ "
save_bd_design
regenerate_bd_layout
validate_bd_design
save_bd_design
make_wrapper -files [get_files $prj_path/auto_dnn_hw.srcs/sources_1/bd/dnn_top_acc/dnn_top_acc.bd] -top
add_files -norecurse $prj_path/auto_dnn_hw.srcs/sources_1/bd/dnn_top_acc/hdl/dnn_top_acc_wrapper.v
update_compile_order -fileset sources_1
save_bd_design


puts "************ Adding  clock  ************ "
add_files -fileset constrs_1 -norecurse ./src/cnn_constraints.xdc
add_files -fileset constrs_1 ./src/IO_connections.xdc
add_files -fileset sim_1 -norecurse     ./src/tb_hs_driver.sv
update_compile_order -fileset sim_1

if { $RUN_SIM } {
    puts "************ Running Simulation  ************ "
    set_property -name {xsim.simulate.log_all_signals} -value {true} -objects [get_filesets sim_1]
    set_property -name {xsim.simulate.runtime}         -value {-all} -objects [get_filesets sim_1]
    set_property        xsim.view                     ./src/tb_hs_driver.wcfg [get_filesets sim_1]
    add_files -fileset sim_1 -norecurse ./src/tb_hs_driver.wcfg
    launch_simulation
    run 5 us
}

if { $RUN_SYN } {
    puts "************ Running Synthesize  ************ "
    reset_run synth_1
    launch_runs synth_1 -jobs 20
    wait_on_run synth_1
    open_run synth_1 -name synth_1



    report_utilization -file [file join $outputDir utilization_report_syn.rpt] -name utilization_report_syn
    if { $RUN_PWR } {
        puts "************ Running Power analyzes ************"
        open_run synth_1 -name synth_1
        set_operating_conditions -grade extended
        report_power -file [file join $outputDir power_report_syn.rpt] -rpx [file join $outputDir power_report.rpx] -name {power_report}
    }


}


if { $RUN_IMPL } {
    puts "************ Running Implementation ************"
    file mkdir $prj_path/auto_dnn_hw.srcs/constrs_1/new
    #close [ open C:/Users/mshahshahani/Documents/dnn_hw/auto_dnn_hw.srcs/constrs_1/new/impl_report.xdc w ]
    #add_files -fileset constrs_1 $prj_path/auto_dnn_hw.srcs/constrs_1/new/dnn_contraints.xdc
    #set_property target_constrs_file $prj_path/auto_dnn_hw.srcs/constrs_1/new/dnn_contraints.xdc [current_fileset -constrset]
    #save_constraints -force
    startgroup
    open_run synth_1 -name synth_1
        set_property package_pin "" [get_ports [list  {acc_out_data[15]} {acc_out_data[14]} {acc_out_data[13]} {acc_out_data[12]} {acc_out_data[11]} {acc_out_data[10]} {acc_out_data[9]} {acc_out_data[8]} {acc_out_data[7]} {acc_out_data[6]} {acc_out_data[5]} {acc_out_data[4]} {acc_out_data[3]} {acc_out_data[2]} {acc_out_data[1]} {acc_out_data[0]} {o_max_lbl[4]} {o_max_lbl[3]} {o_max_lbl[2]} {o_max_lbl[1]} {o_max_lbl[0]} o_ap_done o_ap_ready o_max_v aclk aresetn i_ap_start o_ap_idle]]
        place_ports
    endgroup

    reset_run impl_1
    launch_runs impl_1 -jobs 10
    wait_on_run impl_1
    refresh_design
    report_utilization -file [file join $outputDir utilization_report_impl.rpt] -name utilization_report
    if { $RUN_PWR } {
        puts "************ Running Power analyzes ************"
        set_operating_conditions -grade extended
        report_power -file [file join $outputDir power_report_impl.rpt] -name {power_1}	
    }

}

if { $RUN_BIT } {
launch_runs impl_1 -to_step write_bitstream -jobs 20
}

#close_project
