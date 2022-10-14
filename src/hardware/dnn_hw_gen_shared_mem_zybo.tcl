
set mode [lindex $argv 0]
#set b [lindex $argv 1]

set run_path [pwd]


source src/dnn_hw_gen_cfg.tcl



set prj_path  $run_path/$toplevel_hw
set outputDir $prj_path/report_hw
set HLS_path   $run_path/$toplevel_hls

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

proc add_clock {} {
	file mkdir $prj_path/auto_dnn_hw.srcs/constrs_1
	file mkdir $prj_path/auto_dnn_hw.srcs/constrs_1/new
	close [ open $prj_path/auto_dnn_hw.srcs/constrs_1/new/dnn_contraints.xdc w ]
	add_files -fileset constrs_1 $prj_path/auto_dnn_hw.srcs/constrs_1/new/dnn_contraints.xdc
	set_property target_constrs_file $prj_path/auto_dnn_hw.srcs/constrs_1/new/dnn_contraints.xdc [current_fileset -constrset]

	create_clock -period 10.000 -name DNN_clock [get_ports clk_in1_0]
	save_constraints
}




if { $mode == "new" } {
puts "************ Building a new BD design  ************ "
	create_project -force auto_dnn_hw  $prj_path

	set_property board_part $FPGA_chip [current_project]
	reset_property board_connections [get_projects auto_dnn_hw]

	create_bd_design "dnn_top_acc"
	update_compile_order -fileset sources_1
	set_property  ip_repo_paths  $HLS_path/hls/$sol_name/impl/ip [current_project]
	update_ip_catalog

	catch add_dnn_module

	startgroup
	create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 axi_input_data
	set_property -dict [list CONFIG.SINGLE_PORT_BRAM {1}] [get_bd_cells axi_input_data]

	create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 $input_bram_name
	set_property -dict [list CONFIG.Memory_Type {True_Dual_Port_RAM} CONFIG.Enable_B {Use_ENB_Pin} CONFIG.Use_RSTB_Pin {true} CONFIG.Port_B_Clock {100} CONFIG.Port_B_Write_Rate {50} CONFIG.Port_B_Enable_Rate {100}] [get_bd_cells $input_bram_name]

	connect_bd_intf_net [get_bd_intf_pins axi_input_data/BRAM_PORTA] [get_bd_intf_pins $input_bram_name/BRAM_PORTA]
	connect_bd_intf_net [get_bd_intf_pins $input_bram_name/BRAM_PORTB] [get_bd_intf_pins dnn_LeNet_0/data_port_PORTA]

	create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 axi_kernel
	set_property -dict [list CONFIG.SINGLE_PORT_BRAM {1}] [get_bd_cells axi_kernel]
	create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 $kernel_bram_name
	set_property -dict [list CONFIG.Memory_Type {True_Dual_Port_RAM} CONFIG.Enable_B {Use_ENB_Pin} CONFIG.Use_RSTB_Pin {true} CONFIG.Port_B_Clock {100} CONFIG.Port_B_Write_Rate {50} CONFIG.Port_B_Enable_Rate {100}] [get_bd_cells $kernel_bram_name]
	connect_bd_intf_net [get_bd_intf_pins axi_kernel/BRAM_PORTA] [get_bd_intf_pins $kernel_bram_name/BRAM_PORTA]
	connect_bd_intf_net [get_bd_intf_pins $kernel_bram_name/BRAM_PORTB] [get_bd_intf_pins dnn_LeNet_0/kernel_port_PORTA]


	create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 axi_output_data
	set_property -dict [list CONFIG.SINGLE_PORT_BRAM {1}] [get_bd_cells axi_output_data]
	create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 $output_bram_name
	set_property -dict [list CONFIG.Memory_Type {True_Dual_Port_RAM} CONFIG.Enable_B {Use_ENB_Pin} CONFIG.Use_RSTB_Pin {true} CONFIG.Port_B_Clock {100} CONFIG.Port_B_Write_Rate {50} CONFIG.Port_B_Enable_Rate {100}] [get_bd_cells $output_bram_name]
	connect_bd_intf_net [get_bd_intf_pins axi_output_data/BRAM_PORTA] [get_bd_intf_pins $output_bram_name/BRAM_PORTA]
	connect_bd_intf_net [get_bd_intf_pins $output_bram_name/BRAM_PORTB] [get_bd_intf_pins dnn_LeNet_0/output_port_PORTA]
	endgroup



	startgroup
	set_property -dict [list CONFIG.Enable_32bit_Address {false} CONFIG.Use_Byte_Write_Enable {false} CONFIG.Byte_Size {9} CONFIG.Register_PortA_Output_of_Memory_Primitives {true} CONFIG.Register_PortB_Output_of_Memory_Primitives {true} CONFIG.Use_RSTA_Pin {false} CONFIG.Use_RSTB_Pin {false} CONFIG.use_bram_block {Stand_Alone} CONFIG.EN_SAFETY_CKT {false}] [get_bd_cells $input_bram_name]
	set_property -dict [list CONFIG.Write_Depth_A {8100}] [get_bd_cells $input_bram_name]
	set_property -dict [list CONFIG.Enable_32bit_Address {false} CONFIG.Use_Byte_Write_Enable {false} CONFIG.Byte_Size {9} CONFIG.Register_PortA_Output_of_Memory_Primitives {true} CONFIG.Register_PortB_Output_of_Memory_Primitives {true} CONFIG.Use_RSTA_Pin {false} CONFIG.Use_RSTB_Pin {false} CONFIG.use_bram_block {Stand_Alone} CONFIG.EN_SAFETY_CKT {false}] [get_bd_cells $kernel_bram_name]
	set_property -dict [list CONFIG.Write_Depth_A {8100}] [get_bd_cells $kernel_bram_name]
	set_property -dict [list CONFIG.Enable_32bit_Address {false} CONFIG.Use_Byte_Write_Enable {false} CONFIG.Byte_Size {9} CONFIG.Register_PortA_Output_of_Memory_Primitives {true} CONFIG.Register_PortB_Output_of_Memory_Primitives {true} CONFIG.Use_RSTA_Pin {false} CONFIG.Use_RSTB_Pin {false} CONFIG.use_bram_block {Stand_Alone} CONFIG.EN_SAFETY_CKT {false}] [get_bd_cells $output_bram_name]
	set_property -dict [list CONFIG.Write_Depth_A {8100}] [get_bd_cells $output_bram_name]
	endgroup

	if {$AXI_MODE == "AXI_BRAM"} {
		puts "************ Using $AXI_MODE for data path  ************ "
		startgroup
		create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_0
		set_property -dict [list CONFIG.NUM_MI {4} CONFIG.NUM_SI {1}] [get_bd_cells smartconnect_0]
		make_bd_intf_pins_external  [get_bd_intf_pins smartconnect_0/S00_AXI]
		connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins axi_input_data/S_AXI]
		connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M01_AXI] [get_bd_intf_pins axi_kernel/S_AXI]
		connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M02_AXI] [get_bd_intf_pins axi_output_data/S_AXI]
		connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M03_AXI] [get_bd_intf_pins dnn_LeNet_0/s_axi_control]

		connect_bd_net [get_bd_pins axi_input_data/s_axi_aclk] [get_bd_pins axi_kernel/s_axi_aclk]
		connect_bd_net [get_bd_pins axi_output_data/s_axi_aclk] [get_bd_pins axi_input_data/s_axi_aclk]
		connect_bd_net [get_bd_pins dnn_LeNet_0/ap_clk] [get_bd_pins axi_input_data/s_axi_aclk]
		connect_bd_net [get_bd_pins smartconnect_0/aclk] [get_bd_pins axi_input_data/s_axi_aclk]
		connect_bd_net [get_bd_pins axi_input_data/s_axi_aresetn] [get_bd_pins axi_kernel/s_axi_aresetn]
		connect_bd_net [get_bd_pins axi_kernel/s_axi_aresetn] [get_bd_pins axi_output_data/s_axi_aresetn]
		connect_bd_net [get_bd_pins axi_output_data/s_axi_aresetn] [get_bd_pins dnn_LeNet_0/ap_rst_n]
		connect_bd_net [get_bd_pins dnn_LeNet_0/ap_rst_n] [get_bd_pins smartconnect_0/aresetn]

		apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {New Clocking Wizard} Freq {100} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins dnn_LeNet_0/ap_clk]
		set_property -dict [list CONFIG.PRIM_IN_FREQ.VALUE_SRC USER] [get_bd_cells clk_wiz]
		set_property -dict [list CONFIG.CLK_IN1_BOARD_INTERFACE {Custom} CONFIG.CLK_IN2_BOARD_INTERFACE {Custom} CONFIG.PRIM_IN_FREQ {100.000} CONFIG.RESET_BOARD_INTERFACE {reset} CONFIG.PRIM_SOURCE {Single_ended_clock_capable_pin} CONFIG.CLKIN1_JITTER_PS {100.0} CONFIG.MMCM_CLKFBOUT_MULT_F {10.000} CONFIG.MMCM_CLKIN1_PERIOD {10.000} CONFIG.MMCM_CLKIN2_PERIOD {10.000} CONFIG.CLKOUT1_JITTER {130.958} CONFIG.CLKOUT1_PHASE_ERROR {98.575}] [get_bd_cells clk_wiz]
		make_bd_pins_external  [get_bd_pins clk_wiz/clk_in1]
		make_bd_pins_external  [get_bd_pins clk_wiz/reset]
		connect_bd_net [get_bd_ports reset_0] [get_bd_pins rst_clk_wiz_100M/ext_reset_in]
		endgroup
	} elseif {$AXI_MODE == "ZYNQ"} {
		puts "************ Using $AXI_MODE for data path  ************ "

		create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.3 zynq_ultra_ps_e_0
		apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1" }  [get_bd_cells zynq_ultra_ps_e_0]

        startgroup
        apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} Slave {/axi_input_data/S_AXI} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins axi_input_data/S_AXI]
        apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} Slave {/axi_kernel/S_AXI} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins axi_kernel/S_AXI]
        apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} Slave {/axi_output_data/S_AXI} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins axi_output_data/S_AXI]
        apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} Slave {/dnn_LeNet_0/s_axi_control} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins dnn_LeNet_0/s_axi_control]
        endgroup
        apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {/zynq_ultra_ps_e_0/pl_clk0 (99 MHz)} Clk_xbar {/zynq_ultra_ps_e_0/pl_clk0 (99 MHz)} Master {/zynq_ultra_ps_e_0/M_AXI_HPM1_FPD} Slave {/axi_input_data/S_AXI} ddr_seg {Auto} intc_ip {/axi_smc} master_apm {0}}  [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM1_FPD]

		#puts "************ Using $AXI_MODE for data path  ************ "
		#startgroup
		#create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
		#apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system7_0]
		#apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/processing_system7_0/M_AXI_GP0} Slave {/axi_input_data/S_AXI} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins axi_input_data/S_AXI]
		#apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/processing_system7_0/M_AXI_GP0} Slave {/axi_kernel/S_AXI} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins axi_kernel/S_AXI]
		#apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/processing_system7_0/M_AXI_GP0} Slave {/axi_output_data/S_AXI} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins axi_output_data/S_AXI]
		#apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/processing_system7_0/M_AXI_GP0} Slave {/dnn_LeNet_0/s_axi_control} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins dnn_LeNet_0/s_axi_control]
		#set_property -dict [list CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $AXI_CLK] [get_bd_cells processing_system7_0]
		#connect_bd_net [get_bd_ports reset_0] [get_bd_pins rst_clk_wiz_100M/ext_reset_in]
		#endgroup
    }
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
save_bd_design

puts "************ Adding  clock  ************ "
catch add_clock

if { $RUN_SYN } {
	puts "************ Running Synthesize  ************ "
	reset_run synth_1
	launch_runs synth_1 -jobs 10
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
	reset_run impl_1
	launch_runs impl_1 -jobs 10
	refresh_design
	report_utilization -file [file join $outputDir utilization_report_impl.rpt] -name utilization_report
	if { $RUN_PWR } {
		puts "************ Running Power analyzes ************"
		set_operating_conditions -grade extended
		report_power -file [file join $outputDir power_report_impl.rpt] -name {power_1}	
   }
}

close_project