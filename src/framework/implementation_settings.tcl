# ==============================================================
# Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2019.1 (64-bit)
# Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
# ==============================================================
#
# Settings for Vivado implementation flow
#
set top_module dnn_LeNet
set language verilog
set family virtex7
set device xc7vx485t
set package -ffg1761
set speed -2
set clock ap_clk
set fsm_ext "off"
set target_clk_period_ns "5.000"
# For customizing the implementation flow
set add_io_buffers false ;# true|false
set hlsSolutionName ip_test
set targetPart ${device}${package}${speed}