# ////////////////////////////////////////////////////////////////////////////////////////
# Copyright (c) by
# Company:  IDEA LAB, The University of Texas at Dallas
# Author :  Masoud Shahshahani
#
# Originally Create Date: Mar-5, 2020
# Project Name    : DNN_Framework
# Tool Versions   : Python +3
#
# Description     : Run power Analyzer in Vivado
# Dependencies    :
# Additional Comments:
#
# ///////////////////////////////////////////////////////////////////////////////////////

open_project verilog/project.xpr
update_compile_order -fileset sources_1
open_run synth_1 -name synth_1
report_power -file {verilog/report/rpt_power.xml} -name {rpt_power}