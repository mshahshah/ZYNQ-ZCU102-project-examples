`timescale 1ns / 1ps
//-----------------------------------------------------------------------------
// Title         : xFFT Wrapper
// Project       : iota
//-----------------------------------------------------------------------------
// File          : xFFT_wrapper.sv
// Author        : Masoud Shahshahani (mshahshahani@jmalogicless.com)
// Created       : 04.05.2021
// Last modified : 04.05.2021
//-----------------------------------------------------------------------------
// Description :
//
//-----------------------------------------------------------------------------
// Copyright (c) 2019 by JMA logicless This model is the confidential and
// proprietary property of JMA logicless and the possession or use of this
// file requires a written license from JMA logicless.
//------------------------------------------------------------------------------
// Modification history :
//
//-----------------------------------------------------------------------------

module tb_hs_driver #(parameter DEST_FILE   = "../../../../cnn_acc_sim_output.csv") ();



logic         rst, CLK66, CLK125, CLK500, CLK250;
logic         begin_sim, stop_sim, send_data, send_cfg, start_ifft_comp;

logic         ap_ready, ap_idle, ap_done, ap_start;

logic   [3:0]   clk_gen;
logic   [3:0]   CLK_ARRAY;

integer       in_file, out_file;

assign   CLK250  = clk_gen[1];
assign   CLK125  = clk_gen[2];
assign   CLK66   = clk_gen[3];
assign   aclk    = CLK250;

always #1 CLK500 = ~ CLK500;
initial begin

    out_file = $fopen(DEST_FILE,"w");
    CLK500=0;
    clk_gen = 3'b000;
    rst=0;
    begin_sim=0;
    #400  rst=1;
    #100  begin_sim=1;
    #102  begin_sim=0;
end

always @(posedge CLK500 or negedge rst)
    clk_gen  <=  clk_gen  + 'd1;

hs_driver hs_driver_inst (
.aclk        (aclk),
.o_ap_done   (ap_done),
.o_ap_idle   (ap_idle),
.o_ap_ready  (ap_ready),
.i_ap_start  (ap_start),
.aresetn     (~rst),
.acc_out_addr (acc_out_addr),
.acc_out_data (acc_out_data),
.o_max_v       (o_max_v),
.o_max_lbl     (o_max_lbl)
);

logic [15:0]    counter;
logic signed [15:0]    acc_out_data;
logic [31:0]    acc_out_addr;
logic [4:0]     o_max_lbl;

always @(posedge aclk or negedge rst)
   if(~rst) begin

   end else begin
      if (o_max_v)
        $fwrite(out_file, "[%3d]= %7d    largest label indx=%5d \n", acc_out_addr, acc_out_data, o_max_lbl);
      else
        $fwrite(out_file, "[%3d]= %7d\n", acc_out_addr, acc_out_data);
   end
//------------------------------------------------------------------
always @(posedge aclk or posedge rst)
  if(rst == 1'b0) begin
    ap_start     <=   1'b0;
    counter      <=   'd0;
  end else begin

    if (counter < 'd1000) begin
        ap_start     <=   1'b0;
        counter      <=   counter + 'd1;
    end else if (counter < 'd1100) begin
        ap_start     <=   1'b1;
        counter      <=   counter + 'd1;
    end else begin
        ap_start     <=   1'b0;
    end

    if (o_max_v) begin
    #2000;
    $stop;
    end
  end


endmodule


