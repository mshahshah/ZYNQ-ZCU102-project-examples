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

module hs_driver 
#(parameter
  START_DELAY   = 100,
  END_DELAY     = 500,
  NUM_LBLS      = 10
  )(
  input   logic          aclk,
  input   logic          aresetn,
  input   logic          i_ap_start,
  output  logic          o_ap_done,
  output  logic          o_ap_idle,
  output  logic          o_ap_ready,
  output  logic [31:0]   acc_out_addr,
  output  logic signed [15:0]   acc_out_data,
  output  logic          o_max_v,
  output  logic [4:0]    o_max_lbl
    );


logic signed [15:0]          find_max_array;
logic signed [31:0]          read_port_dout;
logic        [31:0]          addr, addr_i;
logic        [4:0]           max_indx;
logic                        read_port_en;
logic                        valid_data;

assign   acc_out_data  = valid_data ? read_port_dout[15:0] : 'd0;
assign   acc_out_addr  = valid_data ? addr>>>1 : 1'b0 ;
assign   o_max_v = valid_data;
//------------------------------------------------------------------
always @(posedge aclk or posedge aresetn)
  if(aresetn) begin
    find_max_array      <=  'd0;
    max_indx            <=  'b1;
    o_max_lbl           <=  'd0;
    //o_max_v             <=  'd0;
    read_port_en        <= 1'b0;
  end else begin


    if ( (acc_out_data > find_max_array ) && valid_data) begin
      find_max_array  <=  acc_out_data;
      max_indx        <=  addr>>>1;
    end

    if (o_ap_done) begin
      o_max_lbl   <=   max_indx;
      //o_max_v     <=   1'b1;
    end else begin
      o_max_lbl   <=   'd0;
      //o_max_v     <=   1'b0;
    end

  end


dnn_top_acc_wrapper blk1 (
 .O_read_port_addr  (addr),
 .O_read_port_clk   (),
 .O_read_port_din   (read_port_dout),
 .O_read_port_dout  (),
 .O_read_port_en    (valid_data),
 .O_read_port_rst   (),
 .O_read_port_we    (),
.ap_clk_0           (aclk),
.ap_ctrl_0_done     (o_ap_done),
.ap_ctrl_0_idle     (o_ap_idle),
.ap_ctrl_0_ready    (o_ap_ready),
.ap_ctrl_0_start    (i_ap_start),
.ap_rst_0           (aresetn)
);

endmodule


