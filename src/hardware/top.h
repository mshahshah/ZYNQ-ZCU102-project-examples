#ifndef __TOP_H__
#define __TOP_H__

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include "ap_fixed.h"
#include "ap_int.h"
#include "dnn_layers.h"
#include "dnn_configs.h"

//typedef ap_fixed<16,4> in_t;
//typedef ap_fixed<16,4> ker_t;
//typedef ap_fixed<16,4> res_t;
/*
typedef  int   in_t;
typedef  int   ker_t;
typedef  int   res_t;
typedef  int   mid_t;
typedef  int   cfg_t;
*/
//typedef  ap_int<16>   in_t;
//typedef  ap_int<16>   ker_t;
//typedef  ap_int<16>   res_t;
//typedef  ap_int<32>   mid_t;
//typedef  ap_int<16>   cfg_t;

//void dnn_top(in_t data_port[port_cfg::IN_MEM], ker_t kernel_port[port_cfg::KER_MEM], res_t output_port[port_cfg::OUT_MEM]);
void print_configs(void);
//void dnn_topDual(in_t data_port[Pcfg_CONV::IN_MEM], ker_t kernel_port[Pcfg_CONV::KER_MEM], res_t output_port[Pcfg_CONV::OUT_MEM]);
//void dnn_topDual_FC(in_t data_port[Pcfg_FC::IN_MEM], ker_t kernel_port[Pcfg_FC::KER_MEM], res_t output_port[Pcfg_FC::OUT_MEM]);
//void dnn_LeNet(in_t data_port[PORT_CFG::in_mem], ker_t kernel_port[PORT_CFG::ker_mem], res_t output_port[PORT_CFG::out_mem]);
#endif // __MATRIXMUL_H__ not defined
