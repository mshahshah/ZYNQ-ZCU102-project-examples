#ifndef __DNN_CONFIGS_H__
#define __DNN_CONFIGS_H__

#include "top.h"

using namespace std;

#define IN_W         32
#define IN_D         1

#define CONV_IN1     IN_W
#define KERW1        5
#define STRIDE1      1
#define LAYERS_IN1   IN_D
#define LAYERS_OUT1  6
#define CONV_OUT1    int((CONV_IN1-KERW1+1)/STRIDE1)

#define DS_IN2       CONV_OUT1
#define STRIDE2      2
#define LAYERS2      LAYERS_OUT1
#define DS_OUT2      int(DS_IN2/STRIDE2)

#define CONV_IN3     DS_OUT2
#define KERW3        3
#define STRIDE3      1
#define LAYERS_IN3   LAYERS2
#define LAYERS_OUT3  12
#define CONV_OUT3    int((CONV_IN3-KERW3+1)/STRIDE3)

#define DS_IN4       CONV_OUT3
#define STRIDE4      2
#define LAYERS4      LAYERS_OUT3
#define DS_OUT4      int(DS_IN4/STRIDE4)

#define FC_IN5       DS_OUT4*DS_OUT4*LAYERS4
#define FC_OUT5      84
#define FC_IN6       FC_OUT5
#define FC_OUT6      10

#define OUT_W        FC_OUT6

#define BASE_L1      0
#define BASE_L3      BASE_L1 + KERW1*KERW1*LAYERS_OUT1
#define BASE_L5      BASE_L3 + KERW3*KERW3*LAYERS_OUT3
#define BASE_L6      BASE_L5 + FC_IN5*FC_OUT5
#define HIGH_L6      BASE_L6 + FC_IN6*FC_OUT6

#define batch_size 2
const int reshape_factor=2;
/*
struct Pcfg_CONV{
	static const unsigned in_mem     = CONV_IN1 * CONV_IN1 * LAYERS_IN1;
	static const unsigned ker_mem    = KERW1 * KERW1 * LAYERS_OUT1;
	static const unsigned out_mem    = CONV_OUT1 * CONV_OUT1 * LAYERS_OUT1;
};
*/

struct PORT_CFG{
	static const unsigned in_mem     = CONV_IN1 * CONV_IN1 * LAYERS_IN1;
	static const unsigned ker_mem    = HIGH_L6;
	static const unsigned out_mem    = FC_OUT6;
};


struct L1_CONV{
	static const unsigned w_in      = CONV_IN1;
	static const unsigned stride    = STRIDE1;
	static const unsigned w_out     = CONV_OUT1;
	static const unsigned rc_strt   = 0;
	static const unsigned rc_end    = w_in-1;
	static const unsigned w_ker     = KERW1;
	static const unsigned w2_ker    = w_ker*w_ker;
	static const unsigned lyr_in    = LAYERS_IN1;
	static const unsigned lyr_out   = LAYERS_OUT1;
	static const unsigned mux_mult  = w2_ker;
};


struct L2_POOL{
	static const unsigned w_in      = DS_IN2;
	static const unsigned stride    = STRIDE2;
	static const unsigned w_out     = w_in/stride;
	static const unsigned rc_strt   = 0;
	static const unsigned rc_end    = w_in-stride;
	static const unsigned w_ker     = stride;
	static const unsigned w2_ker    = w_ker*w_ker;
	static const unsigned lyr_in    = LAYERS2;
	static const unsigned lyr_out   = lyr_in;
	static const unsigned mux_mult  = w2_ker;
};


struct L3_CONV{
	static const unsigned w_in      = CONV_IN3;
	static const unsigned stride    = STRIDE3;
	static const unsigned w_out     = CONV_OUT3;
	static const unsigned rc_strt   = 0;
	static const unsigned rc_end    = w_in-1;
	static const unsigned w_ker     = KERW3;
	static const unsigned w2_ker    = w_ker*w_ker;
	static const unsigned lyr_in    = LAYERS_IN3;
	static const unsigned lyr_out   = LAYERS_OUT3;
	static const unsigned mux_mult  = w2_ker;
};


struct L4_POOL{
	static const unsigned w_in      = DS_IN4;
	static const unsigned stride    = STRIDE4;
	static const unsigned w_out     = w_in/stride;
	static const unsigned rc_strt   = 0;
	static const unsigned rc_end    = w_in-stride;
	static const unsigned w_ker     = stride;
	static const unsigned w2_ker    = w_ker*w_ker;
	static const unsigned lyr_in    = LAYERS4;
	static const unsigned lyr_out   = lyr_in;
	static const unsigned mux_mult  = w2_ker;
};



struct L5_FC{
	static const unsigned batch_in   = 1;
	static const unsigned batch_out  = 1;
	static const unsigned lyr_in     = FC_IN5;
	static const unsigned lyr_out    = FC_OUT5;
	static const unsigned col_max    = lyr_in  / batch_in;
	static const unsigned row_max    = lyr_out / batch_out;
	static const unsigned fc_ker     = lyr_out  * lyr_in;
	static const unsigned mux_mult   = reshape_factor * 2; // using dual port mem
};

struct L6_FC{
    static const unsigned batch_in   = 1;
	static const unsigned batch_out  = 1;
	static const unsigned lyr_in     = FC_IN6;
	static const unsigned lyr_out    = FC_OUT6;
	static const unsigned col_max    = lyr_in   / batch_in;
	static const unsigned row_max    = lyr_out  / batch_out;
	static const unsigned fc_ker     = lyr_out  * lyr_in;
    static const unsigned mux_mult   = reshape_factor * 2; // using dual port mem
};


#endif // __MATRIXMUL_H__ not defined
