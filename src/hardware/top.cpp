#include "top.h"



void dnn_LeNet(in_t data_port[PORT_CFG::in_mem], ker_t kernel_port[PORT_CFG::ker_mem], res_t output_port[PORT_CFG::out_mem])
{

	interface intf;
	dnn_layers dnn;

	in_t  L1_in  [L1_CONV::lyr_in][L1_CONV::w_in][L1_CONV::w_in];

	ker_t L1_ker    [L1_CONV::lyr_out][L1_CONV::w2_ker];
	ker_t L3_ker    [L3_CONV::lyr_out][L3_CONV::w2_ker];
	ker_t L6_ker    [L5_FC::fc_ker];
	ker_t L7_ker    [L6_FC::fc_ker];

	res_t L1_out    [L1_CONV::lyr_out][L1_CONV::w_out][L1_CONV::w_out];
	res_t L2_out    [L2_POOL::lyr_out][L2_POOL::w_out][L2_POOL::w_out];
	res_t L3_out    [L3_CONV::lyr_out][L3_CONV::w_out][L3_CONV::w_out];
	res_t L4_out    [L4_POOL::lyr_out][L4_POOL::w_out][L4_POOL::w_out];
	res_t L5_out    [L4_POOL::lyr_out * L4_POOL::w_out * L4_POOL::w_out];
	res_t L6_out    [L5_FC::lyr_out];
	res_t L7_out    [L6_FC::lyr_out];

   intf.read_input3D    <in_t , PORT_CFG, L1_CONV>   (data_port,L1_in);
	intf.read_kernel2D   <ker_t, PORT_CFG, L1_CONV>   (BASE_L1, kernel_port,L1_ker);
	intf.read_kernel2D   <ker_t, PORT_CFG, L3_CONV>   (BASE_L3, kernel_port,L3_ker);	
	intf.read_kernel     <ker_t, PORT_CFG, L5_FC>    (BASE_L5, kernel_port,L6_ker);
   intf.read_kernel     <ker_t, PORT_CFG, L6_FC>    (BASE_L6, kernel_port,L7_ker);
	


	dnn.conv_3DT1  <in_t , ker_t   , res_t, mid_t, L1_CONV>  (1, L1_in  , L1_ker , L1_out);
	dnn.ds_3DT1    <in_t , res_t, L2_POOL>                    (2, L1_out , L2_out);

	dnn.conv_3DT1  <in_t , ker_t   , res_t, mid_t, L3_CONV>  (3, L2_out , L3_ker , L3_out);
	dnn.ds_3DT1    <in_t , res_t, L4_POOL>                    (4, L3_out , L4_out);

	dnn.conv2fc    <in_t, L4_POOL>                            (45, L4_out , L5_out);

	dnn.fc_T1      <in_t, ker_t,res_t, mid_t, L5_FC>        (5, L5_out, L6_ker, L6_out);
	dnn.fc_T1      <in_t, ker_t,res_t, mid_t, L6_FC>        (6, L6_out, L7_ker, L7_out);

	intf.write_result   <res_t, PORT_CFG, L6_FC>                (output_port,L7_out);
	//intf1.write_result2D <res_t, PORT_CFG, L4_POOL>              (output_port,L4_out);


}



/*
void dnn_topDual(in_t data_port[Pcfg_CONV::IN_MEM], ker_t kernel_port[Pcfg_CONV::KER_MEM], res_t output_port[Pcfg_CONV::OUT_MEM])
{
#pragma HLS ARRAY_RESHAPE variable=data_port   cyclic factor=2 dim=1
#pragma HLS ARRAY_RESHAPE variable=kernel_port cyclic factor=2 dim=1
#pragma HLS ARRAY_RESHAPE variable=output_port cyclic factor=2 dim=1

	//#pragma HLS RESOURCE  variable=data_port    core=RAM_2P_BRAM
	//#pragma HLS RESOURCE  variable=kernel_port  core=RAM_2P_BRAM
	//#pragma HLS RESOURCE  variable=output_port  core=RAM_2P_BRAM

#pragma HLS INTERFACE  bram  port=data_port
#pragma HLS INTERFACE  bram  port=kernel_port
#pragma HLS INTERFACE  bram  port=output_port

	interface intf1;
	dnn_layer dnn1;

	in_t  input_array   [L1_CONV::w_in][L1_CONV::w_in];
	in_t  input_array2  [L1_CONV::lyr_in][L1_CONV::w_in][L1_CONV::w_in];

	ker_t L1_ker [L1_CONV::lyr_out][L1_CONV::w2_ker];
	res_t conv2_out    [L1_CONV::lyr_out][L1_CONV::w_out][L1_CONV::w_out];



//#pragma HLS ARRAY_RESHAPE variable=input_array cyclic factor=2 dim=1
//#pragma HLS ARRAY_RESHAPE variable=L1_ker cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=L1_ker complete dim=1
//#pragma HLS ARRAY_PARTITION variable=L1_ker complete dim=1


	intf1.read_kernel2D  <ker_t,Pcfg_CONV, L1_CONV>   (kernel_port,L1_ker);
	intf1.read_input3D   <in_t ,Pcfg_CONV, L1_CONV>   (data_port,input_array2);
	dnn1.Conv_DualPortT2 <in_t ,ker_t   , res_t, mid_t, L1_CONV>  (input_array2 , L1_ker , conv2_out);
	intf1.write_result2D <res_t,Pcfg_CONV, L1_CONV>   (output_port,conv2_out);



	//intf1.read_kernel2D  <ker_t,Pcfg_CONV, L1_CONV>   (kernel_port,L1_ker);
	//intf1.read_input3D   <in_t ,Pcfg_CONV, L1_CONV>   (data_port,input_array2);
	//dnn1.Conv_DualPortT2 <in_t ,ker_t   , res_t, mid_t, L1_CONV>  (input_array2 , L1_ker , conv2_out2);
	//intf1.write_result3D <res_t,Pcfg_CONV, L1_CONV>   (output_port,conv2_out2);

}


*/


