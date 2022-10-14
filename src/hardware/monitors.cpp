#include "top.h"
#include "monitors.h"
#include <string>
/*
void print_configs(void)
{
	int layer_configs[6][3] = {
			{CONV_IN1, CONV_OUT1,    BASE_L1},
			{DS_IN2,   DS_OUT2,      0},
			{CONV_IN3, CONV_OUT3,    BASE_L3},
			{DS_IN4,   DS_OUT4,      0},
			{FC_IN5,   FC_OUT5,      BASE_L5},
			{FC_IN6,   FC_OUT6,      BASE_L6}
	};

	cout << "Layers config are as below : " <<endl;
	for (int i = 0 ; i < 6 ; i++ )
		cout << "  Layer "<< i << " : in=" << layer_configs[i][0] << ", out=" << layer_configs[i][1]
			 << "   Base addr = " << layer_configs[i][2]<< endl;

	cout << "--------- Memory information  ----------" << endl;
	cout << "  Total BRAM needed = " << HIGH_L6 <<endl;
	//cout << "  L1 in=" << CONV_IN1 << ", L1 out=" << CONV_OUT1 << endl;
	//cout << "  L2 in=" << DS_IN2   << ", L1 out=" << DS_OUT2   << endl;
	//cout << "  L3 in=" << CONV_IN3 << ", L2 out=" << CONV_OUT3 << endl;
	//cout << "  L4 in=" << DS_IN4   << ", L3 out=" << DS_OUT4   << endl;
}


*/

ofstream create_report_file(std::string filename,int layer_ID)
{
	std::string path = "../../../../sim/hls_out/";
	path = path + "Lyr" + to_string(layer_ID) + "_" + filename +".txt";

	cout << "HLS : file " << path << " is created" << endl;
	ofstream outfile;
	outfile.open(path);
	return outfile;

}



