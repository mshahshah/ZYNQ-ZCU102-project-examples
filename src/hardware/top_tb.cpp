
#include "top.h"
#include "monitors.h"
#define IN_DATA_FILENAME        "../../../../test_files/in_data.txt"
#define IN_COEFF_FILENAME1      "../../../../test_files/kernels.txt"
#define OUTFILENAME         	"../../../../test_files/out_data.txt"

FILE *in_filter_file, *in_coeff_file1;
FILE  *out_filter_file;


int main()
{
	ofstream result;
	int i;
	int retval=0;

	in_t indata_read[PORT_CFG::in_mem];
	ker_t incoef1_read[PORT_CFG::ker_mem];
	res_t results_write[PORT_CFG::out_mem];

	in_filter_file = fopen(IN_DATA_FILENAME, "rt");
	in_coeff_file1 = fopen(IN_COEFF_FILENAME1, "rt");

	cout <<"Read Input and Coefficients from file "<<endl;

	for(i=0; i <PORT_CFG::in_mem; i ++){
		fscanf(in_filter_file, "%d", &indata_read[i]);
		//cout<<"data ["<<i<<"] = "<<indata_read[i]<<endl;
	}
	cout<<endl;
	fclose(in_filter_file);

	for(i=0; i <PORT_CFG::ker_mem; i ++){
		fscanf(in_coeff_file1, "%d", &incoef1_read[i]);
		//cout<<"kernel ["<<i<<"] = "<<incoef1_read[i]<<endl;
	}
	cout<<endl;
	fclose(in_coeff_file1);

	cout<<"clear the HW output array."<<endl;
	for(i=0; i <PORT_CFG::out_mem; i ++){
		results_write[i]=0;
	}
	cout<<"Running the dnn_top module."<<endl;



	//print_configs();
	//dnn_top(indata_read, incoef1_read, results_write);
	//dnn_topDual(indata_read, incoef1_read, results_write);
	UUT(indata_read, incoef1_read, results_write);



	cout<<"Computation on dnn_top is done."<<endl;

	ofstream outfile;
	outfile.open(OUTFILENAME);
	for(i=0; i <PORT_CFG::out_mem; i ++){
		cout<< "The output is = " << setw(10) <<results_write[i] <<endl;
		outfile <<results_write[i] <<endl;
	}
	cout<<endl;
	outfile.close();

	return retval;
}
