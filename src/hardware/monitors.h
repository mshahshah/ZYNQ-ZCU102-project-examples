#ifndef __MONITORS_H__
#define __MONITORS_H__

#include "top.h"
#include <fstream>
#include <iostream>
#include "dnn_configs.h"
using namespace std;

//void print_configs(void);

ofstream create_report_file(std::string filename,int layer_ID);

template<class dtype>
void print_3d(std::string filename, int LyrID, int lyrs, int rows, int cols, dtype *data)
{
#ifdef ACTIVE_PRINTERS
	ofstream outfileIn  = create_report_file(filename,LyrID);
	for (int lyr = 0  ; lyr   < lyrs ; ++lyr  ){
		for (int row = 0; row < rows ; row++) {
			for (int col = 0; col < cols ; col++) {
				outfileIn << setw(9) << *(data + lyr*rows*cols + row*rows + col);
			}
			outfileIn << endl;
		}
		outfileIn << "---------------------- Layer = " << lyr << "  ------------------------" << endl;
	}
	outfileIn.close();
#endif
}


template<class dtype>
void print_2d(std::string filename, int LyrID, int rows, int cols, dtype *data)
{
#ifdef ACTIVE_PRINTERS
	ofstream outfileIn  = create_report_file(filename,LyrID);
	for (int row = 0; row < rows ; row++){
	   for (int col = 0  ; col   < cols ; ++col  )
		  outfileIn << setw(9) << *(data + row*rows + col);
	   outfileIn << endl;
	}
	outfileIn.close();
#endif
}

template<class dtype>
void print_1d(std::string filename, int LyrID, int cols, dtype *data)
{
#ifdef ACTIVE_PRINTERS
	ofstream outfileIn  = create_report_file(filename,LyrID);
	for (int col = 0  ; col   < cols ; ++col  )
		outfileIn << *(data + col)  << endl;
	outfileIn << endl;
	outfileIn.close();
#endif
}


#endif
