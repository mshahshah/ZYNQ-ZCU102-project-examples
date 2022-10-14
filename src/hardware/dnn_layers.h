#ifndef __DNN_LAYERS_H__
#define __DNN_LAYERS_H__
#include "top.h"
#include "dnn_configs.h"
#include "monitors.h"



class interface {

public:

	template<class data_T, typename P_CFG, typename CFG>
	void read_input(data_T data_port[P_CFG::in_mem], data_T input_array[CFG::lyr_in])
	{
		Lcol: for(int col=0 ; col < CFG::lyr_in ; ++col){
			input_array[col] = data_port[col];
		}
	}

	template<class data_T, typename P_CFG, typename CFG>
	void read_kernel(int baseAddr, data_T kernel_port[P_CFG::ker_mem], data_T kernel_array[CFG::fc_ker])
	{
		Lrow: for(int varrow=0 ; varrow < CFG::lyr_out ; ++varrow)
			Lcol: for(int col=0 ; col < CFG::lyr_in ; ++col){
				kernel_array[varrow * CFG::lyr_in + col] = kernel_port[baseAddr + varrow*CFG::lyr_in+col];
			}

	}

	template<class data_T, typename P_CFG, typename CFG>
	void write_result(data_T output_port[P_CFG::out_mem], data_T output_array[CFG::lyr_out])
	{
		Lrow: for(int varrow=0 ; varrow < CFG::lyr_out ; ++varrow){
			output_port[varrow] = output_array[varrow];
		}
	}

	template<class data_T, typename P_CFG, typename CFG>
	void read_kernel2D(int baseAddr, data_T kernel_port[P_CFG::ker_mem], data_T kernel_array[CFG::lyr_out][CFG::w2_ker])
	{
		Llyr: for(int lyr=0 ; lyr < CFG::lyr_out ; ++lyr)
			Lker: for(int kPtr=0 ; kPtr < CFG::w2_ker ; ++kPtr){
				kernel_array[lyr][kPtr] = kernel_port[baseAddr + CFG::w2_ker * lyr + kPtr];
			}
	}

	template<class data_T, typename P_CFG, typename CFG>
	void read_kernel3D(int baseAddr, data_T kernel_port[P_CFG::ker_mem], data_T kernel_array[CFG::lyr_out][CFG::lyr_in][CFG::w2_ker])
	{
		LlyrI: for(int lyrO=0 ; lyrO < CFG::lyr_out ; ++lyrO)
         LlyrO: for(int lyrI=0 ; lyrI < CFG::lyr_out ; ++lyrI)
            Lker: for(int kPtr=0 ; kPtr < CFG::w2_ker ; ++kPtr){
               int mem_index = CFG::w2_ker * lyrI + kPtr + CFG::lyr_out*CFG::w2_ker*lyrO;
               kernel_array[lyrO][lyrI][kPtr] = kernel_port[baseAddr + mem_index];
//               printf("Ker3D : [%2d][%3d][%3d] <- [%5d] = %5d\n", lyrO, lyrI, kPtr, int(baseAddr + mem_index), int(kernel_array[lyrO][lyrI][kPtr]));
			}
	}
   
	template<class data_T, typename P_CFG, typename CFG>
	void read_input3D(data_T data_port[P_CFG::in_mem], data_T input_array[CFG::lyr_in][CFG::w_in][CFG::w_in])
	{
		Llyr: for(int lyr=0 ; lyr < CFG::lyr_in ; ++lyr)
			Lrow: for(int varrow=0 ; varrow < CFG::w_in ; ++varrow)
				Lcol: for(int col=0 ; col < CFG::w_in ; ++col){
					int mem_index = lyr*CFG::w_in*CFG::w_in + col*CFG::w_in + varrow;
					//printf("In3D : [%2d][%3d][%3d] <- %5d\n", lyr, varrow, col, int(data_port[mem_index]));
					input_array[lyr][varrow][col] = data_port[mem_index];
				}
		}

	template<class data_T, typename P_CFG, typename CFG>
	void write_result3D(data_T output_port[P_CFG::out_mem], data_T output_array[CFG::lyr_out][CFG::w_out][CFG::w_out])
	{
		Llyr: for(int lyr=0 ; lyr < CFG::lyr_out ; ++lyr)
			Lrow: for(int varrow=0 ; varrow < CFG::w_out ; ++varrow)
				Lcol: for(int col=0 ; col < CFG::w_out ; ++col){
					int mem_index = lyr * CFG::w_out * CFG::w_out + CFG::w_out * varrow + col;
					output_port[mem_index] = output_array[lyr][varrow][col];
			}
	}

};



class dnn_layers{

public:

// ============================================================================================================================
// ============================================================================================================================
	template<class int_T, class ker_T, class res_T, class mid_T, typename CFG>
	void conv_3DT1(int LyrID, int_T Din[CFG::lyr_in][CFG::w_in][CFG::w_in], ker_T kernels[CFG::lyr_out][CFG::lyr_in][CFG::w2_ker], res_T Dout[CFG::lyr_out][CFG::w_out][CFG::w_out])
	{
#pragma HLS INLINE off
		int temp[CFG::lyr_in], Rptr, Cptr, kerindx;
		int varrow, col, lyrO, lyrI, i, j, blo;
      //printf("\nlyrI Rptr Cptr     Din    lyrO  lyrI i    j     kernels\n");
      //printf("---------------------------------------------------------\n");

         //Lblo: for (blo = 0 ; blo < CFG::batch_out ; ++blo)
         Lrow:  for (varrow = 0  ; varrow  < CFG::w_out  ; ++varrow  )
            Lcol:  for (col = 0  ; col  < CFG::w_out  ; ++col  )
               LlyrO: for (lyrO = 0 ; lyrO < CFG::lyr_out ; ++lyrO )
                  LlyrI: for (lyrI = 0 ; lyrI < CFG::lyr_in  ; ++lyrI )
                     Lwi:   for (i = 0    ; i    < CFG::w_ker   ; ++i    )    // on rows
                        Lwj:   for (j = 0    ; j    < CFG::w_ker   ; ++j    )
                        {  // on columns
                           kerindx  = j + i*CFG::w_ker;
                           Rptr  = varrow * CFG::stride + i ;
                           Cptr  = col * CFG::stride + j ;
                           if   ((i==0) && (j==0))    temp[lyrI] =  ( Din[lyrI][Rptr][Cptr] * kernels[lyrO][lyrI][kerindx]) >> 4;
                           else                       temp[lyrI] =  ( temp[lyrI] + Din[lyrI][Rptr][Cptr] * kernels[lyrO][lyrI][kerindx] ) >> 4;

                           if ((lyrI == CFG::lyr_in-1) && (i==CFG::w_ker-1) && (j==CFG::w_ker-1))
                              {
                              if (temp[lyrI] > 0)   Dout[lyrO][varrow][col] = temp[lyrI];
                              else                  Dout[lyrO][varrow][col] = 0;
                              }

                           //printf("conv_3DT1: Din[%2d][%3d][%3d] : %7d * Ker[%2d][%2d][%3d] : %7d = ", lyrI, Rptr, Cptr, int(Din[lyrI][Rptr][Cptr]), lyrO, lyrI, kerindx, int(kernels[lyrO][lyrI][kerindx]));
                           //printf("conv_3DT1: Dout[%2d][%3d][%3d] = %7d : %7d\n", lyrO, varrow, col, int(Dout[lyrO][varrow][col]), int(temp[lyrI]));
                           }

#ifdef CONV_PRINT
		print_3d <res_t> ("CONVin",  LyrID, CFG::lyr_in , CFG::w_in  , CFG::w_in , Din[0][0]);
		print_3d <res_t> ("CONVout", LyrID, CFG::lyr_in , CFG::w_out , CFG::w_out, Dout[0][0]);
		print_3d <res_t> ("CONVker", LyrID, CFG::lyr_out, CFG::lyr_in, CFG::w2_ker, kernels[0][0]);
#endif
	}
   
// ============================================================================================================================
// ============================================================================================================================
	template<class int_T, class ker_T, class res_T, class mid_T, typename CFG>
	void conv_3DT2(int LyrID, int_T Din[CFG::lyr_in][CFG::w_in][CFG::w_in], ker_T kernels[CFG::lyr_out][CFG::w2_ker], res_T Dout[CFG::lyr_out][CFG::w_out][CFG::w_out])
	{
#pragma HLS INLINE off
		int temp, Rptr1, Cptr1, Rptr3, Cptr3;
		int_T inW2   [CFG::lyr_in][CFG::w2_ker];
		int_T MultW3 [CFG::lyr_in][CFG::lyr_out][CFG::w2_ker];
		int_T MultW2 [CFG::lyr_out][CFG::w2_ker];

		Lrow: for (int varrow = CFG::rc_strt; varrow < CFG::rc_end ; varrow += CFG::stride) {
		Lcol: for (int col = CFG::rc_strt; col < CFG::rc_end ; col += CFG::stride) {
				LlyrI1:  for (int lyr = 0 ; lyr < CFG::lyr_in ; ++lyr )    // on layer
				Lwi1:    for (int i = 0   ; i   < CFG::w_ker  ; ++i   )    // on rows
				Lwj1:    for (int j = 0   ; j   < CFG::w_ker  ; ++j   ) {  // on columns
					Rptr1  = varrow + i ;
					Cptr1  = col + j ;
					inW2[lyr][CFG::w_ker * i + j] = Din[lyr][Rptr1][Cptr1];
					}
// -----------------------------------------------------------------------------------------------
				LlyrO2: for (int lyrO = 0  ;  lyrO < CFG::lyr_out ; ++lyrO ){
				Lker2:  for (int ker2 = 0  ;  ker2 < CFG::w2_ker  ; ++ker2 ){
				LlyrI2: for (int lyrI = 0  ;  lyrI < CFG::lyr_in  ; ++lyrI ){
#pragma HLS ALLOCATION instances=mul limit=100 operation
						MultW3[lyrI][lyrO][ker2] = inW2[lyrI][ker2] * kernels[lyrO][ker2];
						}
					temp = 0;
					LlyrI3:  for (int lyrI = 0   ;  lyrI < CFG::lyr_in  ;  ++lyrI )
						temp = temp + (MultW3[lyrI][lyrO][ker2]>>16);

					MultW2[lyrO][ker2] = temp;
				}
			}

// -----------------------------------------------------------------------------------------------
				LlyrO4: for (int lyrO = 0 ; lyrO < CFG::lyr_out ; ++lyrO ){
				Lwi2:   for (int i = 0    ; i   < CFG::w_ker    ; ++i    )      // on rows
				Lwj2:   for (int j = 0    ; j   < CFG::w_ker    ; ++j    ) {    // on columns
					Rptr3  = varrow + i;
					Cptr3  = col + j;
					Dout[lyrO][Rptr3][Cptr3] = MultW2[lyrO][CFG::w_ker * i + j] ;
					}
				}
			}
		}

#ifdef CONV_PRINT
		print_3d <res_t> ("CONVin",  LyrID, CFG::lyr_in , CFG::w_in  , CFG::w_in , Din[0][0]);
		print_3d <res_t> ("CONVout", LyrID, CFG::lyr_in , CFG::w_out , CFG::w_out, Dout[0][0]);
		print_3d <res_t> ("CONVker", LyrID, CFG::lyr_out, CFG::lyr_in, CFG::w2_ker, kernels[0][0]);
#endif
	}

// ============================================================================================================================
// ============================================================================================================================
	template<class int_T, class res_T, typename CFG>
	void ds_3DT1(int LyrID, int_T Din[CFG::lyr_in][CFG::w_in][CFG::w_in], res_T Dout[CFG::lyr_out][CFG::w_out][CFG::w_out])
	{
		#pragma HLS INLINE off
		int Rptr1, Cptr1;
		int_T value_max[CFG::lyr_in];

      Lrow: for (int varrow = 0 ; varrow <  CFG::w_out  ; ++varrow )
      Lcol: for (int col = 0 ; col <  CFG::w_out  ; ++col )
	   Lwi: for (int i = 0   ; i   <  CFG::w_ker  ; ++i   )    // on rows
	   Lwj: for (int j = 0   ; j   <  CFG::w_ker  ; ++j   )  // on columns
      Llyr: for (int lyr = 0 ; lyr <  CFG::lyr_in ; ++lyr )
               {
					Rptr1  = varrow * CFG::stride + i ;
					Cptr1  = col * CFG::stride + j ;
               if ( (i==0) && (j==0) )
                  value_max[lyr] = Din[lyr][Rptr1][Cptr1];
               else if(value_max[lyr] < Din[lyr][Rptr1][Cptr1])
                  value_max[lyr] = Din[lyr][Rptr1][Cptr1];

               //printf("ds_3DT1: Din[%2d][%3d][%3d] = %7d : value_max[%3d]=%7d", lyr, Rptr1, Cptr1, int(Din[lyr][Rptr1][Cptr1]), lyr, int(value_max[lyr]));

               if ( (i == CFG::w_ker-1) && (j == CFG::w_ker-1) ){
                     Dout[lyr][varrow][col] = value_max[lyr];
                 //    printf("\t Dout[%2d][%3d][%3d] = %7d : i=%2d, j=%2d\n", lyr, varrow, col, int(Dout[lyr][varrow][col]),i,j);
               	   }
               //else
            //	   printf("\n");
      	  }
// -----------------------------------------------------------------------------------------------
#ifdef POOL_PRINT
		print_3d <res_t> ("POOLin" ,LyrID, CFG::lyr_in, CFG::w_in , CFG::w_in , Din[0][0] );
		print_3d <res_t> ("POOLout",LyrID, CFG::lyr_in, CFG::w_out, CFG::w_out, Dout[0][0]);
#endif
	}

// ============================================================================================================================
// ============================================================================================================================
	template<class int_T, class res_T, typename CFG>
	void ds_3DT2(int LyrID, int_T Din[CFG::lyr_in][CFG::w_in][CFG::w_in], res_T Dout[CFG::lyr_out][CFG::w_out][CFG::w_out])
	{
		#pragma HLS INLINE off
		int Rptr1, Cptr1;
		int_T value_avg[CFG::lyr_in];

      Lrow: for (int varrow = 0 ; varrow <  CFG::w_out  ; ++varrow )
      Lcol: for (int col = 0 ; col <  CFG::w_out  ; ++col )
	   Lwi: for (int i = 0   ; i   <  CFG::w_ker  ; ++i   )    // on rows
	   Lwj: for (int j = 0   ; j   <  CFG::w_ker  ; ++j   )  // on columns
      Llyr: for (int lyr = 0 ; lyr <  CFG::lyr_in ; ++lyr )
               {
					Rptr1  = varrow * CFG::stride + i ;
					Cptr1  = col * CFG::stride + j ;
               if ( (i==0) && (j==0) )
                  value_avg[lyr] = Din[lyr][Rptr1][Cptr1];
               else
                  value_avg[lyr] = value_avg[lyr] + Din[lyr][Rptr1][Cptr1];
               
               //printf("ds_3DT1: Din[%2d][%3d][%3d] = %7d : value_avg[%3d]=%7d", lyr, Rptr1, Cptr1, int(Din[lyr][Rptr1][Cptr1]), lyr, int(value_max[lyr]));

               if ( (i == CFG::w_ker-1) && (j == CFG::w_ker-1) ){
                     Dout[lyr][varrow][col] = value_avg[lyr] / CFG::w2_ker;
                 //    printf("\t Dout[%2d][%3d][%3d] = %7d : i=%2d, j=%2d\n", lyr, varrow, col, int(Dout[lyr][varrow][col]),i,j);
               	   }
               //else
            //	   printf("\n");
      	  }
// -----------------------------------------------------------------------------------------------
#ifdef POOL_PRINT
		print_3d <res_t> ("POOLin" ,LyrID, CFG::lyr_in, CFG::w_in , CFG::w_in , Din[0][0] );
		print_3d <res_t> ("POOLout",LyrID, CFG::lyr_in, CFG::w_out, CFG::w_out, Dout[0][0]);
#endif
	}
   
// ============================================================================================================================
// ============================================================================================================================

	template<class int_T, typename CFG>
	void conv2fc(int LyrID, int_T Din[CFG::lyr_out][CFG::w_out][CFG::w_out], int_T Dout[CFG::lyr_out*CFG::w_out*CFG::w_out])
	{
		#pragma HLS INLINE off
		int ptr;

		Llyr: for (int lyr = 0; lyr < CFG::lyr_out ; ++lyr )
		Lrow: for (int varrow = 0; varrow < CFG::w_out   ; ++varrow )
		Lcol: for (int col = 0; col < CFG::w_out   ; ++col ){
		    ptr=lyr*CFG::w_out*CFG::w_out + varrow*CFG::w_out + col;
			Dout[ptr] = Din[lyr][varrow][col];
			//printf("conv2fc: Dout[%4d] = %7d\n", ptr, int(Dout[ptr]));
		}
	}

// ============================================================================================================================
// ============================================================================================================================
    template<class int_T, class ker_T, class res_T, class mid_T, typename CFG>
    void fc_T1(int LyrID, int_T Din[CFG::lyr_in], ker_T fc_ker[CFG::fc_ker], res_T Dout[CFG::lyr_out])
   {
	   #pragma HLS INLINE off
      mid_T temp=0,mult_res[CFG::col_max];
      int in_indx, ker_indx,out_indx;
      
      Lb_out: for (int b_out = 0 ; b_out < CFG::batch_out ; ++b_out ) {
      Lrow:   for (int varrow = 0   ; varrow   < CFG::row_max   ; ++varrow   ) {
      Lb_in:  for (int b_in = 0  ; b_in  < CFG::batch_in  ; ++b_in  ) {
      Lcol:   for (int col = 0   ; col   < CFG::col_max   ; ++col   ) {
                  in_indx  = b_in * CFG::col_max + col ;
                  out_indx = b_out* CFG::row_max + varrow ;
                  //ker_indx = out_indx * CFG::lyr_in + in_indx;
                  ker_indx = varrow * CFG::col_max + col; // regardless of batch size for smaller memories

                  mult_res[col] = Din[in_indx] * fc_ker[ker_indx];
                  //printf("conv2fc: Din[%3d]:%7d * ker[%3d]:%7d ", in_indx, int(Din[in_indx]), ker_indx, int(fc_ker[ker_indx]));
                  if(col == (CFG::col_max - 1)){
                	  temp += mult_res[col];
                	  if (temp > 0)
                	    Dout[out_indx] = temp>>16;
                	  else
                	      Dout[out_indx] = 0;
                	  //printf("\ttemp = %7d , Dout[%3d] = %7d\n", int(temp), out_indx, int(Dout[out_indx]));
                     temp = 0;
                     }
                  else{
                	  temp += mult_res[col];
                	  //printf("\n");
                  }
/*
   #ifdef FC_PRINT
   cout << "   Row*Col : ["<< setw(3) << ker_indx <<"]["<< setw(3)<< in_indx  <<"] :";
   cout << setw(7)<< fc_in[in_indx] << " * " << setw(7) << fc_ker[ker_indx] << " = " << setw(10) << (mult_res[in_indx]>>16) << " : sum = " << setw(10) << (temp>>16) << endl;
   if(col == (CFG::col_max - 1))
   cout<<"the result on ["<<out_indx<<"] = " << fc_out[out_indx]<< endl;
   #endif
   */
                  }
               }
            }
         }

#ifdef FC_PRINT
      print_1d <res_t> ("FCin" , LyrID , CFG::lyr_in , Din);
      print_1d <res_t> ("FCker", LyrID , CFG::fc_ker , fc_ker);
      print_1d <res_t> ("FCout", LyrID , CFG::lyr_out, Dout);
#endif
   }

// ============================================================================================================================
// ============================================================================================================================
    template<class int_T, class ker_T, class res_T, class mid_T, typename CFG>
    void fc_T2(int LyrID, int_T Din[CFG::lyr_in], ker_T fc_ker[CFG::fc_ker], res_T Dout[CFG::lyr_out])
   {
      mid_T temp=0,mult_res[CFG::col_max];
      int in_indx, ker_indx,out_indx;
      
      Lb_in:  for (int b_in = 0  ; b_in  < CFG::batch_in  ; ++b_in  ) {
      Lb_out: for (int b_out = 0 ; b_out < CFG::batch_out ; ++b_out ) {
      Lrow:   for (int varrow = 0   ; varrow   < CFG::row_max   ; ++varrow   ) {
      Lcol:   for (int col = 0   ; col   < CFG::col_max   ; ++col   ) {
                  in_indx  = b_in * CFG::col_max + col ;
                  ker_indx = b_out* CFG::row_max * CFG::col_max + varrow * CFG::col_max + col ;
                  out_indx = b_out* CFG::row_max + varrow ;

                  mult_res[col] = Din[in_indx] * fc_ker[ker_indx];

                  temp += mult_res[col];
                  if(col == (CFG::col_max - 1)){
                	  Dout[out_indx] = temp>>16;
                     temp = 0;
                     }
/*
   #ifdef FC_PRINT
   cout << "   Row*Col : ["<< setw(3) << ker_indx <<"]["<< setw(3)<< in_indx  <<"] :";
   cout << setw(7)<< fc_in[in_indx] << " * " << setw(7) << fc_ker[ker_indx] << " = " << setw(10) << (mult_res[in_indx]>>16) << " : sum = " << setw(10) << (temp>>16) << endl;
   if(col == (CFG::col_max - 1))
   cout<<"the result on ["<<out_indx<<"] = " << fc_out[out_indx]<< endl;
   #endif
   */
                  }
               }
            }
         }

#ifdef FC_PRINT
      print_1d <res_t> ("FCin" , LyrID , CFG::lyr_in , Din);
      print_1d <res_t> ("FCker", LyrID , CFG::fc_ker , fc_ker);
      print_1d <res_t> ("FCout", LyrID , CFG::lyr_out, Dout);
#endif
   }
    /*

   template<class data_T, static const unsigned len>
   void copy_1d_FF(data_T din[len], data_T dout[len]) {
      int i;
      dcopy:for (i=0;i<len; i++) 
         dout[i] = din[i];
   }

   template<class data_T, static const unsigned len>
   void copy_1d_HP(data_T din[len], data_T dout[len]) {
      int i;
      dcopy:for (i=0;i<len; i++) 
         dout[i] = din[i];
   }

   template<class int_T, class ker_T, class res_T, class mid_T, static const unsigned len>
   res_T multiply_1D(int_T din[len],ker_T ker[len]) {
      int i,j;
      res_T temp[len], sum;
      MULT:for (i=0;i<len; i++) {
         temp[i] = din[i]*ker[i];
         //printf("mult_copy : [%d]*[%d]=%d ; sum=%d\n",I1[i],I2[i],temp , sum);
      }
      sum=0;
      SUN:for (j=0; j<len; j++)
         sum += temp[j]>>8;

      //printf("out = %d\n",sum);
      return sum;
   }

// ============================================================================================================================
// ============================================================================================================================
   template<class int_T, class ker_T, class res_T, class mid_T, typename CFG>
   void fc_T3(int LyrID, int_T Din[CFG::lyr_in], ker_T fc_ker[CFG::fc_ker], res_T Dout[CFG::lyr_out])
   {
      int lim = CFG::lyr_out;
      int m = 0;
      int_T t1[CFG::lyr_in];
      ker_T t2[CFG::lyr_in];
      copy_1d_FF <int_T, CFG::lyr_in> (Din,t1);

      main:for (m = 0 ; m < CFG::lyr_out; m ++){
         copy_1d_HP  <ker_T , CFG::lyr_in> (fc_ker,t2);
         Dout[m] = multiply_1D <int_T , ker_T, res_T , mid_T , CFG::lyr_in> (t1,t2);
         //printf("\tY[m] = %d\n", dout[m]);

      }

#ifdef FC_PRINT
      print_1d ("fc_in" , LyrID , CFG::lyr_in , Din);
      print_1d ("fc_ker", LyrID , CFG::fc_ker , fc_ker);
      print_1d ("fc_out", LyrID , CFG::lyr_out, Dout);
#endif
   }
   */

};


#endif // __MATRIXMUL_H__ not defined

