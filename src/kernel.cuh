#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include "cuda.h"

void acc_calc(int nblocks, int nthreads, int Nb_de_pts, float *deviceIn_mas, float *deviceIn_pox, float *deviceIn_poy, float *deviceIn_poz, float *deviceOut_acx, float *deviceOut_acy, float *deviceOut_acz);
#endif
