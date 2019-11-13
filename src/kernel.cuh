#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include "cuda.h"

void acc_calc(int nblocks, int nthreads, float *deviceIn_mas, float *deviceIn_pox, float *deviceIn_poy, float *deviceIn_poz, float *deviceOut_vex, float *deviceOut_vey, float *deviceOut_vez);
#endif
