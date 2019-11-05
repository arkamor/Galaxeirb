#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include "cuda.h"

void acc_calc(int nblocks, int nthreads, struct star * in_addr, struct star * out_addr);


#endif
