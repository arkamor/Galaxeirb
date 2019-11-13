#include "cuda.h"
#include "functions.h"

__global__ void kernel_acc_calc(float *deviceIn_mas, float *deviceIn_pox, float *deviceIn_poy, float *deviceIn_poz, float *deviceOut_vex, float *deviceOut_vey, float *deviceOut_vez){ // Make calc
		
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < Nb_de_pts){
		float dist;
		float tmp;

		float tx = 0.0;
		float ty = 0.0;
		float tz = 0.0;


		for(int j=0;j<Nb_de_pts;j++){

			if(j != i){
				dist = sqrtf(_Square(deviceIn_pox[j] - deviceIn_pox[i])+_Square(deviceIn_poy[j] - deviceIn_poy[i])+_Square(deviceIn_poz[j] - deviceIn_poz[i]));
				if(dist < 1.0) dist = 1.0;
				tmp = mass_factor_X_damping_factor * (1/_Cube(dist)) * deviceIn_mas[i];
				tx += ((deviceIn_pox[j] - deviceIn_pox[i]) * tmp);
				ty += ((deviceIn_poy[j] - deviceIn_poy[i]) * tmp);
				tz += ((deviceIn_poz[j] - deviceIn_poz[i]) * tmp);
			}
		}
		deviceOut_vex[i] = tx;
		deviceOut_vey[i] = ty;
		deviceOut_vez[i] = tz;
	}
}

void acc_calc(int nblocks, int nthreads, float *deviceIn_mas, float *deviceIn_pox, float *deviceIn_poy, float *deviceIn_poz, float *deviceOut_vex, float *deviceOut_vey, float *deviceOut_vez){
	kernel_acc_calc<<<nblocks, nthreads>>>(deviceIn_mas, deviceIn_pox, deviceIn_poy, deviceIn_poz, deviceOut_vex, deviceOut_vey, deviceOut_vez);
}