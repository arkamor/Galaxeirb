#include "cuda.h"
#include "functions.h"

__global__ void kernel_acc_calc(int Nb_de_pts, float *deviceIn_mas, float *deviceIn_pox, float *deviceIn_poy, float *deviceIn_poz, float *deviceOut_acx, float *deviceOut_acy, float *deviceOut_acz){ // Make calc
		
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < Nb_de_pts){
		deviceOut_acx[i] = 0.0;
		deviceOut_acy[i] = 0.0;
		deviceOut_acz[i] = 0.0;

		for(int j=0;j<Nb_de_pts;j++){
			if(j != i){
				float dist = sqrtf(_Square(deviceIn_pox[j] - deviceIn_pox[i])+_Square(deviceIn_poy[j] - deviceIn_poy[i])+_Square(deviceIn_poz[j] - deviceIn_poz[i]));
				if(dist < 1.0) dist = 1.0;
				float tmp = mass_factor_X_damping_factor * (1/_Cube(dist)) * deviceIn_mas[j];
				deviceOut_acx[i] += ((deviceIn_pox[j] - deviceIn_pox[i]) * tmp);
				deviceOut_acy[i] += ((deviceIn_poy[j] - deviceIn_poy[i]) * tmp);
				deviceOut_acz[i] += ((deviceIn_poz[j] - deviceIn_poz[i]) * tmp);
			}
		}
	}
}

void acc_calc(int nblocks, int nthreads, int Nb_de_pts, float *deviceIn_mas, float *deviceIn_pox, float *deviceIn_poy, float *deviceIn_poz, float *deviceOut_acx, float *deviceOut_acy, float *deviceOut_acz){
	kernel_acc_calc<<<nblocks, nthreads>>>(Nb_de_pts, deviceIn_mas, deviceIn_pox, deviceIn_poy, deviceIn_poz, deviceOut_acx, deviceOut_acy, deviceOut_acz);
}