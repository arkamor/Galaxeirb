#include "cuda.h"
#include "functions.h"

struct position{
	float pox; //4o
	float poy; //4o
	float poz; //4o
};	

struct velocite{
	float vex; //4o
	float vey; //4o
	float vez; //4o
};

struct star{ //44o
	float mas; //4o
	struct position pos; //16o
	struct velocite vel; //16o
	int galax; //1 milk 0 andro //8o
};

__global__ void kernel_acc_calc(struct star *ptr){
	struct star p[Nb_de_pts];
	memcpy(&p, (const char *)ptr, Nb_de_pts * sizeof(p));
	float dist;
	for (int i=0;i<Nb_de_pts;i++){ // 1st for each particle ACC calc
		for(int j=0;j<Nb_de_pts;j++){
			if(j != i){
				dist = sqrtf(_Square(p[j].pos.pox - p[i].pos.pox)+_Square(p[j].pos.poy - p[i].pos.poy)+_Square(p[j].pos.poz - p[i].pos.poz));
				if(dist < 1.0) dist = 1.0;
				p[i].vel.vex += ((p[j].pos.pox - p[i].pos.pox) * mass_factor_X_damping_factor * (1/(_Cube(dist))) * p[j].mas);
				p[i].vel.vey += ((p[j].pos.poy - p[i].pos.poy) * mass_factor_X_damping_factor * (1/(_Cube(dist))) * p[j].mas);
				p[i].vel.vez += ((p[j].pos.poz - p[i].pos.poz) * mass_factor_X_damping_factor * (1/(_Cube(dist))) * p[j].mas);
			}
		}
	}
	memcpy(ptr, p, Nb_de_pts * sizeof(p));
}

void acc_calc(int nblocks, int nthreads, struct star * in_addr){
	kernel_acc_calc<<<nblocks, nthreads>>>(in_addr);
}



/*
__global__ void kernel_saxpy( int n, float a, float * x, float * y, float * z ) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < n ) { 
		z[i] = a * x[i] + y [i];
	}
}

void saxpy(int nblocks, int nthreads, int n, float a, float * x, float * y, float * z){
	kernel_saxpy<<<nblocks, nthreads>>>(n, a, x, y, z);
}*/