#include "GL/glew.h"
#include "cuda_runtime.h"
#include <stdio.h>

// --- DEFINITIONS -- //

#define _Square(_x) ((_x)*(_x))
#define _Cube(_x) ((_x)*(_x)*(_x))

#define Nb_de_pts      1024 //81920
#define mass_factor    10
#define time_factor    0.08
#define damping_factor 1
#define mass_factor_X_damping_factor 10

#define windows_width 1200
#define windows_height 600

#define NUM_THREADS_CPU 4

#define NUM_THREADS_GPU 512
#define NUM_BLOCKS_GPU (Nb_de_pts+(NUM_THREADS_GPU-1))/NUM_THREADS_GPU



void DrawGridXZ( float ox, float oy, float oz, int w, int h, float sz );
void ShowAxes();

bool CUDA_MALLOC( void ** devPtr, size_t size);
bool CUDA_MEMCPY( void * dst, const void * src, size_t count, enum cudaMemcpyKind kind );
