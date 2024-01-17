#include "hip/hip_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <hip/hip_runtime.h>
#include "cudautil_hip.h"
/*
*TODO: find the best BLOCK_SIZE 
*/
#define BLOCK_SIZE 8
/*
*TODO: find the best TILEDIM 
*/
#define TILE_DIM 8

/*
*TODO: Task b: Global memory MM implementation 
*/
__global__ void MM(double* a, double* b, double* c,
		int N, int REP ) {

    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;
    for(int r = 0 ; r< REP; ++r)
        if( col < N && row < N) 
        {
            for(int i = 0; i < N; i++) 
            {
                    sum += a[row * N + i] * b[i * N + col];
            }
            c[row * N + col] = sum;
        }
}

/*
*TODO: Task d: Shared memory MM implementation 
*/
__global__ void sharedTiledMM(double* a, double* b, double* c,
		int N, int REP) {
	__shared__ double aTile[TILE_DIM][TILE_DIM];
	__shared__ double bTile[TILE_DIM][TILE_DIM];
	// double aTile[TILE_DIM][TILE_DIM];
	// double bTile[TILE_DIM][TILE_DIM];
	int row = blockIdx.y* blockDim.y+ threadIdx.y; 
	int col= blockIdx.x* blockDim.x+ threadIdx.x; 
	double sum = 0; 
	
	for(int j = 0 ; j<REP ; ++j){
		for (int k = 0; k < N; k += TILE_DIM) { 
			aTile[threadIdx.y][threadIdx.x] = a[ (row * N) + k + threadIdx.x]; 
			bTile[threadIdx.y][threadIdx.x] = b[(threadIdx.y + k)*N + col]; 
			__syncthreads(); 
			for (int i = 0; i < TILE_DIM; i++) 
				sum += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];
			__syncthreads();
		} 
		c[row*N +col] = sum; 
	}
}


int main(int argc, char *argv[]) {

    /*
       +TODO: Task a: print device properties 
    */
    int device = 0;
    hipSetDevice(device);
    // PrintDeviceInfo();
    
    if (argc < 2) {
        printf("For C(NxN) = A(NxN)* B(NxN), Matrix size value N must be provided ! \n");
        exit(1);
    }

    char *pEnd;
    int N = strtol(argv[1], &pEnd, 10);
    if (errno == ERANGE) {
        printf("Problem with the first number  N .");
        exit(2);
    }
    int REP = 10;
    unsigned int grid_rows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);  

    /*Memory allocations and Initializations of maticies*/
    // double *a=(double*)malloc(sizeof(double)*N*N);
    // double *b=(double*)malloc(sizeof(double)*N*N);
    // double *c=(double*)malloc(sizeof(double)*N*N);;
    double *a, *b, *c;
    hipHostMalloc((void**)&a, sizeof(double)*N*N);
    hipHostMalloc((void**)&b, sizeof(double)*N*N);
    hipHostMalloc((void**)&c, sizeof(double)*N*N);

    double *d_a, *d_b, *d_c;
    /*
    * TODO:Task e: Use UVA for device memory  
    */ 
    hipMalloc(&d_a, sizeof(double)*N*N);
    hipMalloc(&d_b, sizeof(double)*N*N);
    hipMalloc(&d_c, sizeof(double)*N*N);

    // Initialization on CPU
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; ++i) 
        for (int j = 0; j < N; ++j) {
            a[i*N+j] = atan(i*N+j);
            b[i*N+j] = cos(i*N+j);
            c[i*N+j] = 0.0;
    }

    // Copy initial values to GPUs
    auto t2 = std::chrono::high_resolution_clock::now();
    hipMemcpy( d_a, a, sizeof(double)*N*N, hipMemcpyHostToDevice );
    hipMemcpy( d_b, b, sizeof(double)*N*N, hipMemcpyHostToDevice );
    hipMemcpy( d_c, c, sizeof(double)*N*N, hipMemcpyHostToDevice );
    

    using dsec = std::chrono::duration<double>;
    double mf = 2.0*(double)N*(double)N*(double)N*(double)REP*1.0e-6;
    
    // Compute Checksum for Simple Correctness Checks
    // double checksum = cpu_matrix_mult_checksum(a, b, N, REP);

    /*
    * Basic MM Kernel Call & Time Measurements
    */
    dim3 dimBlockMM(N,N);


    // auto t0 = std::chrono::high_resolution_clock::now();
    // MM <<<dimGrid, dimBlock >>>(d_a, d_b ,d_c, N, REP);
    // hipDeviceSynchronize();
    // auto t1 = std::chrono::high_resolution_clock::now();

    // //Calculate Flops/sec,
    // double dur = std::chrono::duration_cast<dsec>(t1-t0).count();
    // std::cout<<"MM MFlops/s(N*N="<< N*N <<" ): "<<mf/dur<<std::endl;
    // // // Copy the result back to CPU & correctness check
    // hipMemcpy( c, d_c, sizeof(double)*N*N, hipMemcpyDeviceToHost );
    // // Checksum ( N, c, checksum );

    // //reset_result_array d_c
    // #pragma omp parallel for collapse(2) schedule(static)
    // for (int i = 0; i < N; ++i) 
    //     for (int j = 0; j < N; ++j) {
    //         c[i*N+j] = 0.0;
    // }
    // hipMemcpy( d_c, c, sizeof(double)*N*N, hipMemcpyHostToDevice );
    
    /*    
    *Basic Tiled MM with Shared Memory Kernel Call & Time Measurements
    */
    // auto t2 = std::chrono::high_resolution_clock::now();
    sharedTiledMM <<<dimGrid, dimBlock >>>(d_a, d_b ,d_c, N, REP);
    hipDeviceSynchronize();
    // auto t3 = std::chrono::high_resolution_clock::now();

    //double dur = std::chrono::duration_cast<dsec>(t3-t2).count();
    //Calculate Flops/sec, Correctness Checks & Reset Result array C
    // Copy the result back to CPU & correctness check
    hipMemcpy( c, d_c, sizeof(double)*N*N, hipMemcpyDeviceToHost );
    auto t3 = std::chrono::high_resolution_clock::now();
    double dur = std::chrono::duration_cast<dsec>(t3-t2).count();
    std::cout<<"Shared Tiled MFlops/s for (N*N="<< N*N <<"): "<<mf/dur<<std::endl;
    // Checksum ( N, c, checksum );
    hipFree(a);
    hipFree(b);
    hipHostFree(c);
    // free(a);
    // free(b);
    // free(c);
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    return 0;
}

