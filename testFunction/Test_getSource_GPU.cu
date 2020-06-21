#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

__global__ void ker_Source_GPU(int N, float h, float *F, float min_x, float min_y){
	// Thread index inside the GPU kernel
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int ix, iy;
	float x, y;

	while( index < N * N){
		// Parse the index to ix, iy
		ix = index % N;
		iy = index / N;
		if( (ix == 0) || (ix == N-1) || (iy == 0) || (iy == N-1)){
            F[index] = 0.0;
        }
        else{
            // Calculate the coordinate in (x, y)
		    x = (float) ix * h + min_x;
		    y = (float) iy * h + min_y;
		    // Source from the problem
		    F[index] = 2.0 * x * (y - 1) * (y - 2.0 * x + x * y + 2.0) * expf(x - y);
        }
		
        // Stride
		index = index + blockDim.x * gridDim.x;
	}
	__syncthreads();
}

void doPrint(int N, double *U){
	for(int j = N-1; j >= 0; j = j-1){
		for(int i = 0; i < N; i = i+1){
			printf("%2.3e ", U[i+N*j]);
		}
		printf("\n");
	}
}

void getSource(int N, double L, double *F, double min_x, double min_y){

	double h = L / (double) (N-1);
	double x, y;
	int index;

	#	pragma omp parallel for private(index, x, y)
		for(int iy = 0; iy < N; iy = iy+1){
			for(int ix = 0; ix < N; ix = ix+1){
				
				index = ix + N * iy;
				x = (double)ix * h + min_x;
				y = (double)iy * h + min_y;
	
				// Source from the problem
				F[index] = 2.0 * x * (y - 1) * (y - 2.0 * x + x * y + 2.0) * exp(x - y);
			}
		}

}

int main( int argc, char *argv[] ){
	double L = 1.0;
	int N;
	double *F;
	float *f_F;

	// Settings for GPU
	int blocksPerGrid;
	int threadsPerBlock;

	if(argc != 4){
		N = 16;
		blocksPerGrid = 10;
		threadsPerBlock = 10;
	}
	else{
		N = atoi(argv[1]);
		blocksPerGrid = atoi(argv[2]);
		threadsPerBlock = atoi(argv[3]);
	}

	// TODO: Test for various functions and output time use
	//	printf("~START~\n");
	f_F = (float *)malloc(N * N * sizeof(float));
	F   = (double *)malloc(N * N * sizeof(double));
   
    cudaEvent_t start, stop;
    cudaError_t err = cudaSuccess;
    err = cudaSetDevice( 0 );
    if(err != cudaSuccess){
        printf("Cannot select GPU\n");
        exit(1);
    }

    float gpu_time_use;

    double h = L / (double)(N - 1);
	double min_x = 0.0;
	double min_y = 0.0;
	
    float *d_F;
	
    cudaEventCreate(&start);
	cudaEventCreate(&stop);

//	printf("~HERE~\n");
	cudaEventRecord(start, 0);

	cudaMalloc((void**)&d_F, N * N * sizeof(float));

	ker_Source_GPU <<< blocksPerGrid, threadsPerBlock >>> (N, (float)h, d_F, (float)min_x, (float)min_y);

	cudaMemcpy(f_F, d_F, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&gpu_time_use, start, stop);

	cudaFree(d_F);


//	omp_set_num_threads(16);
	#	pragma omp parallel for
	for(int i = 0; i < N*N; i = i+1){
		F[i] = (double) f_F[i];
	}

	doPrint(N, F);
	
	printf("GPU BlockSize = %d, GridSize = %d, TimeUsed = %lf\n", threadsPerBlock, blocksPerGrid, gpu_time_use);
//	printf("\n");

	double *F_CPU;
	float cpu_time_use;
	cudaEventRecord(start, 0);
	
	F_CPU = (double*) malloc(N * N * sizeof(double));
	
	getSource(N, L, F_CPU, min_x, min_y);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&cpu_time_use, start, stop);
//	doPrint(N, F_CPU);
	printf("CPU TimeUsed = %lf\n", cpu_time_use);	
	printf("SpeedUp = %lf\n", cpu_time_use / gpu_time_use);
	return 0;
}
