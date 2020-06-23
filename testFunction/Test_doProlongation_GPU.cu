/*
 nvcc -arch=compute_52 -code=sm_52 -O3 --compiler-options -fopenmp Test_getSource_GPU.cu
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

__align__(8) texture<float> texMem_float;

__global__ void ker_Zoom_GPU(int N, float h_n, int M, float h_m, float *U_m){

	int index_m = blockDim.x * blockIdx.x + threadIdx.x;
	int index_n;
	int ix_m, iy_m;
	int ix_n, iy_n;
	float a, c;		// the ratio of the coarse grid point to the first met lower left fine grid index in x-dir, y-dir
					// Should be between 0 <= a,c < 1
	float b, d;		// ratio
	float bl, br, tl, tr; 	// value of the bottom left/right and top left/right

	while( index_m < M*M ){
		// Parse the index_m
		ix_m = index_m % M;
		iy_m = index_m / M;

		// Ignore the boundary
		if( (ix_m == 0) || (ix_m == M-1) || (iy_m == 0) || (iy_m == M-1) ){
			// Do nothing
		}
		else{
			// Calculate the ratio and the lower left grid_n index
			ix_n = (int) floorf((float)ix_m * h_m / h_n);
			iy_n = (int) floorf((float)iy_m * h_m / h_n);
			index_n = ix_n + iy_n * N;
			a = fmodf((float)ix_m * h_m, h_n) / h_n;
			c = fmodf((float)iy_m * h_m, h_n) / h_n;
			b = 1.0 - a;
			d = 1.0 - c;

			// Fetch the value
			bl = tex1Dfetch(texMem_float, index_n);
			br = tex1Dfetch(texMem_float, index_n + 1);
			tl = tex1Dfetch(texMem_float, index_n + N);
			tr = tex1Dfetch(texMem_float, index_n + N + 1);

			// Zooming and store inside U_m
			U_m[index_m] = b * d * bl + a * d * br + c * b * tl + a * c * tr;
		}

		// Stride
		index_m = index_m + blockDim.x * gridDim.x;
	}
	
}

void doProlongation_GPU(int N, double *U_c, int M, double *U_f){

	// Settings
	double h_c = 1.0 / (double) (N - 1);	// spacing in coarser grid
	double h_f = 1.0 / (double) (M - 1);	// spacing in finer grid

	// Settings for GPU
	int blocksPerGrid = 10;
	int threadsPerBlock = 10;
	float *d_Uc, *d_Uf;
	float *h_Uc, *h_Uf;
	
	/*
	CPU Part
	*/
	// Allocate host memory
	h_Uc = (float*) malloc(N * N * sizeof(float));
	h_Uf = (float*) malloc(M * M * sizeof(float));

	// Transfer data from double to float
	#	pragma omp parallel for
	for(int i = 0; i < N*N; i = i+1){
		h_Uc[i] = (float) U_c[i];
	}

	/*
	GPU Part
	*/
	// Allocate device memory
	cudaMalloc((void**)&d_Uc, N * N * sizeof(float));
	cudaMalloc((void**)&d_Uf, M * M * sizeof(float));

	// Bind d_Uc to texture memory
	cudaBindTexture(NULL, texMem_float, d_Uc, N * N * sizeof(float));

	// Copy data to device memory and initialize d_Uf as zeros
	cudaMemcpy(d_Uc, h_Uc, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(d_Uf, 0.0, M * M * sizeof(float));

	free(h_Uc);		// h_Uc is no longer needed
	
	// Call the kernel
	ker_Zoom_GPU <<< blocksPerGrid, threadsPerBlock >>> (N, (float) h_c, M, (float) h_f, d_Uf);

	// Copy data back to host memory
	cudaMemcpy(h_Uf, d_Uf, M * M * sizeof(float), cudaMemcpyDeviceToHost);

	// Unbind texture memory and free the device memory
	cudaUnbindTexture(texMem_float);
	cudaFree(d_Uf);
	cudaFree(d_Uc);

	// Transfer data from float to double
	#	pragma omp parallel for
	for(int i = 0; i < M*M; i = i+1){
		U_f[i] = (double) h_Uf[i];
	}

	free(h_Uf);	
}

void doRestriction(int N, double *U_f, int M, double *U_c){
	int ix_f, iy_f; 	// the lower left index needed from the fine grid
	double a, c;		// the ratio of the coarse grid point to the first met lower left fine grid index in x-dir, y-dir
						// Should be between 0 <= a,c < 1
	double b, d;		// ratio 
	int index_c;		// index of the coarse grid inside the 1D-array address
	int index_f; 		// index of the fine grid inside the 1D-array address
	double h_f = 1.0 / (double)(N - 1);			// the delta x of the discretized fine grid.
	double h_c = 1.0 / (double)(M - 1);			// the delta x of the discretized coarse grid.

	// Initialize Coarse Grid, set 0 to all
	memset(U_c, 0.0, M * M * sizeof(double));

	// Run through the coarse grid and do restriction, 
	// but without the boundary, since it is all "0"
	
	#	pragma omp for private(ix_f, iy_f, a, c, b, d, index_c, index_f)
	for(int iy_c = 1; iy_c < (M-1); iy_c = iy_c+1){
		for(int ix_c = 1; ix_c < (M-1); ix_c = ix_c+1){
			
			// Calculate the ratio and the lower left fine grid index
			ix_f = (int) floor((double)ix_c * h_c / h_f);
			iy_f = (int) floor((double)iy_c * h_c / h_f);
			a = fmod((double)ix_c * h_c, h_f) / h_f;
			c = fmod((double)iy_c * h_c, h_f) / h_f;
			b = 1.0 - a;
			d = 1.0 - c;

			// DEBUG info
			// printf("ix_f = %d, iy_f = %d\n", ix_f, iy_f);
			// printf("a = %.3lf, b = %.3lf\n", a, b);
			// printf("c = %.3lf, d = %.3lf\n", c, d);

			// Calculate the coarse grid value
			index_f = ix_f + iy_f * N;
			index_c = ix_c + iy_c * M;
			U_c[index_c] = b * d * U_f[index_f] + a * d * U_f[index_f+1] + c * b * U_f[index_f+N] + a * c * U_f[index_f+N+1];
		}
	}

}


void doPrint(int N, double *U){
	for(int j = N-1; j >= 0; j = j-1){
		for(int i = 0; i < N; i = i+1){
			printf("%2.3e ", U[i+N*j]);
		}
		printf("\n");
	}
}

int main( int argc, char *argv[] ){
	int N, M;

	double *Uf, *Uc;
	double *Uf_CPU;

	if(argc != 3){
		N = 4;
		M = 8;
	}
	else{
		N = atoi(argv[1]);
		M = atoi(argv[2]);
	}

	Uc = (double*) malloc(N * N * sizeof(double));
	Uf = (double*) malloc(M * M * sizeof(double));
	Uf_CPU = (double*) malloc(M * M * sizeof(double));
	
	printf("~ Initialize the test doRestriction_GPU ~\n");
	for(int i = 0; i < N; i = i+1){
		for(int j = 0; j < N; j = j+1){
			Uc[i + N*j] = (double) (i + j);
		}
	}
	
	printf("Uc = \n");
	doPrint(N, Uc);

    cudaEvent_t start, stop;
    cudaError_t err;
    err = cudaSetDevice( 0 );
    if(err != cudaSuccess){
        printf("Cannot select GPU\n");
        exit(1);
    }

    float gpu_time_use;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);	
    
    printf("~Run the test~\n");

    doProlongation_GPU(N, Uc, M, Uf);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&gpu_time_use, start, stop);

	printf("Uf = \n");
	doPrint(M, Uf);
	
	printf("GPU: TimeUsed = %lf\n", gpu_time_use);
	printf("\n");

	float cpu_time_use;
	
	cudaEventRecord(start, 0);
	
	doRestriction(N, Uc, M, Uf_CPU);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&cpu_time_use, start, stop);
	
	printf("Uf_CPU = \n");
	doPrint(M, Uf_CPU);
	printf("CPU: TimeUsed = %lf\n", cpu_time_use);	
	printf("SpeedUp = %lf\n", cpu_time_use / gpu_time_use);

	double diff = 0.0;
	#	pragma omp parallel for
	for(int i = 0; i < M*M; i = i+1){
		diff = diff + abs(Uf[i] - Uf_CPU[i]);
	}

	printf("norm(U - U_CPU) = %lf\n", diff / (double)(M*M));

	return 0;
}
