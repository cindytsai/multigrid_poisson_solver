/*
 nvcc -arch=compute_52 -code=sm_52 -O3 --compiler-options -fopenmp Test_getSource_GPU.cu
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

__global__ void Residual_GPU(int N, float h, float *U, float *F, float *D){
	// Thread index inside the GPU kernel
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int ix, iy;
	float l, r, t, d, c;	// value of the neighboring index and the center index of U

	while( index < N * N){
		// Parse the index to ix, iy
		ix = index % N;
		iy = index / N;

		// Ignore the boundary
		if( (ix == 0) || (ix == N-1) || (iy == 0) || (iy == N-1) ){
			D[index] = 0.0;
		}
		else{
			c = U[index];
			l = U[(ix-1) + N*iy];
			r = U[(ix+1) + N*iy];
			t = U[ix + N*(iy+1)];
			d = U[ix + N*(iy-1)];
			D[index] = ((l + r + t + d - 4.0 * c) / (h * h)) - F[index];
		}

		// Stride
		index = index + blockDim.x * gridDim.x;
	}

	__syncthreads();
}

void getResidual(int N, double L, double* U, double* F, double* D){
	double dx=L/(double) (N-1);
#	pragma omp parallel for	
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			if (i==0 or j==0 or j==N-1 or i==N-1)  *(D+i*N+j)=0.0;
			else  *(D+i*N+j) =  (1.0/pow(dx,2)) * (*(U+(i+1)*N+j) + *(U+(i-1)*N+j) + *(U+i*N+(j+1)) + *(U+i*N+(j-1)) - 4* *(U+i*N+j)) - *(F+i*N+j);
		}
	}
#	pragma omp barrier
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
	double L = 1.0;
	int N;
	float *d_D, *d_F, *d_U;		// device memory
	float *h_D, *h_F, *h_U;		// host memory

	double *F, *U, *D;			// Target of input and outpu

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

	D = (double*) malloc(N * N * sizeof(double));
	F = (double*) malloc(N * N * sizeof(double));
	U = (double*) malloc(N * N * sizeof(double));
	
	printf("~ Initialize the test LU - F = D ~\n");
	for(int i = 0; i < N*N; i = i+1){
		F[i] = rand() / (double)RAND_MAX;
		U[i] = rand() / (double)RAND_MAX;
	}

    cudaEvent_t start, stop;
    cudaError_t err;
    err = cudaSetDevice( 0 );
    if(err != cudaSuccess){
        printf("Cannot select GPU\n");
        exit(1);
    }

    float gpu_time_use;

    printf("~Run the test~");
    double h = L / (double)(N - 1);

    h_D = (float*) malloc(N * N * sizeof(float));
	h_F = (float*) malloc(N * N * sizeof(float));
	h_U = (float*) malloc(N * N * sizeof(float));

	#	pragma omp parallel for
	for(int i = 0; i < N*N; i = i+1){
		h_F[i] = (float) F[i];
		h_U[i] = (float) U[i];
	}
	
    cudaEventCreate(&start);
	cudaEventCreate(&stop);

	printf("~Start GPU code~\n");
	cudaEventRecord(start, 0);

	cudaMalloc((void**)&d_D, N * N * sizeof(float));
	cudaMalloc((void**)&d_F, N * N * sizeof(float));
	cudaMalloc((void**)&d_U, N * N * sizeof(float));

	cudaMemcpy(d_F, h_F, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_U, h_U, N * N * sizeof(float), cudaMemcpyHostToDevice);	

	Residual_GPU <<< blocksPerGrid, threadsPerBlock >>> (N, (float)h, d_U, d_F, d_D);
	
	cudaMemcpy(h_D, d_D, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&gpu_time_use, start, stop);

	cudaFree(d_D);
	cudaFree(d_F);
	cudaFree(d_U);


	#	pragma omp parallel for 
	for (int i = 0; i < N*N; i = i+1){
		D[i] = (double) h_D[i];
	}

	doPrint(N, D);
	
	printf("GPU BlockSize = %d, GridSize = %d, TimeUsed = %lf\n", threadsPerBlock, blocksPerGrid, gpu_time_use);
	printf("\n");

	double *D_CPU;
	float cpu_time_use;
	
	cudaEventRecord(start, 0);
	
	D_CPU = (double*) malloc(N * N * sizeof(double));
	
	getResidual(N, L, U, F, D_CPU);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&cpu_time_use, start, stop);
	
	doPrint(N, D_CPU);
	printf("CPU TimeUsed = %lf\n", cpu_time_use);	
	printf("SpeedUp = %lf\n", cpu_time_use / gpu_time_use);

	double error = 0.0;
	#	pragma omp parallel for
	for(int i = 0; i < N*N; i = i+1){
		error = error + abs(D[i] - D_CPU[i]);
	}

	printf("norm(D - D_CPU) = %lf\n", error / (double)(N*N));

	free(h_F);
	free(h_U);
	free(h_D);
	free(D);
	free(U);
	free(F);

	return 0;
}
