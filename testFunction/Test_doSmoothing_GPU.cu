/*
 nvcc -arch=compute_52 -code=sm_52 -O3 --compiler-options -fopenmp Test_getSource_GPU.cu
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

__global__ void ker_Smoothing_GPU(int N, float h, float *U, float *U0, float *F, int iter, float *err){
	
	// Settings for parallel reduction to calculate the error array 
	extern __shared__ float cache[];
	int ib = blockDim.x / 2;
	int cacheIndex = threadIdx.x;
	float diff =  0.0;

	// Settings for smoothing
	float l, r, t, d;	// value of neighboring index
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int ix, iy;
	
	/*
	Smoothing
	 */
	while( index < N*N ){
		// Parse the index to ix, iy
		ix = index % N;
		iy = index / N;
		
		// Ignore the boundary
		if( (ix == 0) || (ix == N-1) || (iy == 0) || (iy == N-1) ){
			// Do nothing
		}
		else{
			if( iter % 2 == 1){
				// Update in U
				// U0 -> old, U -> new
				l = U0[(ix-1) + N*iy];
				r = U0[(ix+1) + N*iy];
				t = U0[ix + N*(iy+1)];
				d = U0[ix + N*(iy-1)];

				U[index] = 0.25 * (l + r + t + d - h * h * F[index]);
			}
			else{
				// Update in U0
				// U -> old, U0 -> new
				l = U[(ix-1) + N*iy];
				r = U[(ix+1) + N*iy];
				t = U[ix + N*(iy+1)];
				d = U[ix + N*(iy-1)];

				U0[index] = 0.25 * (l + r + t + d - h * h * F[index]);
			}
			diff = diff + fabsf( (U[index] - U0[index]) * 4.0 / (h * h) );
		}

		// Strid
		index = index + blockDim.x * gridDim.x;
	}

	cache[cacheIndex] = diff;

	__syncthreads();

	/*
	Calculate error array with parallel reduction
	 */
	while( ib != 0 ){
		
		if(cacheIndex < ib){
			cache[cacheIndex] = cache[cacheIndex] + cache[cacheIndex + ib];
		}

		__syncthreads();

		ib = ib / 2;
	}

	if(cacheIndex == 0){
		err[blockIdx.x] = cache[0];
	}
}

void doSmoothing(int N, double L, double* U, double* F, int step, double* error){
	double dx=L/(double) (N-1);
	double *U_old;
	U_old = (double *) malloc(N*N*sizeof(double));
	//Gauss-Seidel
	for(int s=0; s<step; s++){

		
		for(int i=0; i<N; i++){
			for(int j=0; j<N; j++){
				U_old[i*N+j]=U[i*N+j];
			}
		}

#     	pragma omp parallel for		
		for(int i=1; i<N-1; i++){
			for(int j= i%2==0 ? 2:1; j<N-1; j+=2){
				*(U+i*N+j) +=  0.25*( *(U_old+(i+1)*N+j) + *(U_old+(i-1)*N+j) + *(U_old+i*N+(j+1)) + *(U_old+i*N+(j-1)) - 4* *(U_old+i*N+j) - pow(dx,2) * *(F+i*N+j));
			}
		}
#		pragma omp barrier
#     	pragma omp parallel for	
		for(int i=1; i<N-1; i++){
			for(int j= i%2==0 ? 1:2; j<N-1; j+=2){
				*(U+i*N+j) +=  0.25*( *(U_old+(i+1)*N+j) + *(U_old+(i-1)*N+j) + *(U_old+i*N+(j+1)) + *(U_old+i*N+(j-1)) - 4* *(U_old+i*N+j) - pow(dx,2) * *(F+i*N+j));
			}
		}
#		pragma omp barrier


	}
	free(U_old);


    double sum1=0.0;
#   pragma omp parallel for reduction( +:sum1 )
	for(int i=1; i<N-1; i++){
		for(int j= i%2==0 ? 2:1; j<N-1; j+=2){
			sum1 += fabs((1.0/pow(dx,2))*( *(U+(i+1)*N+j) + *(U+(i-1)*N+j) + *(U+i*N+(j+1)) + *(U+i*N+(j-1)) - 4* *(U+i*N+j)) - *(F+i*N+j));
		}
	}
    double sum2=0.0;
#   pragma omp parallel for reduction( +:sum2 )
	for(int i=1; i<N-1; i++){
		for(int j= i%2==0 ? 2:1; j<N-1; j+=2){
			sum2 += fabs((1.0/pow(dx,2))*( *(U+(i+1)*N+j) + *(U+(i-1)*N+j) + *(U+i*N+(j+1)) + *(U+i*N+(j-1)) - 4* *(U+i*N+j)) - *(F+i*N+j));
		}
	}
	*error = sum1+sum2;
	*error = *error/N/N;

	
}


void doPrint(int N, double *U){
	for(int j = N-1; j >= 0; j = j-1){
		for(int i = 0; i < N; i = i+1){
			printf("%2.3e ", U[i+N*j]);
		}
		printf("\n");
	}
}

void doSmoothing_GPU(int N, double L, double *U, double *F, int step, double *error){
	// Settings
	double h = L / (double)(N - 1);
	int iter = 1;

	// Settings GPU
	// blocksPerGrid * threadsPerBlock < N*N
	// And since I'm using parallel reduction, threadsPerBLock must be 2^m
	int max_m = 10;
	int max_n = 5;
	int min_n = 0;
	int m = max_m;
	int n = max_n;
	int alter = 1;
	int blocksPerGrid = pow(10, n);
	int threadsPerBlock = pow(2, m);
	int sharedMemorySize;
	float error_f;
	float *d_U, *d_U0, *d_F, *d_err;		// device memory
	float *h_U,        *h_F, *h_err;		// host memory

	/*
	CPU Part
	 */
	// Allocate host memory
	h_F  = (float*) malloc(N * N * sizeof(float));
	h_U  = (float*) malloc(N * N * sizeof(float));
	
	// Change data from double to float
	#	pragma omp parallel for
	for(int i = 0; i < N*N; i = i+1){
		h_F[i] = (float) F[i];
		h_U[i] = (float) U[i];
	}

	/*
	GPU Part
	 */
	// Check that blocksPerGrid * threadsPerBlock < N*N
	//        and threadsPerBlock = 2^m
	while( blocksPerGrid * threadsPerBlock > N*N ){
		// Decrease the exponential part
		if( (alter % 4 != 0) || (n == min_n) ){
			m = m - 1;
			alter = alter + 1;
		}
		if( (alter % 4 == 0) && (n > min_n) ){
			n = n - 1;
			m = max_m;
			alter = alter + 1;
		}
		blocksPerGrid = pow(10, n);
		threadsPerBlock = pow(2, m);
	}
	sharedMemorySize = threadsPerBlock * sizeof(float);
	h_err = (float*) malloc(blocksPerGrid * sizeof(float));

	// Allocate device memory
	cudaMalloc((void**)&d_U , N * N * sizeof(float));
	cudaMalloc((void**)&d_U0, N * N * sizeof(float));
	cudaMalloc((void**)&d_F , N * N * sizeof(float));
	cudaMalloc((void**)&d_err, blocksPerGrid * sizeof(float));

	// Copy data to device memory
	cudaMemcpy( d_U,  h_U, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy( d_F,  h_F, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_U0,  d_U, N * N * sizeof(float), cudaMemcpyDeviceToDevice);

	free(h_F);    // h_F are no longer needed

	// Do the iteration with "step" steps
	while( iter <= step ){
		
		ker_Smoothing_GPU <<< blocksPerGrid, threadsPerBlock, sharedMemorySize >>> (N, (float) h, d_U, d_U0, d_F, iter, d_err);
		
		iter = iter + 1;

	}
	
	// Copy data back to host memory
	// d_err -> h_err
	// d_U   -> h_U   if step is odd
	// d_U0  -> h_U   if step is even
	cudaMemcpy(h_err, d_err, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
	if( step % 2 == 1){
		cudaMemcpy(h_U, d_U, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	}
	else{
		cudaMemcpy(h_U, d_U0, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	}

	// Free the device memory
	cudaFree(d_U);
	cudaFree(d_U0);
	cudaFree(d_F);
	cudaFree(d_err);

	/*
	Subsequent processing
	 */
	// Calculate the error from error array
	error_f = 0.0;
	for(int i = 0; i < blocksPerGrid; i = i+1){
		error_f = error_f + h_err[i];
	}
	error_f = error_f / (float)(N*N);
	
	*error = (double) error_f;

	// Transform data from float to double
	#	pragma omp parallel for 
	for (int i = 0; i < N*N; i = i+1){
		U[i] = (double) h_U[i];
	}

	free(h_err);
	free(h_U);
}


int main( int argc, char *argv[] ){
	double L = 1.0;
	int N;

	double *F, *U;  // Target of input and output
	double *F_CPU, *U_CPU;
	double error, error_CPU;

	if(argc != 2){
		N = 16;
	}
	else{
		N = atoi(argv[1]);
	}

	F = (double*) malloc(N * N * sizeof(double));
	U = (double*) malloc(N * N * sizeof(double));
	F_CPU = (double*) malloc(N * N * sizeof(double));
	U_CPU = (double*) malloc(N * N * sizeof(double));
	
	printf("~ Initialize the test doSmoothing_GPU ~\n");
	for(int i = 0; i < N*N; i = i+1){
		F[i] = rand() / (double)RAND_MAX;
		F_CPU[i] = F[i];
		U[i] = rand() / (double)RAND_MAX;
		U_CPU[i] = U[i];
	}

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

    doSmoothing_GPU(N, L, U, F, 10, &error);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&gpu_time_use, start, stop);

	doPrint(N, U);
	
	printf("GPU: TimeUsed = %lf, grid error = %lf\n", gpu_time_use, error);
	printf("\n");

	float cpu_time_use;
	
	cudaEventRecord(start, 0);
	
	doSmoothing(N, L, U_CPU, F_CPU, 10, &error_CPU);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&cpu_time_use, start, stop);
	
	doPrint(N, U_CPU);
	printf("CPU: TimeUsed = %lf, grid error = %lf\n", cpu_time_use, error_CPU);	
	printf("SpeedUp = %lf\n", cpu_time_use / gpu_time_use);

	double diff = 0.0;
	#	pragma omp parallel for
	for(int i = 0; i < N*N; i = i+1){
		diff = diff + abs(U[i] - U_CPU[i]);
	}

	printf("norm(U - U_CPU) = %lf\n", diff / (double)(N*N));

	return 0;
}
