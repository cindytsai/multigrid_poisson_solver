/*
 nvcc -arch=compute_52 -code=sm_52 -O3 --compiler-options -fopenmp Test_getSource_GPU.cu
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

__global__ void ker_GaussSeideleven_GPU(int N, double h, double *U, double *F){
	// Settings
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int parity, ix, iy;
	int index;		// index of the point to be update
	double l, r, t, d;

	while( i < (N*N) / 2){
		// Parse the index of even chestbox
		ix = (2 * i) % N;
		iy = ((2 * i) / N) % N;
		parity = (ix + iy) % 2;
		ix = ix + parity;

		// Ignore the boundary
		if( (ix == 0) || (ix == N-1) || (iy == 0) || (iy == N-1) ){
			// Do nothing
		}
		else{
			index = ix + iy * N;
			l = U[(ix-1) + N*iy];
			r = U[(ix+1) + N*iy];
			t = U[ix + N*(iy+1)];
			d = U[ix + N*(iy-1)];

			U[index] = 0.25 * (l + r + t + d - h * h * F[index]);
		}
		
		// Stride
		index = index + blockDim.x * gridDim.x;
	}

	__syncthreads();

}

__global__ void ker_GaussSeidelodd_GPU(int N, double h, double *U, double *F){
	// Settings
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int parity, ix, iy;
	int index;		// index of the point to be update
	double l, r, t, d;

	while( i < (N*N) / 2){
		// Parse the index of even chestbox
		ix = (2 * i) % N;
		iy = ((2 * i) / N) % N;
		parity = (ix + iy + 1) % 2;
		ix = ix + parity;

		// Ignore the boundary
		if( (ix == 0) || (ix == N-1) || (iy == 0) || (iy == N-1) ){
			// Do nothing
		}
		else{
			index = ix + iy * N;
			l = U[(ix-1) + N*iy];
			r = U[(ix+1) + N*iy];
			t = U[ix + N*(iy+1)];
			d = U[ix + N*(iy-1)];

			U[index] = 0.25 * (l + r + t + d - h * h * F[index]);
		}
		
		// Stride
		index = index + blockDim.x * gridDim.x;
	}

	__syncthreads();

}

__global__ void ker_Error_GPU(int N, double h, double *U, double *F, double *err){
	// Settings for parallel reduction to calculate the error array 
	extern __shared__ double cache[];
	int ib = blockDim.x / 2;
	int cacheIndex = threadIdx.x;
	double diff = 0.0;

	// Settings for getting the residual
	double l, r, t, d, c;		// value of neighboring index and the center index
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int ix, iy;

	// Getting the residual
	while( index < N*N ){
		// Parse the index to ix, iy
		ix = index % N;
		iy = index / N;

		// Ignore the boundary
		if( (ix == 0) || (ix == N-1) || (iy == 0) || (iy == N-1) ){
			// Do nothing
		}
		else{
			l = U[(ix-1) + N*iy];
			r = U[(ix+1) + N*iy];
			t = U[ix + N*(iy+1)];
			d = U[ix + N*(iy-1)];
			c = U[index];

			diff = diff + fabs( ( (l + r + t + d - 4.0 * c) / (h * h) ) - F[index] );
		}

		// Stride
		index = index + blockDim.x * gridDim.x;
	}

	cache[cacheIndex] = diff;

	__syncthreads();

	// Calculate error array with parallel reduction
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

void GaussSeidel_GPU(int N, double L, double *U, double *F, double target_error){
	// Settings
	double h = L / (double) (N-1);
	int iter = 1;

	// Settings for GPU
	int max_m = 10;
	int max_n = 5;
	int min_n = 0;
	int m = max_m;
	int n = max_n;
	int alter = 1;
	int blocksPerGrid = pow(10, n);
	int threadsPerBlock = pow(2, m);
	long long int sharedMemorySize;
	double error = target_error + 1.0;
	double *d_U, *d_F, *d_err;		// device memory
	double             *h_err;		// host memory

	/*
	GPU Part
	 */
	// Check that blocksPerGrid * threadsPerBlock < ((N*N) / 2)
	//        and threadsPerBlock = 2^m
	while( blocksPerGrid * threadsPerBlock > (N*N) / 2 ){
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
	sharedMemorySize = threadsPerBlock * sizeof(double);

	// Allocate host memory for error array
	h_err = (double*) malloc(blocksPerGrid * sizeof(double));

	// Allocate device memory
	cudaMalloc((void**)&d_U , N * N * sizeof(double));
	cudaMalloc((void**)&d_F , N * N * sizeof(double));
	cudaMalloc((void**)&d_err, blocksPerGrid * sizeof(double));

	// Copy data to device memory
	cudaMemcpy(d_U, U, N * N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_F, F, N * N * sizeof(double), cudaMemcpyHostToDevice);

	// Do the iteration until it is smaller than the target_error
	while( error > target_error ){
		// Iteration Even / Odd index
		ker_GaussSeideleven_GPU <<< blocksPerGrid, threadsPerBlock >>> (N, h, d_U, d_F);
		ker_GaussSeidelodd_GPU <<< blocksPerGrid, threadsPerBlock >>> (N, h, d_U, d_F);

		// Get the error
		ker_Error_GPU <<< blocksPerGrid, threadsPerBlock, sharedMemorySize >>> (N, h, d_U, d_F, d_err);
		
		// DEBUG info
		printf("Done ker_Error_GPU\n");
		cudaError_t cuda_error;
		cuda_error = cudaMemcpy(h_err, d_err, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);

		if(cuda_error != cudaSuccess){
			printf("cudaMemcpy failed\n");
			exit(1);
		}
		else{
			printf("cudaMemcpy success\n");
		}

		// Calculate the error from error array
		error = 0.0;
		for(int i = 0; i < blocksPerGrid; i = i+1){
			error = error + h_err[i];
		}
		error = error / (double)(N*N);
		
		// DEBUG info
		printf("iter = %d, err = %lf\n", iter, error);
		fflush(stdout);

		iter = iter + 1;
	}

	// Copy data back to host memory
	cudaMemcpy(U, d_U, N * N * sizeof(double), cudaMemcpyDeviceToHost);

	// Free the device memory
	cudaFree(d_U);
	cudaFree(d_F);
	cudaFree(d_err);

	// Free the host memory
	free(h_err);
}

void doExactSolver_GPU(int N, double L, double *U, double *F, double target_error, int option){
	// Option can only be 1 for now, since there is only Gauss-Seidel Even/Odd Method for now.
	if( option == 0 ){
		printf("Using GPU, no Inverse Matrix Method in doExactSolver_GPU!\n");
		exit(1);
	}

	// Gauss-Seidel Even/Odd Method
	if( option == 1){
		GaussSeidel_GPU(N, L, U, F, target_error);
	}

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

void GaussSeidel(int N, double L, double *U, double *F, double target_error){
	
	double h = L / (double)(N - 1);
	double err = target_error + 1.0;
	int iter = 0;

	int index, ix, iy;	// Index of the point to be update
	int l, r, t, b;		// Index of the neighbers

	int *ieven, *iodd;	// Index of even / odd chestbox
	double *U_old;		// For storing U during iteration
	double *Residual; 	// Get the residual to compute the error

	/*
	Prepared and initialize
	 */
	ieven = (int*) malloc(((N * N) / 2) * sizeof(int));
	iodd  = (int*) malloc(((N * N) / 2) * sizeof(int));
	U_old = (double*) malloc(N * N * sizeof(double));
	Residual = (double*) malloc(N * N * sizeof(double));

	// For even chestbox index
	#	pragma omp parallel for
	for(int i = 0; i < ((N * N) / 2); i = i+1){
		int parity, ix, iy;
		ix = (2 * i) % N;
		iy = ((2 * i) / N) % N;
		parity = (ix + iy) % 2;
		ix = ix + parity;
		ieven[i] = ix + iy * N;
	}
	// For odd chestbox index
	#	pragma omp parallel for
	for(int i = 0; i < ((N * N) / 2); i = i+1){
		int parity, ix, iy;
		ix = (2 * i) % N;
		iy = ((2 * i) / N) % N;
		parity = (ix + iy + 1) % 2;
		ix = ix + parity;
		iodd[i] = ix + iy * N;
	}

	// Initialize
	memset(U_old, 0.0, N * N * sizeof(double));
	memset(U, 0.0, N * N * sizeof(double));

	// Start the Gauss-Seidel Iteration
	while( err > target_error ){
		
		// Update even chestbox
		#	pragma omp parallel
		{
			#	pragma omp for private(index, ix, iy, l, r, t, b)
			for(int i = 0; i < (N * N) / 2; i = i+1){
				// Center index
				index = ieven[i];
				ix = index % N;
				iy = index / N;

				// Do not update the boundary
				if((ix == 0) || (ix == (N-1)) || (iy == 0) || (iy == (N-1))){
					continue;
				}

				// Neighboring index
				l = (ix - 1) + iy * N;
				r = (ix + 1) + iy * N;
				t = ix + (iy + 1) * N;
				b = ix + (iy - 1) * N;

				// Update result to U
				U[index] = 0.25 * (U_old[l] + U_old[r] + U_old[t] + U_old[b] - pow(h, 2) * F[index]);
			}

			// Update odd chestbox
			#	pragma omp for private(index, ix, iy, l, r, t, b)
			for(int i = 0; i < (N * N) / 2; i = i+1){
				// Center index
				index = iodd[i];
				ix = index % N;
				iy = index / N;

				// Do not update the boundary
				if((ix == 0) || (ix == (N-1)) || (iy == 0) || (iy == (N-1))){
					continue;
				}

				// Neighboring index
				l = (ix - 1) + iy * N;
				r = (ix + 1) + iy * N;
				t = ix + (iy + 1) * N;
				b = ix + (iy - 1) * N;

				// Update result to U
				U[index] = 0.25 * (U[l] + U[r] + U[t] + U[b] - pow(h, 2) * F[index]);			
			}			
		}


		// Compute the error, without the boundary, since it is always "0"
		iter = iter + 1;
		err = 0.0;
		getResidual(N, L, U, F, Residual);

		#	pragma omp parallel for reduction( +:err )
		for(int j = 1; j < (N-1); j = j+1){
			for(int i = 1; i < (N-1); i = i+1){
				err = err + fabs(Residual[i+N*j]);
			}
		}
		err = err / (double)((N - 2) * (N - 2));

		// Move U to U_old and start the next iteration
		memcpy(U_old, U, sizeof(double) * N * N);
	}

	// Free the temperary resource
	free(ieven);
	free(iodd);
	free(U_old);
	free(Residual);
}

void doExactSolver(int N, double L, double *U, double *F, double target_error, int option){

	// Gauss-Seidel even / odd method
	if(option == 1){
		GaussSeidel(N, L, U, F, target_error);
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
	double L = 1.0;
	int N;

	double *F, *U;  // Target of input and output
	double *F_CPU, *U_CPU;

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
	
	printf("~ Initialize the test doExactSolver_GPU ~\n");
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

    doExactSolver_GPU(N, L, U, F, 1e-3, 1);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&gpu_time_use, start, stop);

	doPrint(N, U);
	
	printf("GPU: TimeUsed = %lf\n", gpu_time_use);
	printf("\n");

	float cpu_time_use;
	
	cudaEventRecord(start, 0);
	
	doExactSolver(N, L, U_CPU, F_CPU, 1e-3, 1);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&cpu_time_use, start, stop);
	
	doPrint(N, U_CPU);
	printf("CPU: TimeUsed = %lf\n", cpu_time_use);	
	printf("SpeedUp = %lf\n", cpu_time_use / gpu_time_use);

	double diff = 0.0;
	#	pragma omp parallel for
	for(int i = 0; i < N*N; i = i+1){
		diff = diff + abs(U[i] - U_CPU[i]);
	}

	printf("norm(U - U_CPU) = %lf\n", diff / (double)(N*N));

	return 0;
}
