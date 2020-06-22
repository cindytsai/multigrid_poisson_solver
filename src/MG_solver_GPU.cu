#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

/*
GPU kernel
 */
__global__ void ker_Source_GPU(int, float, float*, float, float);
__global__ void ker_Residual_GPU(int, float, float*, float*, float*);
__global__ void ker_Smoothing_GPU(int, float, float*, float*, float*, int, float*);
__global__ void ker_GaussSeideleven_GPU(int, double, double*, double*);
__global__ void ker_GaussSeidelodd_GPU(int, double, double*, double*);
__global__ void ker_Error_GPU(int, double, double*, double*, double*);
__global__ void ker_Zoom_GPU(int, float*, int, float*);

/*
Wrap the GPU kernel as CPU function
 */
// Main Function
void getSource_GPU(int, double, double*, double, double);
void getResidual_GPU(int, double, double*, double*, double*);
void doSmoothing_GPU(int, double, double*, double*, int, double*);
void doExactSolver_GPU(int, double, double*, double*, double, int);
void doRestriction_GPU(int, double*, int, double*);
void doProlongation_GPU(int, double*, int, double*);
void doPrint(int, double*);
void doPrint2File(int, double*, char*);

// Sub Routine
void GaussSeidel_GPU(int, double, double*, double*, double);


int main(){
	// Settings for GPU
	cudaEvent_t start, stop;
	cudaError_t err = cudaSuccess;
	err = cudaSetDevice( 0 );
	if(err != cudaSuccess){
		printf("Cannot select GPU\n");
		exit(1);
	}

	// Settings for OpenMP
	omp_set_num_threads( 16 );

	/*
	Multigrid Poisson Solver
	 */
	
	// Reset the device
	cudaDeviceReset( 0 );

	return 0;
}

/*
GPU Kernel
 */
__global__ void ker_Source_GPU(int N, float h, float *F, float min_x, float min_y){
	// Thread index inside the GPU kernel
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int ix, iy;
	float x, y;

	while( index < N * N ){
		// Parse the index to ix, iy
		ix = index % N;
		iy = index / N;
		if( (ix == 0) || (ix == N-1) || (iy == 0) || (iy == N-1)){
            // Problem Dependent
            F[index] = 0.0;
        }
        else{
            // Calculate the coordinate in (x, y)
		    x = (float) ix * h + min_x;
		    y = (float) iy * h + min_y;
		    // Problem Dependent
		    F[index] = 2.0 * x * (y - 1) * (y - 2.0 * x + x * y + 2.0) * expf(x - y);
        }
		
        // Stride
		index = index + blockDim.x * gridDim.x;
	}
	__syncthreads();
}

__global__ void Boundary_GPU(int N, float L, float *F, float min_x, float min_y){

}

__global__ void ker_Residual_GPU(int N, float h, float *U, float *F, float *D){
	// Thread index inside the GPU kernel
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int ix, iy;
	float l, r, t, d, c;	// value of the neighboring index and the center index of U

	while( index < N * N ){
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
			if( iter % 2 == 1 ){
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

__global__ void ker_GaussSeideleven_GPU(int N, float h, float *U, float *F){
	// Settings
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int parity, ix, iy;
	int index;		// index of the point to be update
	float l, r, t, d;

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
		i = i + blockDim.x * gridDim.x;
	}
}

__global__ void ker_GaussSeidelodd_GPU(int N, float h, float *U, float *F){
	// Settings
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int parity, ix, iy;
	int index;		// index of the point to be update
	float l, r, t, d;

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
		i = i + blockDim.x * gridDim.x;
	}
}

__global__ void ker_Error_GPU(int N, float h, float *U, float *F, float *err){
	// Settings for parallel reduction to calculate the error array 
	extern __shared__ float cache[];
	int ib = blockDim.x / 2;
	int cacheIndex = threadIdx.x;
	float diff = 0.0;

	// Settings for getting the residual
	float l, r, t, d, c;		// value of neighboring index and the center index
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

			diff = diff + fabsf( ( (l + r + t + d - 4.0 * c) / (h * h) ) - F[index] );
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

__global__ void ker_Zoom_GPU(int N, float *U_n, int M, float *U_m){
	
}

void getSource_GPU(int N, double L, double *F, double min_x, double min_y){
	double h = L / (double)(N - 1);

	// Settings
	int blocksPerGrid = 10;
	int threadsPerBlock = 10;
	float *d_F;		// device memory
	float *h_F;		// host memory

	/*
	CPU Part
	 */
	h_F = (float*) malloc(N * N * sizeof(float));

	/*
	GPU Part
	 */
	cudaMalloc((void**)&d_F, N * N * sizeof(float));

	ker_Source_GPU <<< blocksPerGrid, threadsPerBlock >>> (N, (float)h, d_F, (float)min_x, (float)min_y);

	cudaMemcpy(h_F, d_F, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_F);

	// Transform back to double
	#	pragma omp parallel for
	for(int i = 0; i < N*N; i = i+1){
		F[i] = (double) h_F[i];
	}

	free(h_F);
}

void getResidual_GPU(int N, double L, double *U, double *F, double *D){
	double h = L / (double)(N - 1);

	// Settings
	int blocksPerGrid = 10;
	int threadsPerBlock = 10;
	float *d_D, *d_F, *d_U;		// device memory
	float *h_D, *h_F, *h_U;		// host memory

	/*
	CPU Part
	 */
	// Allocate host memory
	h_D = (float*) malloc(N * N * sizeof(float));
	h_F = (float*) malloc(N * N * sizeof(float));
	h_U = (float*) malloc(N * N * sizeof(float));

	// Change data from double to float
	#	pragma omp parallel for
	for(int i = 0; i < N*N; i = i+1){
		h_F[i] = (float) F[i];
		h_U[i] = (float) U[i];
	}

	/*
	GPU Part
	 */
	cudaMalloc((void**)&d_D, N * N * sizeof(float));
	cudaMalloc((void**)&d_F, N * N * sizeof(float));
	cudaMalloc((void**)&d_U, N * N * sizeof(float));

	cudaMemcpy(d_F, h_F, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_U, h_U, N * N * sizeof(float), cudaMemcpyHostToDevice);

	ker_Residual_GPU <<< blocksPerGrid, threadsPerBlock >>> (N, (float)h, d_U, d_F, d_D);
	
	cudaMemcpy(h_D, d_D, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_D);
	cudaFree(d_F);
	cudaFree(d_U);

	// Change data from float to double
	#	pragma omp parallel for 
	for (int i = 0; i < N*N; i = i+1){
		D[i] = (double) h_D[i];
	}

	free(h_F);
	free(h_U);
	free(h_D);
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

void doPrint(int N, double *U){
	for(int j = N-1; j >= 0; j = j-1){
		for(int i = 0; i < N; i = i+1){
			printf("%2.3e ", U[i+N*j]);
		}
		printf("\n");
	}
}

void doPrint2File(int N, double *U, char *file_name){
	// Create file
	FILE *output;
	output = fopen(file_name, "w");

	// Print result to CSV form
	for(int j = N-1; j >= 0; j = j-1){
		for(int i = 0; i < N; i = i+1){
			if(i == N-1){
				fprintf(output, "%lf\n", U[i+N*j]);
			}
			else{
				fprintf(output, "%lf,", U[i+N*j]);
			}
		}
	}

	// Close file
	fclose(output);
}

void GaussSeidel_GPU(int N, double L, double *U, double *F, double target_error){
	// Settings
	double h = L / (double) (N-1);

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
	float error = (float) target_error + 1.0;
	float *d_U, *d_F, *d_err;		// device memory
	float *h_U, *h_F, *h_err;		// host memory

	/*
	CPU Part
	 */
	// Allocate the memory
	h_U = (float*) malloc(N * N * sizeof(float));
	h_F = (float*) malloc(N * N * sizeof(float));
	
	// Transfer the data from double to float
	#	pragma omp parallel for
	for(int i = 0; i < N*N; i = i+1){
		h_F[i] = (float) F[i];
	}

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
	sharedMemorySize = threadsPerBlock * sizeof(float);

	// Allocate host memory for error array
	h_err = (float*) malloc(blocksPerGrid * sizeof(float));

	// Allocate device memory
	cudaMalloc((void**)&d_U , N * N * sizeof(float));
	cudaMalloc((void**)&d_F , N * N * sizeof(float));
	cudaMalloc((void**)&d_err, blocksPerGrid * sizeof(float));

	// Copy data to device memory
	cudaMemset(d_U, 0.0, N * N * sizeof(float));
	cudaMemcpy(d_F, h_F, N * N * sizeof(float), cudaMemcpyHostToDevice);

	free(h_F); 		// h_F is no longer needed.
	
	// Do the iteration until it is smaller than the target_error
	while( error > target_error ){
		// Iteration Even / Odd index
		ker_GaussSeideleven_GPU <<< blocksPerGrid, threadsPerBlock >>> (N, (float) h, d_U, d_F);
		ker_GaussSeidelodd_GPU <<< blocksPerGrid, threadsPerBlock >>> (N, (float) h, d_U, d_F);
		
		// Get the error
		ker_Error_GPU <<< blocksPerGrid, threadsPerBlock, sharedMemorySize >>> (N, (float) h, d_U, d_F, d_err);
		cudaMemcpy(h_err, d_err, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

		// Calculate the error from error array
		error = 0.0;
		for(int i = 0; i < blocksPerGrid; i = i+1){
			error = error + h_err[i];
		}
		error = error / (float)(N*N);
	}

	// Copy data back to host memory
	cudaMemcpy(h_U, d_U, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	// Free the device memory
	cudaFree(d_U);
	cudaFree(d_F);
	cudaFree(d_err);

	// Transfer data from float back to double
	#	pragma omp parallel for
	for(int i = 0; i < N*N; i = i+1){
		U[i] = (double) h_U[i];
	}

	// Free the host memory
	free(h_err);
	free(h_U);
}
