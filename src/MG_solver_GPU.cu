#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include <string.h>
#include <fstream>
#include <cuda_runtime.h>
#include "linkedlist.h"

using namespace std;

/*
GPU kernel
 */
__global__ void ker_Source_GPU(int, float, float*, float, float);
__global__ void ker_Residual_GPU(int, float, float*);
__global__ void ker_GridAddition_GPU(int, float*);
__global__ void ker_Smoothing_GPU(int, float, float*, float*, int, float*);
__global__ void ker_GaussSeideleven_GPU_Double(int, double, double*, double*);
__global__ void ker_GaussSeideleven_GPU_Single(int, float, float*, float*);
__global__ void ker_GaussSeidelodd_GPU_Double(int, double, double*, double*);
__global__ void ker_GaussSeidelodd_GPU_Single(int, float, float*, float*);
__global__ void ker_Error_GPU_Double(int, double, double*, double*, double*);
__global__ void ker_Error_GPU_Single(int, float, float*, float*, float*);
__global__ void ker_Zoom_GPU(int, float, int, float, float*);

/*
Wrap the GPU kernel as CPU function
 */
// Main Function
void getSource_GPU(int, double, double*, double, double);
void getResidual_GPU(int, double, double*, double*, double*);
void doGridAddition_GPU(int, double*, double*);
void doSmoothing_GPU(int, double, double*, double*, int, double*);
void doExactSolver_GPU(int, double, double*, double*, double, int);
void doRestriction_GPU(int, double*, int, double*);
void doProlongation_GPU(int, double*, int, double*);
void doPrint(int, double*);
void doPrint2File(int, double*, char*);

// Sub Routine
void GaussSeidel_GPU_Double(int, double, double*, double*, double);
void GaussSeidel_GPU_Single(int, double, double*, double*, double);

/*
Global Variable
 */
// declare the texture memory for GPU functions
__align__(8) texture<float> texMem_float1;
__align__(8) texture<float> texMem_float2;

int main( int argc, char *argv[] ){
	// Settings for GPU
	cudaEvent_t start, stop;
	cudaError_t err = cudaSuccess;
	err = cudaSetDevice( 0 );
	if(err != cudaSuccess){
		printf("Cannot select GPU\n");
		exit(1);
	}

	/*
	
	Settings of testing different cycle and different numbers of OpenMP threads
	
		$ ./MG_CPU (N_THREADS_OMP) (cycle_filename.txt)

	     N_THREADS_OMP:     Number of OpenMP threads
	cycle_filename.txt:     The cycle structure
	
	 */
	int N_THREADS_OMP;		// OpenMP threads
	ifstream f_read;		// Read cycle structure file
	char file_name[50];

	if( argc != 3){
		printf("[ ERROR ]: Wrong input numbers of parameter.\n");
		exit(1);
	}

	// Set OpenMP thread
	N_THREADS_OMP = atoi(argv[1]);
	omp_set_num_threads( N_THREADS_OMP );
	printf("OpenMP threads = %d\n", N_THREADS_OMP);

	// Read cycle structure file
	f_read.open(argv[2]);
	printf("Cycle structure file name = %s\n", argv[2]);
	
	if( f_read.is_open() != true ){
		printf("[ ERROR ]: Cannot open file %s\n", argv[2]);
		exit(1);
	}

	// Create clock
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/*
	Start to read the cycle structure
	 */
	double L;				// Interest region length L
	double min_x, min_y;	// Lower left point of the interest region

	int con_step;			// Options to control step
	int con_N;				// Options to control N

	int N_max;				// Initial (Maximum) grid size
	int N_min;				// Coarsest (Minimum) grid size
	int *N_array;			// Store the auto generate N 
	int len = 0;			// Length of the N_array
	int len_flag = 0;		// Record location in N_array for the current level
	LinkedList cycle;		// Store variable in each level
	int node;				// Operation {-1, 0, 1}
	int step;				// Step of smoothing
	int next_N;				// Next grid size N
	double exactSolverTargetError;	// Target error for the exact solver
	int exactSolverOption;			// Options for the exact solver
	
	int N;					// Current grid size N
	double *U, *F, *D;		// U, F, D at that level
	double *tempU;			// Temperary U after prolongation
	double *ptrError;		// Error after smoothing step

	float time_used;		// Time used by Multigrid Method

	// Problem interest region
	f_read >> L >> min_x >> min_y;

	// Ways we want to control smoothing step and grid size N
	f_read >> con_step >> con_N;

	// Input initial (maximum) N_max and coarsest grid N_min
	f_read >> N_max >> N_min;

	if( con_N == 1 ){
		/*
		N = N / 2 on next level
		 */
		// Find length of the N_array		
		N = N_max;
		while( N >= N_min ){
			len = len + 1;
			N = N / 2;
		}
		
		// Allocate N_array memory
		N_array = (int*) malloc(len * sizeof(int));

		// Assign generated N
		N = N_max;
		for(int i = 0; i < len; i = i+1){
			N_array[i] = N;
			N = N / 2;
		}
	}
	if( con_N == 2 ){
		/*
		N = N - 1 on next level
		 */
		// Find len and allocate memory
		len = N_max - N_min + 1;
		N_array = (int*) malloc(len * sizeof(int));

		// Assign generated N
		N = N_max;
		for(int i = 0; i < len; i = i+1){
			N_array[i] = N;
			N = N - 1;
		}
	}

	// Initialize the cycle
	cycle.Push_back(N_max);
	cycle.Set_Problem(L, min_x, min_y);	// it's useless here, but yah, anyway =-=
	F = cycle.Get_F();
	N = cycle.Get_N();
	getSource_GPU(N, L, F, min_x, min_y);

	// Start the clock
	cudaEventRecord(start, 0);

	while( f_read.eof() != true ){
		
		f_read >> node;

		if( node == 2 ){
			break;
		}
		
		/*
		Do smoothing then do restriction
		 */
		if( node == -1 ){
			// Get smoothing step, next grid size N
			if( con_step == 0 && con_N == 0 ){
				f_read >> step >> next_N;
			}
			if( con_step == 0 && con_N != 0 ){
				f_read >> step;
				next_N = N_array[len_flag + 1];

				len_flag = len_flag + 1;
			}
			if( con_step != 0 && con_N == 0 ){
				f_read >> next_N;
				step = con_step;
			}
			if( con_step != 0 && con_N != 0 ){
				next_N = N_array[len_flag + 1];
				step = con_step;

				len_flag = len_flag + 1;
			}

			/*
			Smoothing and get the residual
			 */
			if( step == -1 ){
				// Use error trigger
				// TODO
			}
			else if( step == 0 ){
				// Do nothing, skip smoothing
			}
			else{
				// Smoothing
				N = cycle.Get_N();
				U = cycle.Get_U();
				F = cycle.Get_F();
				ptrError = cycle.Get_ptr_smoothingError();
				memset(U, 0.0, N * N * sizeof(double));
				doSmoothing_GPU(N, L, U, F, step, ptrError);

				printf("          ~Smoothing~\n");
				printf("Current Grid Size N = %d\n", N);
				printf("    Smoothing Steps = %d\n", step);
				printf("              Error = %lf\n", *ptrError);

				// Get the residual
				D = cycle.Get_D();
				getResidual_GPU(N, L, U, F, D);
			}

			/*
			Restriction
			 */
			// Do restriction on minus residual
			if( step != 0 ){
				// Flip the sign of D
				#	pragma omp parallel for
				for(int i = 0; i < N*N; i = i+1){
					D[i] = -D[i];
				}

				// Create ListNodes for next level
				cycle.Push_back(next_N);

				// Do restriction
				// And store at next level source term F
				doRestriction_GPU(N, D, next_N, cycle.Get_F());

				printf("             *\n");
				printf("             |\n");
				printf(" Restriction |\n");
				printf("             |\n");
				printf("             *\n");

			}
			else{
				// Full Multigrid Method
				// TODO
			}

		}
		/*
		Do the exact solver
		 */
		else if( node == 0 ){

			f_read >> exactSolverTargetError >> exactSolverOption;
			
			N = cycle.Get_N();
			U = cycle.Get_U();
			F = cycle.Get_F();

			doExactSolver_GPU(N, L, U, F, exactSolverTargetError, exactSolverOption);

			printf("          ~Exact Solver~\n");
			printf("Current Grid Size N = %d\n", N);
			if(exactSolverOption == 1){
				printf("   Use Exact Solver = GaussSeidel Even / Odd \n");
				printf("                      with double precision\n");
			}
			if(exactSolverOption == 2){
				printf("   Use Exact Solver = GaussSeidel Even / Odd \n");
				printf("                      with single precision\n");
			}
			printf("       Target Error = %.3e\n", exactSolverTargetError);

		}
		/*
		Do prolongation and then do smoothing
		 */
		else if( node == 1 ){
			// Get the smoothing step
			if( con_step == 0 && con_N == 0 ){
				f_read >> step;
			}
			if( con_step == 0 && con_N != 0 ){
				f_read >> step;
				len_flag = len_flag - 1;
			}
			if( con_step != 0 && con_N == 0 ){
				step = con_step;
			}
			if( con_step != 0 && con_N != 0 ){
				step = con_step;
				len_flag = len_flag - 1;
			}

			/*
			Do prolongation
			 */
			// Get the grid size next_N at the previous ListNode
			next_N = cycle.Get_prev_N();
			N = cycle.Get_N();
			U = cycle.Get_U();
			tempU = (double*) malloc(next_N * next_N * sizeof(double));
			doProlongation_GPU(N, U, next_N, tempU);
			
			printf("             *\n");
			printf("             |\n");
			printf("Prolongation |\n");
			printf("             |\n");
			printf("             *\n");

			// Remove the lastNode of cycle
			cycle.Remove_back();

			// Add tempU to U
			N = cycle.Get_N();
			U = cycle.Get_U();
			doGridAddition_GPU(N, U, tempU);

			// Free tempU, it is no longer needed.
			free(tempU);

			/*
			Do smoothing
			 */
			if( step == -1 ){
				// Use error trigger
				// TODO
			}
			else if( step == 0 ){
				// Do nothing , skip smoothing
			}
			else{
				// Smoothing
				F = cycle.Get_F();
				ptrError = cycle.Get_ptr_smoothingError();
				doSmoothing_GPU(N, L, U, F, step, ptrError);				

				printf("          ~Smoothing~\n");
				printf("Current Grid Size N = %d\n", N);
				printf("    Smoothing Steps = %d\n", step);
				printf("              Error = %lf\n", *ptrError);
			}
			
		}

	}

	// Stop the clock, and calculate the time used
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_used, start, stop);

	// Calculate the error of Multigrid Method, 
	// using getSource as source term F
	N = cycle.Get_N();
	U = cycle.Get_U();
	D = cycle.Get_D();
	F = cycle.Get_F();
	getSource_GPU(N, L, F, min_x, min_y);
	getResidual_GPU(N, L, U, F, D);

	double MGerror;
	for(int i = 0; i < N*N; i = i+1){
		MGerror = MGerror + fabs(D[i]);
	}
	MGerror = MGerror / (double)(N*N);

	// Print out final result
	printf("\n\n");
	printf("===== Final Result =====\n");
	printf("    Error = %lf\n", MGerror);
	printf("Time Used = %lf (ms)\n", time_used);

	// Setting output file name
	strcpy(file_name, "");
	strcat(file_name, "Sol_GPU_");
	strcat(file_name, argv[2]);
	doPrint2File(N, U, file_name);

	printf("Output file name = %s\n", file_name);
	// Reset the device
	cudaDeviceReset();

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

__global__ void ker_Residual_GPU(int N, float h, float *D){
	// Texture Memory
	// texMem_float1 -> U
	// texMem_float2 -> F

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
			c = tex1Dfetch(texMem_float1, index);
			l = tex1Dfetch(texMem_float1, (ix-1) + N*iy);
			r = tex1Dfetch(texMem_float1, (ix+1) + N*iy);
			t = tex1Dfetch(texMem_float1, ix + N*(iy+1));
			d = tex1Dfetch(texMem_float1, ix + N*(iy-1));

			D[index] = ((l + r + t + d - 4.0 * c) / (h * h)) - tex1Dfetch(texMem_float2, index);
		}

		// Stride
		index = index + blockDim.x * gridDim.x;
	}

	__syncthreads();
}

__global__ void ker_GridAddition_GPU(int N, float *U1){
	// Texture memory
	// texMem_float1 -> U2
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	while( index < N*N ){
		U1[index] = U1[index] + tex1Dfetch(texMem_float1, index);

		// Stride
		index = index + blockDim.x * gridDim.x;
	}

}

__global__ void ker_Smoothing_GPU(int N, float h, float *U, float *U0, int iter, float *err){
	// Texture memory
	// texMem_float2 -> F

	// Settings for parallel reduction to calculate the error array 
	extern __shared__ __align__(sizeof(float)) unsigned char cache_smoothing[];
	float *cache = reinterpret_cast<float *>(cache_smoothing);
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

				U[index] = 0.25 * (l + r + t + d - h * h * tex1Dfetch(texMem_float2, index));
			}
			else{
				// Update in U0
				// U -> old, U0 -> new
				l = U[(ix-1) + N*iy];
				r = U[(ix+1) + N*iy];
				t = U[ix + N*(iy+1)];
				d = U[ix + N*(iy-1)];

				U0[index] = 0.25 * (l + r + t + d - h * h * tex1Dfetch(texMem_float2, index));
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

__global__ void ker_GaussSeideleven_GPU_Double(int N, double h, double *U, double *F){
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
		i = i + blockDim.x * gridDim.x;
	}

}

__global__ void ker_GaussSeideleven_GPU_Single(int N, float h, float *U, float *F){
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

__global__ void ker_GaussSeidelodd_GPU_Double(int N, double h, double *U, double *F){
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
		i = i + blockDim.x * gridDim.x;
	}

}

__global__ void ker_GaussSeidelodd_GPU_Single(int N, float h, float *U, float *F){
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

__global__ void ker_Error_GPU_Double(int N, double h, double *U, double *F, double *err){
	// Settings for parallel reduction to calculate the error array 
	extern __shared__ __align__(sizeof(double)) unsigned char cache_error_double[];
	double *cache = reinterpret_cast<double *>(cache_error_double);
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

__global__ void ker_Error_GPU_Single(int N, float h, float *U, float *F, float *err){
	// Settings for parallel reduction to calculate the error array 
	extern __shared__ __align__(sizeof(float)) unsigned char cache_error_single[];
	float *cache = reinterpret_cast<float *>(cache_error_single);
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

__global__ void ker_Zoom_GPU(int N, float h_n, int M, float h_m, float *U_m){
	// Texture memory
	// texMem_float1 -> U_n
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
			bl = tex1Dfetch(texMem_float1, index_n);
			br = tex1Dfetch(texMem_float1, index_n + 1);
			tl = tex1Dfetch(texMem_float1, index_n + N);
			tr = tex1Dfetch(texMem_float1, index_n + N + 1);

			// Zooming and store inside U_m
			U_m[index_m] = b * d * bl + a * d * br + c * b * tl + a * c * tr;
		}

		// Stride
		index_m = index_m + blockDim.x * gridDim.x;
	}
	
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

	cudaBindTexture(NULL, texMem_float1, d_U, N * N * sizeof(float));
	cudaBindTexture(NULL, texMem_float2, d_F, N * N * sizeof(float));

	cudaMemcpy(d_F, h_F, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_U, h_U, N * N * sizeof(float), cudaMemcpyHostToDevice);

	ker_Residual_GPU <<< blocksPerGrid, threadsPerBlock >>> (N, (float)h, d_D);
	
	cudaMemcpy(h_D, d_D, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaUnbindTexture(texMem_float1);
	cudaUnbindTexture(texMem_float2);

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

void doGridAddition_GPU(int N, double *U1, double *U2){
	// Settings
	int blocksPerGrid = 10;
	int threadsPerBlock = 10;
	float *d_U1, *d_U2;		// device memory
	float *h_U1, *h_U2;		// host memory

	/*
	CPU Part
	 */
	// Allocate host memory
	h_U1 = (float*) malloc(N * N * sizeof(float));
	h_U2 = (float*) malloc(N * N * sizeof(float));
	
	// Transfer data from float to double
	#	pragma omp parallel for
	for(int i = 0; i < N*N; i = i+1){
		h_U1[i] = (float) U1[i];
		h_U2[i] = (float) U2[i];
	}

	/*
	GPU Part
	 */
	cudaMalloc((void**)&d_U1, N * N * sizeof(float));
	cudaMalloc((void**)&d_U2, N * N * sizeof(float));

	cudaBindTexture(NULL, texMem_float1, d_U2, N * N * sizeof(float));

	cudaMemcpy(d_U1, h_U1, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_U2, h_U2, N * N * sizeof(float), cudaMemcpyHostToDevice);

	ker_GridAddition_GPU <<< blocksPerGrid, threadsPerBlock >>> (N, d_U1);

	cudaMemcpy(h_U1, d_U1, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaUnbindTexture(texMem_float1);

	cudaFree(d_U1);
	cudaFree(d_U2);

	// Transfer data from float back to double
	#	pragma omp parallel for
	for(int i = 0; i < N*N; i = i+1){
		U1[i] = (double) h_U1[i];
	}

	free(h_U1);
	free(h_U2);
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

	// Bind d_F to texture memory
	cudaBindTexture(NULL, texMem_float2, d_F, N * N * sizeof(float));

	// Copy data to device memory
	cudaMemcpy( d_U,  h_U, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_U0,  d_U, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy( d_F,  h_F, N * N * sizeof(float), cudaMemcpyHostToDevice);

	free(h_F);    // h_F are no longer needed

	// Do the iteration with "step" steps
	while( iter <= step ){
		
		ker_Smoothing_GPU <<< blocksPerGrid, threadsPerBlock, sharedMemorySize >>> (N, (float) h, d_U, d_U0, iter, d_err);
		
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

	// Unbind texture memory
	cudaUnbindTexture(texMem_float2);

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

	// Gauss-Seidel Even/Odd Method with Double Precision
	if( option == 1 ){
		GaussSeidel_GPU_Double(N, L, U, F, target_error);
	}
    
	// Gauss-Seidel Even/Odd Method with Single Precision
	if( option == 2 ){
		GaussSeidel_GPU_Single(N, L, U, F, target_error);
	}

}

void doRestriction_GPU(int N, double *U_f, int M, double *U_c){

	// Settings
	double h_f = 1.0 / (double) (N - 1);	// spacing in finer grid
	double h_c = 1.0 / (double) (M - 1);	// spacing in coarser grid

	// Settings for GPU
	int blocksPerGrid = 10;
	int threadsPerBlock = 10;
	float *d_Uf, *d_Uc;
	float *h_Uf, *h_Uc;

	/*
	CPU Part
	 */
	// Allocate host memory
	h_Uf = (float*) malloc(N * N * sizeof(float));
	h_Uc = (float*) malloc(M * M * sizeof(float));

	// Transfer data from double to float
	#	pragma omp parallel for
	for(int i = 0; i < N*N; i = i+1){
		h_Uf[i] = (float) U_f[i];
	}

	/*
	GPU Part
	 */
	// Allocate device memory
	cudaMalloc((void**)&d_Uf, N * N * sizeof(float));
	cudaMalloc((void**)&d_Uc, M * M * sizeof(float));

	// Bind d_Uf to texture memory
	cudaBindTexture(NULL, texMem_float1, d_Uf, N * N * sizeof(float));

	// Copy data to device memory and initialize d_Uc as zeros
	cudaMemcpy(d_Uf, h_Uf, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(d_Uc, 0.0, M * M * sizeof(float));

	free(h_Uf);		// h_Uf is no longer needed

	// Call the kernel
	ker_Zoom_GPU <<< blocksPerGrid, threadsPerBlock >>> (N, (float)h_f, M, (float)h_c, d_Uc);

	// Copy data back to host memory
	cudaMemcpy(h_Uc, d_Uc, M * M * sizeof(float), cudaMemcpyDeviceToHost);

	// Unbind texture memory and free the device memory
	cudaUnbindTexture(texMem_float1);
	cudaFree(d_Uf);
	cudaFree(d_Uc);

	// Transfer data from float to double
	#	pragma omp parallel for
	for(int i = 0; i < M*M; i = i+1){
		U_c[i] = (double) h_Uc[i];
	}

	free(h_Uc);
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
	cudaBindTexture(NULL, texMem_float1, d_Uc, N * N * sizeof(float));

	// Copy data to device memory and initialize d_Uf as zeros
	cudaMemcpy(d_Uc, h_Uc, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(d_Uf, 0.0, M * M * sizeof(float));

	free(h_Uc);		// h_Uc is no longer needed
	
	// Call the kernel
	ker_Zoom_GPU <<< blocksPerGrid, threadsPerBlock >>> (N, (float) h_c, M, (float) h_f, d_Uf);

	// Copy data back to host memory
	cudaMemcpy(h_Uf, d_Uf, M * M * sizeof(float), cudaMemcpyDeviceToHost);

	// Unbind texture memory and free the device memory
	cudaUnbindTexture(texMem_float1);
	cudaFree(d_Uf);
	cudaFree(d_Uc);

	// Transfer data from float to double
	#	pragma omp parallel for
	for(int i = 0; i < M*M; i = i+1){
		U_f[i] = (double) h_Uf[i];
	}

	free(h_Uf);	
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

void GaussSeidel_GPU_Double(int N, double L, double *U, double *F, double target_error){
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
	int sharedMemorySize;
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
	cudaMemset(d_U, 0.0, N * N * sizeof(double));
	cudaMemcpy(d_F, F, N * N * sizeof(double), cudaMemcpyHostToDevice);
	
	// Do the iteration until it is smaller than the target_error
	while( error > target_error ){
		// Iteration Even / Odd index
		ker_GaussSeideleven_GPU_Double <<< blocksPerGrid, threadsPerBlock >>> (N, h, d_U, d_F);
		ker_GaussSeidelodd_GPU_Double <<< blocksPerGrid, threadsPerBlock >>> (N, h, d_U, d_F);
		
		// Get the error
		ker_Error_GPU_Double <<< blocksPerGrid, threadsPerBlock, sharedMemorySize >>> (N, h, d_U, d_F, d_err);
		cudaMemcpy(h_err, d_err, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);

		// Calculate the error from error array
		error = 0.0;
		for(int i = 0; i < blocksPerGrid; i = i+1){
			error = error + h_err[i];
		}
		error = error / (double)(N*N);
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

void GaussSeidel_GPU_Single(int N, double L, double *U, double *F, double target_error){
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
		ker_GaussSeideleven_GPU_Single <<< blocksPerGrid, threadsPerBlock >>> (N, (float) h, d_U, d_F);
		ker_GaussSeidelodd_GPU_Single <<< blocksPerGrid, threadsPerBlock >>> (N, (float) h, d_U, d_F);
		
		// Get the error
		ker_Error_GPU_Single <<< blocksPerGrid, threadsPerBlock, sharedMemorySize >>> (N, (float) h, d_U, d_F, d_err);
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
