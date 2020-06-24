#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "LinkList.h"

/*
GPU kernel
 */
__global__ void ker_Source_GPU(int, float, float*, float, float);
__global__ void ker_Residual_GPU(int, float, float*);
__global__ void ker_Smoothing_GPU(int, float, float*, float*, float*, int, float*);
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
	Settings
	cycle: The cycle
	N:     Grid size
	 */
	int len = 18;
	int cycle[len]={-1,-1,-1,0,1,-1,0,1,1,-1,-1,0,1,-1,0,1,1,1};
	//int len = 11;
	//int cycle[len]={-1,-1,-1,-1,-1,0,1,1,1,1,1};
	int N = 256;
	int M;
	int step = 1;
	char file_name[50];
	double *U, *F, *D, *D_c, *V, *V_f;
	double smoothing_error = 0;


	LinkedList* list = new LinkedList(N);
	double L= list->Get_L();
	ListNode* fine_node, * coarse_node;

	for(int ll=0; ll<len; ll++){
		if (cycle[ll]==-1){
			printf("go to coarser level\n");
			fine_node = list->Get_coarsest_node();
			N   = fine_node->Get_N();
			U 	= fine_node->Get_U();
			F 	= fine_node->Get_F();
			D 	= fine_node->Get_D();

			list -> Push();
			coarse_node = list->Get_coarsest_node();
			M = coarse_node->Get_N();

			D_c = coarse_node->Get_F();
			V 	= coarse_node->Get_U();
			// Initialize
			memset(U, 0.0, N*N*sizeof(double));
			getSource_GPU(N, L, F, 0.0, 0.0);

			doSmoothing_GPU(N, L, U, F, step, &smoothing_error);
			printf("smoothing error= %f\n", smoothing_error);
			getResidual_GPU(N, L, U, F, D);
			doRestriction_GPU(N, D, M, D_c);
			for(int j = 0; j < M; j = j+1){
				for(int i = 0; i < M; i = i+1){
					D_c[i+j*M] = -D_c[i+j*M];
				}
			}
		}

		if(cycle[ll]==0) { 
			printf("DoExactSolver\n");
			doExactSolver_GPU(M, L, V, D_c, 0.000001, 1);

		} 
	
		if(cycle[ll]==1){
			printf("Go to finer level\n");
			coarse_node = list->Get_coarsest_node();
			M = coarse_node->Get_N();
			V 	= coarse_node->Get_U();
			fine_node = coarse_node->Get_prev();
			N = fine_node->Get_N();
			U = fine_node->Get_U();
			F = fine_node->Get_F();


			V_f = (double*) malloc(N * N * sizeof(double));

		
			doProlongation_GPU(M, V, N, V_f);
			
			for(int j = 0; j < N; j = j+1){
				for(int i = 0; i < N; i = i+1){
					U[i+j*N] = U[i+j*N] + V_f[i+j*N];
				}
			}
			doSmoothing_GPU(N, L, U, F, step, &smoothing_error);
			printf("smoothing error=%f\n", smoothing_error);
			free(V_f);
			list->Pop();

		}
	}
	D = fine_node->Get_D();

	getResidual_GPU(N, L, U, F, D);
	double error=0;

	for(int j=0; j<N; j++){
		for(int i=0; i<N; i++){
			error+=fabs(D[i+j*N]);
		}
	}
	error = error/N/N;

	printf("error = %f\n", error);
	strcpy(file_name, "MG_GPU_Test.txt");
	doPrint2File(N, U, file_name);

	delete list;

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

__global__ void ker_Smoothing_GPU(int N, float h, float *U, float *U0, float *F, int iter, float *err){
	
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
	cudaMemcpy(d_U0,  d_U, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy( d_F,  h_F, N * N * sizeof(float), cudaMemcpyHostToDevice);

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
