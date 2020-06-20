#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

/*
GPU kernel
 */
__global__ void Source_GPU(int, float, float*, float, float);
__global__ void Residual_GPU(int, float, float*, float*, float*);
__global__ void Error_GPU(int, float, float*, float*, float*, float*); // Do this part if needed
__global__ void Smoothing_GPU(int, float, float*, float*, int, float*);
__global__ void GaussSeidel_GPU(int, double, double*, double*, double);
__global__ void Zoom_GPU(int, float*, int, float*);

/*
Wrap the GPU kernel as CPU function
 */
void getSource_GPU(int, double, double*, double, double);
void getResidual_GPU(int, double, double*, double*, double*);
void getError_GPU(int, double, double*, double*, double*, double*); // Do this part if needed
void doSmoothing_GPU(int, double, double*, double*, int, double*);
void doExactSolver_GPU(int, double, double*, double*, double, int);
void doRestriction_GPU(int, double*, int, double*);
void doProlongation_GPU(int, double*, int, double*);
void doPrint(int, double*);
void doPrint2File(int, double*, char*);


int main(){
	// Select GPU to use
	cudaEvent_t start, stop;
    cudaError_t err = cudaSuccess;
    err = cudaSetDevice( 0 );
    if(err != cudaSuccess){
        printf("Cannot select GPU\n");
        exit(1);
    }

    // Set OpenMP Thread Number
    omp_set_num_threads( 16 );

    // Call the functions

	return 0;
}

/*
GPU Kernel
 */
__global__ void Source_GPU(int N, float h, float *F, float min_x, float min_y){
	// Thread index inside the GPU kernel
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int ix, iy;
	float x, y;

	while( index < N * N){
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

__global__ void Smoothing_GPU(int N, float L, float *U, float *F, int step, float *error){

}

__global__ void GaussSeidel_GPU(int N, 	double L, double *U, double *F, double target_error){

}

__global__ void Zoom_GPU(int N, float *U_n, int M, float *U_m){
	
}

void getSource_GPU(int N, double L, double *F, double min_x, double min_y){
	double h = L / (double)(N - 1);

	// Settings
	int blocksPerGrid = 10;
	int threadsPerBlock = 10;
	float *d_F;		// device memory
	float *h_F;		// host memory

	// CPU Part
	h_F = (float*) malloc(N * N * sizeof(float));

	// GPU Part
	cudaMalloc((void**)&d_F, N * N * sizeof(float));

	Source_GPU <<< blocksPerGrid, threadsPerBlock >>> (N, (float)h, d_F, (float)min_x, (float)min_y);

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

	// GPU Part
	cudaMalloc((void**)&d_D, N * N * sizeof(float));
	cudaMalloc((void**)&d_F, N * N * sizeof(float));
	cudaMalloc((void**)&d_U, N * N * sizeof(float));

	cudaMemcpy(d_F, h_F, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_U, h_U, N * N * sizeof(float), cudaMemcpyHostToDevice);

	Residual_GPU <<< blocksPerGrid, threadsPerBlock >>> (N, (float)h, d_U, d_F, d_D);
	
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