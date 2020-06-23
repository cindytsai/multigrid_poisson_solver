/*
nvcc -arch=compute_52 -code=sm_52,sm_52 -O3 -m64 textureMemory.cu
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// declare the texture
__align__(8) texture<float>  texMem;

__global__ void readTexture(int N, float *output){
	
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	
	while(index < N){

		output[index] = output[index] + tex1Dfetch(texMem, index);
		
		index = index + blockDim.x * gridDim.x;
	}

}

int main( int argc, char *argv[] ){
	int blocksPerGrid;
	int threadsPerBlock;
	int N;

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

	float *d_A, *d_B;
	float *A, *B, *C;
	float *C_CPU;		// C = 2A + B

	A = (float*) malloc(N * sizeof(float));
	B = (float*) malloc(N * sizeof(float));
	C = (float*) malloc(N * sizeof(float));
	C_CPU = (float*) malloc(N * sizeof(float));

	for(int i = 0; i < N; i = i+1){
		A[i] = rand() / (float)RAND_MAX;
		B[i] = rand() / (float)RAND_MAX;
	}

	for(int i = 0; i < N; i = i+1){
		C_CPU[i] = 2 * A[i] + B[i];
	}

	cudaMalloc((void**)&d_A, N * sizeof(float));
	cudaMalloc((void**)&d_B, N * sizeof(float));

	cudaBindTexture(NULL, texMem, d_A, N * sizeof(float));

	cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

	readTexture <<< blocksPerGrid, threadsPerBlock >>> (N, d_B);

	cudaUnbindTexture(texMem);

	cudaBindTexture(NULL, texMem, d_B, N * sizeof(float));

	readTexture <<< blocksPerGrid, threadsPerBlock >>> (N, d_A);

	cudaMemcpy(C, d_A, N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaUnbindTexture(texMem);

	cudaFree(d_A);
	cudaFree(d_B);

	float diff = 0.0;

	for(int i = 0; i < N; i = i+1){
		printf("%lf    %lf\n", C[i], C_CPU[i]);
		diff = diff + fabs(C[i] - C_CPU[i]);
	}
	diff = diff / (float) N;

	printf("diff = %lf\n", diff);

	free(A);
	free(B);
	free(C);
	free(C_CPU);

	return 0;
}