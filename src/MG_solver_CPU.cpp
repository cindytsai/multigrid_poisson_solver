#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <math.h>
#include <cstring>

/*
For problem to be solved
 */
void getSource(int, double, double*, double, double);
void getBoundary(int, double, double*, double, double);
/*
MultiGrid Functions
 */
// Main Function
void getResidual(int, double, double*, double*, double*);
void doSmoothing(int, double, double*, double*, int);
void doExactSolver(int, double, double*, double*, double, int);
void doRestriction(int, double*, int, double*);
void doProlongation(int, double*, int, double*);
void doPrint(int, double*);

// Sub Routine
void InverseMatrix(int, double*, double*);
void GaussSeidel(int, double, double*, double*, double);


int main(){
	/*
	Settings
	cycle: The cycle
	N:     Grid size
	 */
	// int cycle[5] = {-1, -1, 0, 1, 1};
	// int N;
	// int L;
	
	/*
	Main MultiGrid Solver
	 */
	// for(int i = 0; i < cycle.length; i = i+1){

	// }
	
	/*
	Testing getSource, GaussSeidel
	 */
	int N = 18;
	double L = 1.0;
	double target_error = 1e-10;
	double *U, *F;
	U = (double*) malloc(N * N * sizeof(double));
	F = (double*) malloc(N * N * sizeof(double));

	getSource(N, L, F, 0.0, 0.0);
	doPrint(N, F);
	GaussSeidel(N, L, U, F, target_error);
	doPrint(N, U);

	free(U);
	free(F);


	return 0;
}


/*
For the problem to be solved
 */

// Get the source f of the problem Lu = f
void getSource(int N, double L, double *F, double min_x, double min_y){

	double h = L / (double) (N-1);
	double x, y;
	int index;

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

// Get the boundary of the problem Lu = f
// Changes made in F on the boundary only, the others will initialized as 0.
void getBoundary(int N, double L, double *F, double min_x, double min_y){
	double h = L / (double) (N-1);
	double x, y;
	int index;

	memset(F, 0, N * N);

	for(int ix = 0; ix < N; ix = ix+1){
		// Bottom boundary
		F[ix] = 0.0;
		// Top boundary
		F[ix + N * (N - 1)] = 0.0;
	}
	for(int iy = 0; iy < N; iy = iy+1){
		// Left boundary
		F[N * iy] = 0.0;
		// Right boundary
		F[(N - 1) + N * iy] = 0.0;
	}
}

/*
MultiGrid Functions
 */
void doExactSolver(int N, double L, double *U, double *F, double target_error, int option){

	// // Inverse Matrix
	// if(option == 0){
	// 	InverseMatrix(N, U, F);
	// }

	// // Gauss-Seidel even / odd method
	// if(option == 1){
	// 	GaussSeidel(N, L, U, F, target_error);
	// }
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
	memset(U_c, 0, M * M * sizeof(double));

	// Run through the coarse grid and do restriction, 
	// but without the boundary, since it is all "0"
	for(int iy_c = 1; iy_c < (M-1); iy_c = iy_c+1){
		for(int ix_c = 1; ix_c < (M-1); ix_c = ix_c+1){
			
			// Calculate the ratio and the lower left fine grid index
			ix_f = (int) floor((double)ix_c * h_c / h_f);
			iy_f = (int) floor((double)iy_c * h_c / h_f);
			a = fmod((double)ix_c * h_c, h_f) / h_f;
			c = fmod((double)iy_c * h_c, h_f) / h_f;
			b = 1.0 - a;
			d = 1.0 - c;

			// // Debug
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

void InverseMatrix(int N, double *U, double *F){

	/*
	Create the Laplacian Operator
	 */
	
	/*
	Calculate the inverse Laplacian Operator
	 */
	
	/*
	Matrix Multiplication
	 */
	
}

void GaussSeidel(int N, double L, double *U, double *F, double target_error){
	
	double h = L / (double)(N - 1);
	double err = target_error + 1.0;
	int iter = 0;

	int index, ix, iy;	// Index of the point to be update
	int l, r, t, b;		// Index of the neighbers

	int *fw, *bw;		// Record the forward and backward index of the index
	int *ieven, *iodd;	// Index of even / odd chestbox
	double *U_old;		// For storing U during iteration
	double *Residual; 	// Get the residual to compute the error

	/*
	Prepared and initialize
	 */
	ieven = (int*) malloc(((N * N) / 2) * sizeof(int));
	iodd  = (int*) malloc(((N * N) / 2) * sizeof(int));
	fw = (int*) malloc(N * sizeof(int));
	bw = (int*) malloc(N * sizeof(int));
	U_old = (double*) malloc(N * N * sizeof(double));
	Residual = (double*) malloc(N * N * sizeof(double));

	// For even chestbox index
	for(int i = 0; i < ((N * N) / 2); i = i+1){
		int parity, ix, iy;
		ix = (2 * i) % N;
		iy = ((2 * i) / N) % N;
		parity = (ix + iy) % 2;
		ix = ix + parity;
		ieven[i] = ix + iy * N;
	}
	// For odd chestbox index
	for(int i = 0; i < ((N * N) / 2); i = i+1){
		int parity, ix, iy;
		ix = (2 * i) % N;
		iy = ((2 * i) / N) % N;
		parity = (ix + iy + 1) % 2;
		ix = ix + parity;
		iodd[i] = ix + iy * N;
	}

	// For forward and backward index
	for(int i = 0; i < N; i = i+1){
		fw[i] = (i + 1) % N;
		bw[i] = (i - 1 + N) % N;
	}

	// Initialize
	memset(U_old, 0, sizeof(U_old));
	memset(U, 0, sizeof(U));


	// Start the Gauss-Seidel Iteration
	while( err > target_error ){
		
		// Update even chestbox
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
			l = bw[ix] + iy * N;
			r = fw[ix] + iy * N;
			t = ix + fw[iy] * N;
			b = ix + bw[iy] * N;

			// Update result to U
			U[index] = 0.25 * (U_old[l] + U_old[r] + U_old[t] + U_old[b] - pow(h, 2) * F[index]);
		}

		// Update odd chestbox
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
			l = bw[ix] + iy * N;
			r = fw[ix] + iy * N;
			t = ix + fw[iy] * N;
			b = ix + bw[iy] * N;

			// Update result to U
			U[index] = 0.25 * (U[l] + U[r] + U[t] + U[b] - pow(h, 2) * F[index]);			
		}

		// Compute the error, without the boundary, since it is always "0"
		iter = iter + 1;
		err = 0.0;
		getResidual(N, L, U, F, Residual);
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
	free(fw);
	free(bw);
	free(ieven);
	free(iodd);
	free(U_old);
	free(Residual);
}