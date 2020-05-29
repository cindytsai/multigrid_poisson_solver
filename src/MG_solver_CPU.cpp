#include <cstdio>
#include <cstdlib>
#include <stdio.h> 
#include <stdlib.h> 
#include <omp.h>
#include <math.h>
#include <cstring>
#include <assert.h>

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
void doPrint2File(int, double*, char*);

// Sub Routine
void InverseMatrix(int, double, double*, double*);
void GaussSeidel(int, double, double*, double*, double);


int main(){
    /*
	Settings
	cycle: The cycle
	N:     Grid size
	 */
	int N = 129;
	int M = 65;
	double L = 1.0;
	int step = 10;
	char file_name[50];
	double *U, *F, *D, *D_c, *V, *V_f;

	U 	= (double*) malloc(N * N * sizeof(double));
	F 	= (double*) malloc(N * N * sizeof(double));
	D 	= (double*) malloc(N * N * sizeof(double));
	D_c = (double*) malloc(M * M * sizeof(double));
	V 	= (double*) malloc(M * M * sizeof(double));
	V_f = (double*) malloc(N * N * sizeof(double));

	// Initialize
	memset(U, 0.0, N*N*sizeof(double));
	getSource(N, L, F, 0.0, 0.0);

	doSmoothing(N, L, U, F, step);
	getResidual(N, L, U, F, D);
	doRestriction(N, D, M, D_c);
	for(int j = 0; j < M; j = j+1){
		for(int i = 0; i < M; i = i+1){
			D_c[i+j*M] = -D_c[i+j*M];
		}
	}
	// D_c = -D_c
	doExactSolver(M, L, V, D_c, 0.001, 1);
	doProlongation(M, V, N, V_f);
	
	for(int j = 0; j < N; j = j+1){
		for(int i = 0; i < N; i = i+1){
			U[i+j*N] = U[i+j*N] + V_f[i+j*N];
		}
	}

	doSmoothing(N, L, U, F, step);

	strcpy(file_name, "Two_Grid-test.txt");
	doPrint2File(N, U, file_name);
	
	free(U);
	free(F);
	free(D);
	free(D_c);
	free(V);
	free(V_f);

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

	memset(F, 0.0, N * N * sizeof(double));

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
// Main Function
void getResidual(int N, double L, double* U, double* F, double* D){
	double dx=L/(double) (N-1);
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			if (i==0 or j==0 or j==N-1 or i==N-1)  *(D+i*N+j)=0.0;
			else  *(D+i*N+j) =  (1.0/pow(dx,2)) * (*(U+(i+1)*N+j) + *(U+(i-1)*N+j) + *(U+i*N+(j+1)) + *(U+i*N+(j-1)) - 4* *(U+i*N+j)) - *(F+i*N+j);
		}
	}

}

void doSmoothing(int N, double L, double* U, double* F, int step){
	double dx=L/(double) (N-1);
	double delta;
	//Gauss-Seidel
	for(int s=0; s<step; s++){
		for(int i=1; i<N-1; i++){
			for(int j=1; j<N-1; j++){
				delta = 0.25*( *(U+(i+1)*N+j) + *(U+(i-1)*N+j) + *(U+i*N+(j+1)) + *(U+i*N+(j-1)) - 4* *(U+i*N+j) - pow(dx,2) * *(F+i*N+j));
		        *(U+i*N+j) += delta;
			}
		}
	}
	
}

void doExactSolver(int N, double L, double *U, double *F, double target_error, int option){

	// Inverse Matrix
	if(option == 0){
		InverseMatrix(N, L, U, F);
	}

	// Gauss-Seidel even / odd method
	if(option == 1){
		GaussSeidel(N, L, U, F, target_error);
	}
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

void doProlongation(int N, double* U_c, int M, double* U_f){
	double L = 1.0;
	double c_dx = L / (double) (N-1), f_dx = L/(double) (M-1);
	double ratio = c_dx/f_dx;

	for(int i=0; i<N-1; i++){
		for(int j=0; j<N-1; j++){
			double c1x = j*c_dx, c1y = i*c_dx;
			double c2x = c1x+c_dx, c2y = c1y;
			double c3x = c1x, c3y = c1y+c_dx;
			double c4x = c2x, c4y = c3y;
			double c1 = *(U_c + i*N + j), c2 = *(U_c + i*N + j+1);
			double c3 = *(U_c + (i+1)*N + j), c4 = *(U_c + (i+1)*N +j+1);
			
			for(int k=ceil(i*ratio); k<ceil((i+1)*ratio); k++){
				for(int l=ceil(j*ratio); l<ceil((j+1)*ratio); l++){
					double f_x = l*f_dx, f_y =k*f_dx;

					assert(c2x - f_x >= 0);
					assert(f_x - c1x >= 0);
					assert(c4x - f_x >= 0);
					assert(f_x - c3x >= 0);
					assert(c3y - f_y >= 0);
					assert(f_y - c1y >= 0);
					*(U_f + k*M + l) = ((c1*(c2x - f_x) + c2*(f_x - c1x))*(c3y - f_y)  + (c3 * (c4x - f_x) + c4 * (f_x - c3x))*(f_y - c1y))/c_dx/c_dx;
					if (l==M-2){
						f_x = L;
						*(U_f+ k*M + M-1) = ((c1 * (c2x - f_x) + c2 * (f_x - c1x))*(c3y - f_y)  + (c3 * (c4x - f_x) + c4 * (f_x - c3x))*(f_y - c1y))/c_dx/c_dx;
					}

				}
				if(k==M-2){
					k=M-1;
					double f_x = 0, f_y = L;
					for(int l=ceil(j*ratio); l<ceil((j+1)*ratio); l++){
						f_x = l*f_dx;
						f_y = k*f_dx;
						assert(c2x - f_x >= 0);
						assert(f_x - c1x >= 0);
						assert(c4x - f_x >= 0);
						assert(f_x - c3x >= 0);
						assert(c3y - f_y >= 0);
						assert(f_y - c1y >= 0);
						*(U_f + (M-1)* M + l) = ((c1 * (c2x - f_x) + c2 * (f_x - c1x))*(c3y - f_y)  + (c3 * (c4x - f_x) + c4 * (f_x - c3x))*(f_y - c1y))/c_dx/c_dx;
						if (l==M-2){
							f_x = L;
							*(U_f+ k*M + M-1) = ((c1 * (c2x - f_x) + c2 * (f_x - c1x))*(c3y - f_y)  + (c3 * (c4x - f_x) + c4 * (f_x - c3x))*(f_y - c1y))/c_dx/c_dx;
						}
					}
				}	
			}
			
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


// Sub Routine
void InverseMatrix(int N, double Length, double *X, double *F){
	/*
	Settings and Initialization
	 */
	double lu_sum;						// For calculating LU-decomposition
	int swap_row, check_row, flag;

	double sum;							// For solving Ax = b

	double h = Length / (double) (N - 1); 	// delta x of discretized grid
	double h2 = pow(h, 2);				// h ** 2
	int index;							// Index of the 1D-array U (Discretize grid)
	int l, r, t, b;						// Index of the neighboring in 1D-array U (Discretize grid)
	int row;							// Row of the Laplacian Matrix
	double *A;							// Discretized Laplacian operator Matrix form.
	double *L, *U;						// For LU-decomposition , PA = LU
	int *P;
	double *Z;							// Solve for Ax = b -> LUx = b -> z = Ux, Lz = b
	int MatrixSize = N * N;
	
	A = (double*) malloc(MatrixSize * MatrixSize * sizeof(double));
	L = (double*) malloc(MatrixSize * MatrixSize * sizeof(double));
	U = (double*) malloc(MatrixSize * MatrixSize * sizeof(double));
	P = (int*) malloc(MatrixSize * sizeof(int));
	Z = (double*) malloc(MatrixSize * sizeof(double));

	memset(A, 0.0, MatrixSize * MatrixSize * sizeof(double));
	memset(L, 0.0, MatrixSize * MatrixSize * sizeof(double));
	memset(U, 0.0, MatrixSize * MatrixSize * sizeof(double));
	memset(P, 0.0, MatrixSize * sizeof(int));
	memset(X, 0.0, MatrixSize * sizeof(double));
	memset(Z, 0.0, MatrixSize * sizeof(double));
	
	// Set the diagonal of U as 1
	// Set Permutation P equal to its own index from each row
	for(int i = 0; i < MatrixSize; i = i+1){
		for(int j = 0; j < MatrixSize; j = j+1){
			if( i == j){
				U[i * MatrixSize + j] = 1.0;
			}
		}
		P[i] = i;
	}

	/*
	Create the Laplacian Operator A_mn
	m -> row, n -> column, same indexing as the ordinary matrix.
	For each grid points, corresponds to one Laplacian Matrix A row.
	 */
	row = 0;	// Start from 0th-row of Laplacian Matrix A
	for(int j = 0; j < N; j = j+1){
		for(int i = 0; i < N; i = i+1){
			
			if( i == 0 || i == N-1 || j == 0 || j == N-1){
				// Boundary after laplacian Matrix is itself -> Set as 1
				index = i + N * j;
				A[index + row * MatrixSize] = 1.0;
			}
			else{
				// Others
				index = i + N * j;
				l = index - 1;
				r = index + 1;
				t = index + N;
				b = index - N;
				A[index + row * MatrixSize] = -4.0 / h2;
				A[l + row * MatrixSize] = 1.0 / h2;
				A[r + row * MatrixSize] = 1.0 / h2;
				A[t + row * MatrixSize] = 1.0 / h2;
				A[b + row * MatrixSize] = 1.0 / h2;
			}

			// Go to next row in Laplacian Matrix A
			row = row + 1;
		}
	}

	// DEBUG info
	// printf("----Laplacian Matrix----\n");
	// doPrint(MatrixSize, A);
	
	/*
	Do the LU-decomposition of Laplacian Operator A
	PA = LU
	 */
	flag = 0;
	check_row = 0;
	swap_row = check_row + 1;

	do{
		for(int k = 0; k < MatrixSize; k = k+1){
			
			// Compute L
			for(int i = k; i < MatrixSize; i = i+1){
				
				// Compute L_ik
				lu_sum = 0.0;
				for(int j = 0; j < k; j = j+1){
					lu_sum = lu_sum + L[MatrixSize*i+j] * U[MatrixSize*j+k];
				}
				L[MatrixSize*i+k] = A[P[i]*MatrixSize+k] - lu_sum;
			

				// Check if L[i][i] == 0.0, if so, swap row i with row below i ( swap_row > i )
				if(i == k && L[i*MatrixSize+k] == 0.0){
					
					if(swap_row >= MatrixSize){
						// Which is impossible in Laplacian Matrix :)
						printf("Having Zero Pivote ! det(A) = 0\n");
						flag = 0;
					}
					else{
						P[i] = swap_row;
						P[swap_row] = i;
						flag = 1;
					}
					swap_row = swap_row + 1;
					break;
				}
	
				if(i == k && L[i*MatrixSize+k] != 0.0){
					check_row = check_row + 1;
					swap_row = check_row + 1;
					flag = 0;
				}
			}

			if(flag == 1) break;

			// Compute U
			for(int j = k; j < MatrixSize; j = j+1){
				lu_sum = 0.0;
				for(int i = 0; i < k; i = i+1){
					lu_sum = lu_sum + L[k*MatrixSize+i] * U[i*MatrixSize+j];
				}
				U[k*MatrixSize+j] = (1.0 / L[k*MatrixSize+k]) * (A[P[k]*MatrixSize+j] - lu_sum);
			}
		}

	}while(flag != 0);

	// DEBUG info
	// printf("----L----\n");
	// doPrint(MatrixSize, L);
	// printf("----U----\n");
	// doPrint(MatrixSize, U);
	// printf("----P----\n");
	// for(int i = 0; i < MatrixSize; i = i+1){
	// 	printf("%d\n", P[i]);
	// }
	
	/*
	Solve for U, where LU = F or Ax = b equation
	Ax = b and PA = LU
	LUx = b;
	Lz = b; 
	*/
	
	// Solve for Z
	for(int i = 0; i < MatrixSize; i = i+1){
		sum = 0.0;
		for(int k = 0; k < i; k = k+1){
			sum = sum + L[i*MatrixSize+k] * Z[k];
		}
		Z[i] = (1.0 / L[i*MatrixSize+i]) * (F[i] - sum);
	}

	// Solve for X
	for(int i = MatrixSize-1; i >= 0; i = i-1){
		sum = 0.0;
		for(int k = i+1; k < MatrixSize; k = k+1){
			sum = sum + U[i*MatrixSize+k] * X[k];
		}
		X[i] = Z[i] - sum;
	}

	// DEBUG info
	// printf("CHECK INVERSE METHOD\n");
	// for(int i = 0; i < MatrixSize; i = i+1){
	// 	sum = 0.0;
	// 	for(int j = 0; j < MatrixSize; j = j+1){
	// 		sum = sum + A[i*MatrixSize+j] * X[j];
	// 	}
	// 	printf("%lf\n", sum - F[i]);
	// }
	
	
	// Free the temperary resource
	free(A);
	free(U);
	free(L);
	free(P);
	free(Z);
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
	memset(U_old, 0.0, N * N * sizeof(double));
	memset(U, 0.0, N * N * sizeof(double));

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


