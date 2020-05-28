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
void getResidual(int, double, double*, double*, double*);

void doSmoothing(int, double, double*, double*, int);
void doExactSolver(int, double, double*, double*, double);
void doRestriction(int, double*, int, double*);
void doProlongation(int, double*, int, double*);




//int main(){
	/*
	Settings
	cycle: The cycle
	N:     Grid size
	 */
	//int cycle[5] = {-1, -1, 0, 1, 1};
	//int N;
	//int L;
	
	/*
	Main MultiGrid Solver
	 */
	// for(int i = 0; i < cycle.length; i = i+1){

	// }

	//return 0;
//}

/*

int main(){
	int N=3, M=5;
	double *U_f, *U_c;
	U_f = (double *)malloc(M*M*sizeof(double));
	U_c = (double *)malloc(N*N*sizeof(double));


	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			*(U_c+i*N+j)=(i+j)*10;
			printf("%lf ", *(U_c+i*N+j));
		}
		printf("\n");
	}
	doProlongation(N, U_c, M, U_f);

	for(int i=0; i<M; i++){
		for(int j=0; j<M; j++)
			printf("%lf ", *(U_f+i*M+j));
		printf("\n");
	}


}

*/

int main(){
	int N=16;
	double L=1.0;
	double *U, *F, *D;
	F = (double *)malloc(N*N*sizeof(double));
	U = (double *)malloc(N*N*sizeof(double));
	D = (double *)malloc(N*N*sizeof(double));

	getSource(N, L, F, 0.0, 0.0);
	getResidual(N, L, U, F, D);
	doSmoothing(N,  L, U, F, 100);



	FILE *file1;

    file1 = fopen("residual", "w");

  	for (int i = 0; i < N; i++) {
    	for (int j = 0; j < N; j++) {
      		if (j != N-1) {fprintf(file1, "%f,", *(D + i*N + j));}
      		else {fprintf(file1, "%f\n", *(D + i*N + j));}
    	}
  	}

  	FILE *file2;

    file2 = fopen("solution", "w");

  	for (int i = 0; i < N; i++) {
    	for (int j = 0; j < N; j++) {
      		if (j != N-1) {fprintf(file2, "%f,", *(U + i*N + j));}
      		else {fprintf(file2, "%f\n", *(U + i*N + j));}
      
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

void getResidual(int N, double L, double* U, double* F, double* D){
	double dx=L/(double) (N-1);
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			if (i==0 or j==0 or j==N-1)  *(D+i*N+j)=0.0;
			else  *(D+i*N+j) =  *(U+(i+1)*N+j) + *(U+(i-1)*N+j) + *(U+i*N+(j+1)) + *(U+i*N+(j-1)) - 4* *(U+i*N+j) - pow(dx,2) * *(F+i*N+j);
		}
	}

}


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
/*
Functions
 */