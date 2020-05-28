#include <cstdio>
#include <cstdlib>
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


void doProlongation(int N, double* U_c, int M, double* U_f){
	double L = 1.0;
	double c_dx = L / (N-1), f_dx = L/(M-1);
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

/*
Functions
 */