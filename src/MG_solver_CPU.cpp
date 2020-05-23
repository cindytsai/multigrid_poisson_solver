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
void getResidual(int, double, double*, double*, double*);

void doSmoothing(int, double, double*, double*, int);
void doExactSolver(int, double, double*, double*, double);
void doRestriction(int, double*, int, double*);
void doProlongation(int, double*, int, double*);


int main(){
	/*
	Settings
	cycle: The cycle
	N:     Grid size
	 */
	int cycle[5] = {-1, -1, 0, 1, 1};
	int N;
	int L;
	
	/*
	Main MultiGrid Solver
	 */
	// for(int i = 0; i < cycle.length; i = i+1){

	// }

	return 0;
}


/*
Functions
 */