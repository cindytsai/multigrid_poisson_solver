# Multigrid Poisson Solver
Specified on Laplacian Operator.

## Structure
phi = ... </br>
N = ..</br>
cycle = 1D-array = {-1 , -1, 0, 1, 1}</br>

### V-cycle / W-cycle
for i in cycle:</br>
&nbsp;if(cycle[i] == -1): doRestriction</br>
&nbsp;if(cycle[i] ==  0): doExactSolver</br>
&nbsp;if(cycle[i] ==  1): doProlongation</br>

### FMG
while():</br>
&nbsp;triggers:</br>

## Functions
All of the grids are stored as 1D array, with size N x N, including the boundary.

### Residual
* void getResidual: Get the residual d = Lu - f
* Input Variable
  * Grid size: `int N`
  * 

### Smoothing
* void doSmoothing: Change made inside `double *phi`
* Input Variable
  * Grid size: `int N`
  * 1D-array address: `double *phi`
  * Steps: `int step`

### Exact Solver
* void doExactSolver: Change made inside `double *phi`
* Input Variable
  * Grid size: `int N`
  * 1D-array address: `double *phi`

### Restriction
* void doRestriction: Change made inside `int *M` and `double *R_phi`
* Input Variable
  * Grid size: `int N`
  * 1D-array address: `double *phi`
  * Grid size (h/2): `int *M`
  * 1D-array address: `double *R_phi`

### Prolongation
* void doProlongation: Change made inside `int *M` and `double *P_phi`
* Input Variable
  * Grid size (h): `int N`
  * 1D-array address: `double *phi`
  * Grid size (h/2): `int *M`
  * 1D-array address: `double *P_phi`
