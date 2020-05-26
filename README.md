# Multigrid Poisson Solver
Specified on Laplacian Operator.

## Basic Notation
Suppose the interest region is a square only.

| Variable Name |                 Definition                |            Type            |
|:-------------:|:-----------------------------------------:|:--------------------------:|
|       L       | Width of the interest region.             |           double           |
|       U       | Approximate solution of size N x N.       | double*, 1-D array address |
|       F       | Discretize source function to size N x N. | double*, 1-D array address |
|       D       | Residual of size N x N.                   | double*, 1-D array address |

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

### Source
* void getSource: Get the discretize form of the source function of size N x N.
* Input Variable:
  * Grid size: `int N`
  * L: `double L`
  * Source f [1D-array address]: `double *F`
  * x minimum range: `double min_x`
  * y minimum range: `double min_y`

### Boundary
* void getBoundary: Get the discretize form of the boundary of size N x N, only the boundary are setted, the others are 0.
* Input Variable:
  * Grid size: `int N`
  * L: `double L`
  * Boundary [1D-array address]: `double *F`
  * x minimum range: `double min_x`
  * y minimum range: `double min_y`

### Residual
* void getResidual: Get the residual d = Lu - f
* Input Variable
  * Grid size: `int N`
  * L: `double L`
  * Approximate solution [1D-array address]: `double *U`
  * Source f [1D-array address]: `double *F`
  * Residual d [1D-array address]: `dobule *D`

### Smoothing
* void doSmoothing: Change made inside `double *U`
* Input Variable
  * Grid size: `int N`
  * L: `double L`
  * Approximate solution [1D-array address]: `double *U`
  * Source f [1D-array address]: `double *F`
  * Steps: `int step`

### Exact Solver
* void doExactSolver: Change made inside `double *U`
* Input Variable
  * Grid size: `int N`
  * L: `double L`
  * Approximation solution [1D-array address]: `double *U`
  * Souce f [1D-array address]: `double *F`
  * Target error: `double target_error`
  * Solver options: `int option`
    * `option == 0`: use Inverse Matrix
    * `option == 1`: use Gauss-Seidel, with even/odd method.

### Restriction
* void doRestriction: Change made inside `double *U_c`
* Input Variable
  * Grid size: `int N`
  * To be restrict [1D-array address]: `double *U_f`
  * Grid size: `int M`
  * After restriction[1D-array address]: `double *U_c`

### Prolongation
* void doProlongation: Change made inside `double *U_f`
* Input Variable
  * Grid size: `int N`
  * To be prolongate [1D-array address]: `double *U_c`
  * Grid size: `int M`
  * After prolongation [1D-array address]: `double *U_f`
