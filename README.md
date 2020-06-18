# Multigrid Poisson Solver
Specified on Laplacian Operator.

## Contents
* [Basic Notation](https://github.com/cindytsai/multigrid_poisson_solver#basic-notation)
* [Workflow and Structure](https://github.com/cindytsai/multigrid_poisson_solver#workflow-and-structure)
* [CPU Functions](https://github.com/cindytsai/multigrid_poisson_solver#cpu-functions)
* [GPU Functions](https://github.com/cindytsai/multigrid_poisson_solver#gpu-functions)

## Basic Notation
Suppose the interest region is a square only.

| Variable Name |                 Definition                |            Type            |
|:-------------:|:-----------------------------------------:|:--------------------------:|
|       L       | Width of the interest region.             |           double           |
|       U       | Approximate solution of size N x N.       | double*, 1-D array address |
|       F       | Discretize source function to size N x N. | double*, 1-D array address |
|       D       | Residual of size N x N.                   | double*, 1-D array address |

## Workflow and Structure
phi = ... </br>
N = ..</br>
cycle = 1D-array = {-1 , -1, 0, 1, 1}</br>

### Link List Data Structure

#### Node
* 

#### Functions

### V-cycle / W-cycle
for i in cycle:</br>
&nbsp;if(cycle[i] == -1): doRestriction</br>
&nbsp;if(cycle[i] ==  0): doExactSolver</br>
&nbsp;if(cycle[i] ==  1): doProlongation</br>

### FMG
while():</br>
&nbsp;triggers:</br>

## CPU Functions
All of the grids are stored as 1D array, with size N x N, including the boundary.

### Source
* void getSource: Get the discretize form of the source function of size N x N. Change made inside `double *F`.
  * Input Variable:
    * Grid size: `int N`
    * Interest Region length L: `double L`
    * Source f [1D-array address]: `double *F`
    * x minimum range: `double min_x`
    * y minimum range: `double min_y`

### Boundary
* void getBoundary: Get the discretize form of the boundary of size N x N, only the boundary are setted, the others are 0.
  * Input Variable:
    * Grid size: `int N`
    * Interest Region length L: `double L`
    * Boundary [1D-array address]: `double *F`
    * x minimum range: `double min_x`
    * y minimum range: `double min_y`

### Residual
* void getResidual: Get the residual d = Lu - f
  * Input Variable:
    * Grid size: `int N`
    * Interest Region length L: `double L`
    * Approximate solution [1D-array address]: `double *U`
    * Source f [1D-array address]: `double *F`
    * Residual d [1D-array address]: `dobule *D`

### Smoothing
* void doSmoothing: Change made inside `double *U`
  * Input Variable:
    * Grid size: `int N`
    * Interest Region length L: `double L`
    * Approximate solution [1D-array address]: `double *U`
    * Source f [1D-array address]: `double *F`
    * Steps: `int step`
    * Error: `double *error`

### Exact Solver
* void doExactSolver: Change made inside `double *U`
  * Input Variable:
    * Grid size: `int N`
    * Interest Region length L: `double L`
    * Approximation solution [1D-array address]: `double *U`
    * Souce f [1D-array address]: `double *F`
    * Target error: `double target_error`
    * Solver options: `int option`
      * `option == 0`: use Inverse Matrix
      * `option == 1`: use Gauss-Seidel, with even/odd method.

#### Inverse Matrix Method
* void InverseMatrix: Calculate the Inverse Matrix of the current discretized Laplacian, and do the multiplication to get the answer `double *U`.
  * Input Variable:
    * Grid size: `int N`
    * Interest region length L: `double Length`
    * Exact solution [1D-array address]: `double *X`
    * Source Term f [1D-array address]: `double *F`
* **NOTES:**
  * Parallel with OpenMP later, since the performance is generally same as Gauss-Seidel, and it is hard to parallelize.

#### Gauss-Seidel Relaxation Method
* void GaussSeidel: Change made inside exact solution `double *U`. Relax till it reaches the target error.
  * Input Variable:
    * Grid size: `int N`
    * Interest Region length L: `double L`
    * Exact solution [1D-array address]: `double *U`
    * Source Term [1D-array address]: `double *F`
    * Target error: `double target_error`

### Restriction
* void doRestriction: Change made inside `double *U_c`
  * Input Variable:
    * Grid size: `int N`
    * To be restrict [1D-array address]: `double *U_f`
    * Grid size: `int M`
    * After restriction[1D-array address]: `double *U_c`
  * **NOTES:**
    1. Restriction is specific on residual, and since we only don't do relaxation on the boundary, so the boundary of restriction target grid is always "0".

### Prolongation
* void doProlongation: Change made inside `double *U_f`
  * Input Variable:
    * Grid size: `int N`
    * To be prolongate [1D-array address]: `double *U_c`
    * Grid size: `int M`
    * After prolongation [1D-array address]: `double *U_f`

### Print
* void doPrint: Print out the grid on screen, with the normal x-y coordinate.
  * Input Variable:
    * Grid size: `int N`
    * To be printed [1D-array address]: `double *U`

### Output as file
* void doPrint2File: Print in file, with normal x-y coordinate.
  * Input Variable:
    * Grid size: `int N`
    * To be printed [1D-array address]: `double *U`
    * output file name: `char *filename`
  
## GPU Functions
All of the grids are stored as 1D array, with size N x N, including the boundary.
Since the GPU is specialized in doing single precision computation, all the subroutine function of calling GPU kernal are using `single precision`, only ExactSolver is `double precision`. After calling GPU kernel, typecasting back to `double precision`.

### Source (Problem Dependent)
* void getSource_GPU: Get the discretize form of the source function of size N x N. Change made inside `double *F`.
  * Input Variable:
    * Grid size: `int N`
    * Interest Region length L: `double L`
    * Source f [1D-array address]: `double *F`
    * x minimum range: `double min_x`
    * y minimum range: `double min_y`

* \_\_global\_\_ void Source_GPU: Get the discretize form of the source function of size N x N. Change made inside `float *F`.
  * Input Variable:
    * Grid size: `int N`
    * delta x: `float h`
    * Source f [1D-array address]: `float *F`
    * x minimum range: `float min_x`
    * y minimum range: `float min_y`

### Residual
* void getResidual_GPU: Get the residual d = Lu - f
  * Input Variable:
    * Grid size: `int N`
    * Interest Region length L: `double L`
    * Approximate solution [1D-array address]: `double *U`
    * Source f [1D-array address]: `double *F`
    * Residual d [1D-array address]: `dobule *D`

* \_\_global\_\_ void Residual_GPU: Get the residual d = Lu - f, changes made inside `float D`
  * Input Variable:
    * Grid size: `int N`
    * Interest Region length L: `float L`
    * Approximate solution [1D-array address]: `float *U`
    * Source f [1D-array address]: `float *F`
    * Residual d [1D-array address]: `float *D`

### Error
* void getError_GPU: Get the error define as adding all the elements in the residual, save the result in `double *error`.
  * Input Variable:
    * Grid size: `int N`
    * Interest Region length L: `double L`
    * Approximate solution [1D-array address]: `double *U`
    * Source f [1D-array address]: `double *F`
    * Residual d [1D-array address]: `dobule *D`
    * Error: `double *error`

* \_\_global\_\_ void Error_GPU: Save the result in `float *error`.
  * Input Variable:
    * Grid size: `int N`
    * Interest Region length L: `float L`
    * Approximate solution [1D-array address]: `float *U`
    * Source f [1D-array address]: `float *F`
    * Residual d [1D-array address]: `float *D`
    * Error: `float *error`

* **NOTES:**
  * Do this part if really needed.
  * Save this to the laste to finish.

### Smoothing
* void doSmoothing_GPU: Change made inside `double *U`, and save the error from the latest smoothing in `double *error`.
  * Input Variable:
    * Grid size: `int N`
    * Interest Region length L: `double L`
    * Approximate solution [1D-array address]: `double *U`
    * Source f [1D-array address]: `double *F`
    * Steps: `int step`
    * Error: `double *error`

* \_\_global\_\_ void Smoothing_GPU: Change made inside `float *U`, and save the error from the latest smoothing in `float *error`.
  * Input Variable:
    * Grid size: `int N`
    * Interest Region length L: `float L`
    * Approximate solution [1D-array address]: `float *U`
    * Source f [1D-array address]: `float *F`
    * Steps: `int step`
    * Error: `float *error`

### Exact Solver
* void doExactSolver_GPU: Change made inside `double *U`.
  * Input Variable:
    * Grid size: `int N`
    * Interest Region length L: `double L`
    * Approximation solution [1D-array address]: `double *U`
    * Souce f [1D-array address]: `double *F`
    * Target error: `double target_error`
    * Solver options: `int option`
      * `option == 0`: _blank_
      * `option == 1`: use Gauss-Seidel, with even/odd method.

* \_\_global\_\_ void GaussSeidel_GPU: Change made inside `double *U`.
  * Input Variable:
    * Grid size: `int N`
    * Interest Region length L: `double L`
    * Approximation solution [1D-array address]: `double *U`
    * Souce f [1D-array address]: `double *F`
    * Target error: `double target_error`

### Restriction / Prolongation
* void doRestriction: Change made inside `double *U_c`
  * Input Variable:
    * Grid size: `int N`
    * To be restrict [1D-array address]: `double *U_f`
    * Grid size: `int M`
    * After restriction[1D-array address]: `double *U_c`

* void doProlongation: Change made inside `double *U_f`
  * Input Variable:
    * Grid size: `int N`
    * To be prolongate [1D-array address]: `double *U_c`
    * Grid size: `int M`
    * After prolongation [1D-array address]: `double *U_f`

* \_\_global\_\_ void Zoom_GPU: Zoom in/out from grid size `int N` to grid size `int M`. The final result is inside `float *U_m`.
  * Input Variable:
    * Grid size before zooming: `int N`
    * Grid to be zoomed [1D-array address]: `float *U_n`
    * Grid size after zooming: `int M`
    * Result of the grid: `float *U_m`

* **NOTES:**
  1. Restriction is specific on residual, and since we only don't do relaxation on the boundary, so the boundary of restriction target grid is always "0".

