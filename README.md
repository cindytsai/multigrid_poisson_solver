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

![Definition](/res/Definition.jpg)

## Workflow and Structure

### Compile
* Compile directly

Remember to change SM Version for the GPU.
```
g++ -fopenmp -o MG_CPU MG_solver_CPU.cpp
nvcc -arch=compute_52 -code=sm_52 -O3 --compiler-options -fopenmp -o MG_GPU MG_solver_GPU.cu
```
* Using `Makefile`

Change `SMVERSION` to your GPU version.
```
make
```
Clean up.
```
make clean
```

### Cycle Structure File
* **con_step**

| con_step |                     Operations                     |
|:--------:|:--------------------------------------------------:|
|    -1    | Use trigger that depends when to go to next level. |
|     0    | Assign smoothing steps in each level manually.     |
|    INT   | Do smoothing for this many steps at any level.     |

* **con_N**

| con_N |                              Operations                             |
|:-----:|:-------------------------------------------------------------------:|
|   0   | Assign grid size N at each level manually, and input minimum N = 1. |
|   1   | Grid size N goes to N/2 at next level, should input minimum N.      |
|   2   |    Grid size N goes to N-1 at next level, should input minimum N.   |

* **node**

| node  |             Operations             |
|:-----:|:----------------------------------:|
|   -1  | Smoothing and then do restriction. |
|   0   | Use the exact solver.              |
|   1   | Do prolongation and the smoothing. |


`Cycle.txt`:
```
(Interest region length L) (min_x) (min_y)
(con_step) (con_N)
(N) (N_min)
(node)
(node option)
(node)
(node option)
(node)
(node option)
...
```

### Run
**N_THREADS_OMP:** Number of OpenMP threads </br>
**file_name**: Cycle Structure file
```
./MG_CPU N_THREADS_OMP file_name
```

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

### Grid Addition
* void doGridAddition: Add two Grids together `U1 + U2`, and store the result inside `double *U1`.
  * Input Variable:
    * Grid size: `int N`
    * Grid 1 [1D-array address]: `double *U1`
    * Grid 2 [1D-array address]: `double *U2`

### Smoothing
* void doSmoothing: Change made inside `double *U`, and save the error from the latest smoothing in `double *error`.
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
Since the GPU is specialized in doing single precision computation, all the subroutine function of calling GPU kernal are using `single precision`. After calling GPU kernel, typecasting back to `double precision`.
Originally, I use `double precision` for `doExactSolver_GPU`, but it turns out that it's tooooo slow.

### Source (Problem Dependent)
* void getSource_GPU: Get the discretize form of the source function of size N x N. Change made inside `double *F`.
  * Input Variable:
    * Grid size: `int N`
    * Interest Region length L: `double L`
    * Source f [1D-array address]: `double *F`
    * x minimum range: `double min_x`
    * y minimum range: `double min_y`

* \_\_global\_\_ void ker_Source_GPU: Get the discretize form of the source function of size N x N. Change made inside `float *F`.
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

* \_\_global\_\_ void ker_Residual_GPU: Get the residual d = Lu - f, changes made inside `float D`
  * Input Variable:
    * Grid size: `int N`
    * delta x: `float h`
    * Approximate solution [1D-array address]: `float *U`
    * Source f [1D-array address]: `float *F`
    * Residual d [1D-array address]: `float *D`

### Smoothing
* void doSmoothing_GPU: Change made inside `double *U`, and save the error from the latest smoothing in `double *error`.
  * Input Variable:
    * Grid size: `int N`
    * Interest Region length L: `double L`
    * Approximate solution [1D-array address]: `double *U`
    * Source f [1D-array address]: `double *F`
    * Steps: `int step`
    * Error: `double *error`
  * **NOTES:**
    1. Using parallel reduction, so `threadsPerBlock = pow(2,m)`, and `threadsPerBlock * blocksPerGrid <= N*N`. 
    1. The function sets the range of the block size is 2^0 ~ 2^10, and grid size is 10^0 ~ 10^5.
    1. The selecting `threadsPerBlock` and `blocksPerGrid` method here assumes that the greater the `threadsPerBlock * blocksPerGrid` and `threadsPerBlock` is, the faster it is.
    1. The error is defined as Sum( | Lu - f | ) / (N * N) = Sum( | U - U0 | ) * 4 / ((h * h) * (N * N)).
  * **TODOs if have time:**
    - [ ] Improve the performance with sync with all threads within a grid _Cooperative Kernel_, so that we can save some time on calculating unwanted errors during the iterations.

* \_\_global\_\_ void ker_Smoothing_GPU: Change made inside `float *U` or `float *U0`, and save the error from the latest smoothing in `float *err`. Using Jacobi Method, without even / odd method. 
  * Input Variable:
    * Grid size: `int N`
    * delta x: `float h`
    * Approximate solution [1D-array address]: `float *U`
    * U_old [1D-array addreses]: `float *U0`
    * Source f [1D-array address]: `float *F`
    * Current numbers of iterations: `int iter`
    * Error array [1D-array address]: `float *err`
  * **NOTES:**
    * Changes made after `step` steps
      * If `step` is odd  :  get `d_U`
      * If `step` is even :  get `d_U0`

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
      * `option == 1`: use Gauss-Seidel, with even/odd method, GPU kernel in `double precision`.
      * `option == 2`: use Gauss-Seidel, with even/odd method, GPU kernel in `single precision`.

#### Gauss-Seidel Relaxation Method
* Double Precision
  * void GaussSeidel_GPU_Double: Change made inside exact solution `double *U`. Relax till it reaches the target error.
    * Input Variable:
      * Grid size: `int N`
      * Interest Region length L: `double L`
      * Exact solution [1D-array address]: `double *U`
      * Source Term [1D-array address]: `double *F`
      * Target error: `double target_error`

  * \_\_global\_\_ void ker_GaussSeideleven_GPU_Double: Change made inside `double *U`, update even index only.
    * Input Variable:
      * Grid size: `int N`
      * delta x: `double h`
      * Approximation solution [1D-array address]: `double *U`
      * Souce f [1D-array address]: `double *F`

  * \_\_global\_\_ void ker_GaussSeidelodd_GPU_Double: Change made inside `double *U`, update odd index only.
    * Input Variable:
      * Grid size: `int N`
      * delta x: `double h`
      * Approximation solution [1D-array address]: `double *U`
      * Souce f [1D-array address]: `double *F`

  * \_\_global\_\_ void ker_Error_GPU_Double: Get the error of U, define as Sum( | Lu - F | ) / (N * N).
    * Input Variable:
      * Grid size: `int N`
      * delta x: `double h`
      * Approximation solution [1D-array address]: `double *U`
      * Souce f [1D-array address]: `double *F`
      * Error array [1D-array address]: `double *err`

* Single Precision
  * void GaussSeidel_GPU_Single: Change made inside exact solution `double *U`. Relax till it reaches the target error.
    * Input Variable:
      * Grid size: `int N`
      * Interest Region length L: `double L`
      * Exact solution [1D-array address]: `double *U`
      * Source Term [1D-array address]: `double *F`
      * Target error: `double target_error`
  
  * \_\_global\_\_ void ker_GaussSeideleven_GPU_Single: Change made inside `float *U`, update even index only.
    * Input Variable:
      * Grid size: `int N`
      * delta x: `float h`
      * Approximation solution [1D-array address]: `float *U`
      * Souce f [1D-array address]: `float *F`

  * \_\_global\_\_ void ker_GaussSeidelodd_GPU_Single: Change made inside `float *U`, update odd index only.
    * Input Variable:
      * Grid size: `int N`
      * delta x: `float h`
      * Approximation solution [1D-array address]: `float *U`
      * Souce f [1D-array address]: `float *F`

  * \_\_global\_\_ void ker_Error_GPU_Single: Get the error of U, define as Sum( | Lu - F | ) / (N * N).
    * Input Variable:
      * Grid size: `int N`
      * delta x: `float h`
      * Approximation solution [1D-array address]: `float *U`
      * Souce f [1D-array address]: `float *F`
      * Error array [1D-array address]: `float *err`  

* **NOTES:**
  1. Write two kernel, one deals with even index, the other deals with odd index, so that we can make sure that updates on even/odd are all done.

* **TODOs if have time:**
  - [ ] Try using sync with all threads within a grid _Cooperative Kernel_, which means forge two kernels together.



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

* \_\_global\_\_ void ker_Zoom_GPU: Zoom in/out from grid size `int N` to grid size `int M`. The final result is inside `float *U_m`.
  * Input Variable:
    * Grid size before zooming: `int N`
    * Spacing between points before zooming: `float h_n`
    * Spacing between points after zooming: `float h_m`
    * Grid size after zooming: `int M`
    * Result of the grid: `float *U_m`
  * **NOTES:**
    1. The grid before zooming is binded to the texture memory, so no need to pass in the pointer to the kernel.

* **NOTES:**
  1. Restriction is specific on residual, and since we only don't do relaxation on the boundary, so the boundary of restriction target grid is always "0".
  1. The boundary of prolongation target grid is "0".

