# Multigrid Poisson Solver

## Structure
phi = ...
N = ..
cycle = 1D-array = {-1 , -1, 0, 1, 1}

### V-cycle / W-cycle
for i in cycle:
    if(cycle[i] == -1): doRestriction
    if(cycle[i] ==  0): doExactSolver
    if(cycle[i] ==  1): doProlongation

### FMG
while():
    triggers:

## Functions
All of the grids are stored as 1D array, with size N x N, including the boundary.

### Smoothing
* void doSmoothing: Change made inside `double *phi`
* Input Variable
  * Grid size: `int N`
  * 1D-array address: `double *phi`

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