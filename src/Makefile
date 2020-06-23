BIN1	:= MG_CPU
BIN2	:= MG_GPU

all: $(BIN1) $(BIN2)

$(BIN1): MG_solver_CPU.cpp LinkList.h
	g++ -fopenmp -o $(BIN1) MG_solver_CPU.cpp

$(BIN2): MG_solver_GPU.cu LinkList.h
	nvcc -arch=compute_52 -code=sm_52 -O3 --compiler-options -fopenmp -o $(BIN2) MG_solver_GPU.cu

clean: 
	rm -f $(BIN1) $(BIN2)