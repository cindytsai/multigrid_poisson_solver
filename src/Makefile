BIN1        := MG_CPU
BIN2        := MG_GPU
SMVERSION   := 75

all: $(BIN1) $(BIN2)

$(BIN1): MG_solver_CPU.cpp linkedlist.h linkedlist.cpp
	g++ -fopenmp -o $(BIN1) MG_solver_CPU.cpp linkedlist.cpp

$(BIN2): MG_solver_GPU.cu linkedlist.h linkedlist.cpp
	nvcc -arch=compute_$(SMVERSION) -code=sm_$(SMVERSION) -O3 --compiler-options -fopenmp -o $(BIN2) MG_solver_GPU.cu linkedlist.cpp

clean: 
	rm -f $(BIN1) $(BIN2)