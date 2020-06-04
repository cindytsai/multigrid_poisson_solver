MG: MG_solver_CPU.o
	g++-9 -o MG MG_solver_CPU.o
MG_solver_CPU.o: MG_solver_CPU.cpp LinkList.h
	g++-9 -c MG_solver_CPU.cpp

all: MG

clean: 
	rm -f MG_solver_CPU.o MG