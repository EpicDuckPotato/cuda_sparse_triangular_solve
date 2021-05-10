.PHONY: cuda_tests gpu_solver cpu_solver

all:
	make cuda_tests
	make gpu_solver
	make cpu_solver

cuda_tests: cudaSolver.o
	g++ -w -o cuda_tests src/cuda_tests.cpp src/*.c cudaSolver.o -L/usr/local/depot/cuda-10.2/lib64 -I/usr/local/depot/cuda-10.2/include -I"include" -lcusparse -lcudart -O3 -Wextra -std=c++11
gpu_solver: cudaSolver.o
	g++ -w -o gpu_solver src/gpu_solver.cpp src/*.c cudaSolver.o -L/usr/local/depot/cuda-10.2/lib64 -I/usr/local/depot/cuda-10.2/include -I"include" -lcusparse -lcudart -O3 -Wextra -std=c++11
cpu_solver:
	g++ -w -o cpu_solver src/cpu_solver.cpp src/*.c -I"include" -O3 -Wextra -std=c++11
cudaSolver.o: src/cudaSolver.cu
	nvcc -std=c++11 -c -arch=sm_30 src/cudaSolver.cu -I"include" -lcusparse
clean:
	rm cuda_tests gpu_solver cpu_solver cudaSolver.o
