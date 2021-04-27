cuda_tests: cudaSolver.o
	g++ -o cuda_tests src/cuda_tests.cpp src/*.c cudaSolver.o -L/usr/local/depot/cuda-10.2/lib64 -I/usr/local/depot/cuda-10.2/include -I"include" -lcusparse -lcudart -O3 -Wextra -std=c++11
gpu_solver: cudaSolver.o
	g++ -o gpu_solver src/gpu_solver.cpp src/*.c cudaSolver.o -L/usr/local/depot/cuda-10.2/lib64 -I/usr/local/depot/cuda-10.2/include -I"include" -lcusparse -lcudart -O3 -Wextra -std=c++11
cudaSolver.o: src/cudaSolver.cu
	nvcc -std=c++11 -c -arch=sm_30 src/cudaSolver.cu -I"include" -lcusparse
