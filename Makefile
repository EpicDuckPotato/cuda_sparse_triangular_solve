CXX = g++
CXXFLAGS = -O3 -Wall -I"/home/noopygbhat/SuiteSparse-5.9.0/include" -I"include"
LDFLAGS = -L../SuiteSparse/lib/ -lcholmod -lamd -lcolamd -lcamd -lccolamd -lmetis $(LAPACK) $(BLAS)

OBJECTS = *.o
EXECUTABLE = lt_solve

all:$(EXECUTABLE)

*.o:*.cpp

$(EXECUTABLE):$(OBJECTS)
	$(CXX) -o $(EXECUTABLE) $(OBJECTS) $(CXXFLAGS) $(LDFLAGS)   

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)	


	rm *.txt


	


 	
