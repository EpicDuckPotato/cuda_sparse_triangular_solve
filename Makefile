CXX = g++
CXXFLAGS = -O3 -Wall -I"include"
DEPS = *.h


%.o:%.cpp %.c $(DEPS)

$(EXECUTABLE):$(OBJECTS)
	$(CXX) -o $(EXECUTABLE) $(OBJECTS) $(CXXFLAGS) $(LDFLAGS)   

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)	
