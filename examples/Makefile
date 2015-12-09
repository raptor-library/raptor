CXX = mpicxx
RAPTOR_PATH = $(HOME)/Documents/Parallel

CXX_FLAGS = -O2 -std=c++11\
	-I/$(RAPTOR_PATH)/raptor/external/\
	-I/$(RAPTOR_PATH)/raptor/raptor/\
    -I/home/bienz2/Documents/Parallel/hypre/src/hypre/include

CXX_LIBS = -L/$(RAPTOR_PATH)/raptor/build/raptor\
	-lraptor\
    -L/home/bienz2/Documents/Parallel/hypre/src/hypre/lib\
    -lHYPRE

all : benchmark.cpp
	$(CXX) $(CXX_FLAGS) benchmark.cpp $(CXX_LIBS) -o benchmark.out



