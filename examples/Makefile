CXX = CC
RAPTOR_PATH = $(HOME)

CXX_FLAGS = -O2 -std=c++11\
	-I/$(RAPTOR_PATH)/raptor/external/\
	-I/$(RAPTOR_PATH)/raptor/raptor/\

CXX_LIBS = -L/$(RAPTOR_PATH)/raptor/build/raptor\
	-lraptor

all : benchmark.cpp
	tau_cxx.sh $(CXX_FLAGS) benchmark.cpp $(CXX_LIBS) -o benchmark.out
	#$(CXX) $(CXX_FLAGS) benchmark.cpp $(CXX_LIBS) -o benchmark.out


