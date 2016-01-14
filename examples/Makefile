CXX = mpicxx
RAPTOR_PATH = $(HOME)/Documents/Parallel

CXX_FLAGS = -O2 -std=c++11

RAPTOR_FLAGS = -I../raptor/
RAPTOR_LIBS = -L../build/lib -lraptor

HYPRE_FLAGS = -I../../hypre/src/hypre/include
HYPRE_LIBS = -L../../hypre/src/hypre/lib -lHYPRE\
	-Wall -llapack -lblas

MFEM_FLAGS = -I../../mfem\
	-I../../metis/include

MFEM_LIBS = -L../../mfem -lmfem\
	-L../../metis/build/Linux-x86_64/libmetis/ -lmetis

all : benchmark.cpp
	$(CXX) $(CXX_FLAGS) $(RAPTOR_FLAGS) benchmark.cpp $(RAPTOR_LIBS) -o benchmark

hypre: benchmark_hypre.cpp
	$(CXX) $(CXX_FLAGS) $(RAPTOR_FLAGS) $(MFEM_FLAGS) $(HYPRE_FLAGS) benchmark_hypre.cpp $(RAPTOR_LIBS) $(MFEM_LIBS) $(HYPRE_LIBS) -o benchmark_hypre

mfem: benchmark_mfem.cpp
	$(CXX) $(CXX_FLAGS) $(RAPTOR_FLAGS) $(MFEM_FLAGS) $(HYPRE_FLAGS) benchmark_mfem.cpp $(RAPTOR_LIBS) $(MFEM_LIBS) $(HYPRE_LIBS) -o benchmark_mfem

clean:
	rm benchmark
	rm benchmark_hypre
	rm benchmark_mfem



