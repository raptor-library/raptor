#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "core/par_matrix.hpp"
#include <mpi.h>

using namespace raptor;

int main(int argc, char* argv[]){
	MPI_Init(&argc, &argv);
	
	int rank, num_procs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);


	//Initialize ParBSRMatrix
    std::vector<std::vector<double>> on_blocks = {{1,0,2,1}, {6,7,8,2}, {1,4,5,1},
	    					{4,3,0,0}, {7,2,0,0}};
    std::vector<std::vector<double>> off_blocks = {{1,0,0,1}, {2,0,0,0}, {3,0,1,0}};
    std::vector<std::vector<int>> on_indx = {{0,0}, {0,1}, {1,1}, {2,1}, {2,2}};
    std::vector<std::vector<int>> off_indx = {{0,4}, {1,3}, {2,5}};

    ParBSRMatrix* A = new ParBSRMatrix(12, 12, 2, 2);

    for(int i=0; i<on_blocks.size(); i++){
        A->add_block(on_indx[i][0], on_indx[i][1], on_blocks[i]);
        A->add_block(on_indx[i][0]+3, on_indx[i][1]+3, on_blocks[i]);
    }

    for(int i=0; i<off_blocks.size(); i++){
        A->add_block(off_indx[i][0], off_indx[i][1], off_blocks[i]);
        A->add_block(off_indx[i][0]+3, off_indx[i][1]-3, off_blocks[i]);
    }

    // Finalize test matrix
    A->finalize(true, 2);

    // Vectors for Multiplication
    //ParVector x(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);
    //ParVector b(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    
    // Test SpMV
    //x.set_const_value(1.0);
    //A->mult(x, b);
   	int lof = 0;
    
	A->on_proc->print();
	Matrix* factors = A->on_proc->ilu_k(lof);

	printf("Rank = %d\n",rank);
	factors ->print();

    // Delete A
    delete A;


	MPI_Finalize();
}
