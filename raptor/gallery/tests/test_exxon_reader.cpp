#include <assert.h>

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/exxon_reader.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Create A from diffusion stencil
    char* folder = "/home/bienz2/exxonmobildata/SPE10-4x4-20141227/matrix_blk_coord";
    char* iname = "index_R";
    char* fname = "matrix_blk_coord_TS414_TSA0_NI0_FT0.010000_R";
    char* suffix = ".bcoord";

    int* global_num_rows;

    ParCSRMatrix* A = exxon_reader(folder, iname, fname, suffix, &global_num_rows);
    ParVector x = ParVector(A->global_num_cols, A->local_num_cols, A->first_local_col);
    ParVector b = ParVector(A->global_num_rows, A->local_num_rows, A->first_local_row);
    x.set_const_value(1.0);
    
    A->mult(x, b);
    double bnorm = b.norm(2);
    if (rank == 0) printf("Bnorm = %e\n", bnorm);

    delete A;

    MPI_Finalize();
}   

