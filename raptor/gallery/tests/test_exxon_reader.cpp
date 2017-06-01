#include <assert.h>

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/exxon_reader.hpp"
#include "gallery/matrix_IO.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Create A from diffusion stencil
    char* folder = "/home/bienz2/verification-twomatrices/DA_V5_blk_coord_binary-3x3-blk-np16";
    char* fname = "DA_V5_blk_coord_binary-3x3-blk-np16_TS6_TSA0_NI0_R";
    char* iname = "index_R";
    char* suffix = ".bcoord_bin";
    char* suffix_x = ".sol_bin";
    char* suffix_b = ".rhs_bin";

    int* global_num_rows;

    ParCSRMatrix* A = exxon_reader(folder, iname, fname, suffix, &global_num_rows);
    ParVector x;
    ParVector b;
    exxon_vector_reader(folder, fname, suffix_x, x);
    exxon_vector_reader(folder, fname, suffix_b, b);

    ParVector b_rap = ParVector(A->global_num_rows, A->local_num_rows, A->first_local_row);
    A->mult(x, b_rap);

    for (int i = 0; i < A->local_num_rows; i++)
    {
        assert(fabs(b.local[i] - b_rap.local[i]) < 1e-08);
    }

    delete A;

    MPI_Finalize();
}   

