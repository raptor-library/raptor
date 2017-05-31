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
    char* folder = "/Users/abienz/Documents/Parallel/exxon/mat_32";
    char* fname = "matrix_blk_coord_TS880_TSA0_NI1_FT0.010000_R";
    char* iname = "index_R";
    char* suffix = ".bcoord";
    char* fname_mat = "/Users/abienz/Documents/Parallel/exxon/mat_32/exxonmat32_reordered.mtx";
    char* fname_rows = "/Users/abienz/Documents/Parallel/exxon/mat_32/exxonmat32_rows.txt";

    int* global_num_rows;

    ParCSRMatrix* A = exxon_reader(folder, iname, fname, suffix, &global_num_rows);
    ParVector x = ParVector(A->global_num_cols, A->local_num_cols, A->first_local_col);
    ParVector b = ParVector(A->global_num_rows, A->local_num_rows, A->first_local_row);
    x.set_const_value(1.0);

/*    int local_num_rows;
    int first_local_row = 0;
    FILE* row_file = fopen(fname_rows, "r");
    for (int i = 0; i < rank; i++)
    {
        fscanf(row_file, "%d\n", &local_num_rows);
        first_local_row += local_num_rows;
    }
    fscanf(row_file, "%d\n", &local_num_rows);
    fclose(row_file);
    ParCSRMatrix* Amtx = readParMatrix(fname_mat, MPI_COMM_WORLD, 1, 0, local_num_rows,
            local_num_rows, first_local_row, first_local_row);

    A->mult(x, b);
    double bnorm = b.norm(2);
    if (rank == 0) printf("Bnorm = %e\n", bnorm);
    printf("A->comm->num_sends = %d\n", A->comm->send_data->num_msgs);

    assert(A->local_num_rows == Amtx->local_num_rows);
    assert(A->local_num_cols == Amtx->local_num_cols);
    assert(A->local_nnz == Amtx->local_nnz);
    assert(A->comm->send_data->num_msgs == Amtx->comm->send_data->num_msgs);
    assert(A->comm->recv_data->num_msgs == Amtx->comm->recv_data->num_msgs);
    assert(A->comm->send_data->size_msgs == Amtx->comm->send_data->size_msgs);
    assert(A->comm->recv_data->size_msgs == Amtx->comm->recv_data->size_msgs);
*/
    delete A;
//    delete Amtx;

    MPI_Finalize();
}   

