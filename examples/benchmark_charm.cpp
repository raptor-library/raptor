#include <mpi.h>
#include <math.h>
#include "core/types.hpp"
#include "util/linalg/spmv.hpp"
#include "gallery/external/mfem_wrapper.hpp"
#include "gallery/external/hypre_wrapper.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"
#include "hypre_async.h"

#include <string>
#include <sstream>

using namespace raptor;

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

_TRACE_END();

    // Get Local Process Rank, Number of Processes
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Get Command Line Arguments (Must Have 5)
    // TODO -- Fix how we parse command line
    int num_tests = 10;
    int num_elements = 10;
    if (argc > 1)
    {
        num_tests = atoi(argv[1]);
        if (argc > 2)
        {
            num_elements = atoi(argv[2]);
        }
    }

    // Declare Variables
    ParMatrix* A;
    ParVector* x;
    ParVector* b;
    Hierarchy* ml;
    ParMatrix* A_l;
    ParVector* x_l;
    ParVector* b_l;

    long local_nnz;
    long global_nnz;
    index_t num_levels;
    index_t len_b, len_x;
    index_t local_rows;
    data_t b_norm;
    data_t t0, tfinal;
    data_t* b_data;
    data_t* x_data;

    //Initialize variable for clearing cache between tests
    index_t cache_size = 10000;
    data_t* cache_list = new data_t[cache_size];

    // Get matrix and vectors from MFEM
    //mfem_laplace(&A, &x, &b, mesh, num_elements, order);
    int dim = 3;
    int grid[dim] = {num_elements, num_elements, num_elements};
    data_t* sten = laplace_stencil_27pt();
    A = stencil_grid(sten, grid, dim);
    delete[] sten;
    b = new ParVector(A->global_cols, A->local_cols, A->first_col_diag);
    x = new ParVector(A->global_rows, A->local_rows, A->first_row);
    x->set_const_value(1.0);

    // Calculate and Print Number of Nonzeros in Matrix
    local_nnz = 0;
    if (A->local_rows)
    {
        local_nnz = A->diag->nnz + A->offd->nnz;
    }
    global_nnz = 0;
    MPI_Reduce(&local_nnz, &global_nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Nonzeros = %lu\n", global_nnz);

    t0, tfinal;

    // Create hypre (amg_data) and raptor (ml) hierarchies (they share data)
    ml = create_wrapped_hierarchy(A, x, b);

    num_levels = ml->num_levels;

_TRACE_BEGIN();

    int ids[num_levels][8];

    int init_id = rank*200;

    for (int i = 0; i < num_levels; i++)
    {
        std::ostringstream oss;
        oss << "SpMV" << i;
        std::string spmv_string = oss.str();
        char const* spmv_name = spmv_string.c_str();
        int spmv_id = _TRACE_REGISTER_FUNCTION_ID((char*) spmv_name, init_id + (i*8) + 1);
        ids[i][0] = spmv_id;

        std::ostringstream diag_oss;
        diag_oss << "DiagSpMV" << i;
        std::string diag_spmv_string = diag_oss.str();
        char const* diag_spmv_name = diag_spmv_string.c_str();
        int diag_spmv_id = _TRACE_REGISTER_FUNCTION_ID((char*) diag_spmv_name, init_id + (i*8) + 2);
        ids[i][1] = diag_spmv_id;

        std::ostringstream offd_oss;
        offd_oss << "OffdSpMV" << i;
        std::string offd_spmv_string = offd_oss.str();
        char const* offd_spmv_name = offd_spmv_string.c_str();
        int offd_spmv_id = _TRACE_REGISTER_FUNCTION_ID((char*) offd_spmv_name, init_id + (i*8) + 3);
        ids[i][2] = offd_spmv_id;

        std::ostringstream waitany_oss;
        waitany_oss << "WaitAny (Recv) " << i;
        std::string waitany_string = waitany_oss.str();
        char const* waitany_name = waitany_string.c_str();
        int waitany_id = _TRACE_REGISTER_FUNCTION_ID((char*) waitany_name, init_id + (i*8) + 4);
        ids[i][3] = waitany_id;

        std::ostringstream waitall_oss;
        waitall_oss << "WaitAll (Recv) " << i;
        std::string waitall_string = waitall_oss.str();
        char const* waitall_name = waitall_string.c_str();
        int waitall_id = _TRACE_REGISTER_FUNCTION_ID((char*) waitall_name, init_id + (i*8) + 5);
        ids[i][4] = waitall_id;

        std::ostringstream waitall2_oss;
        waitall2_oss << "WaitAll (Send) " << i;
        std::string waitall2_string = waitall2_oss.str();
        char const* waitall2_name = waitall2_string.c_str();
        int waitall2_id = _TRACE_REGISTER_FUNCTION_ID((char*) waitall2_name, init_id + (i*8) + 6);
        ids[i][5] = waitall2_id;

        std::ostringstream init_recv_oss;
        init_recv_oss << "Init Recv " << i;
        std::string init_recv_string = init_recv_oss.str();
        char const* init_recv_name = init_recv_string.c_str();
        int init_recv_id = _TRACE_REGISTER_FUNCTION_ID((char*) init_recv_name, init_id + (i*8) + 7);
        ids[i][6] = init_recv_id;

        std::ostringstream init_send_oss;
        init_send_oss << "Init Send " << i;
        std::string init_send_string = init_send_oss.str();
        char const* init_send_name = init_send_string.c_str();
        int init_send_id = _TRACE_REGISTER_FUNCTION_ID((char*) init_send_name, init_id + (i*8) + 8);
        ids[i][7] = init_send_id;
    }

    ml->x_list[0] = x;
    ml->b_list[0] = b;

    for (int i = 0; i < num_levels; i++)
    {
        A_l = ml->A_list[i];
        x_l = ml->x_list[i];
        b_l = ml->b_list[i];

        // Test CSC Synchronous SpMV
        _TRACE_BEGIN_FUNCTION_ID(ids[i][0]);
        for (int j = 0; j < num_tests; j++)
        {
            parallel_spmv(A_l, x_l, b_l, 1.0, 0.0, 0, ids[i]);
        }
        _TRACE_END_FUNCTION_ID(ids[i][0]);

    }

    delete ml;

    delete A;
    delete x;
    delete b;

    delete[] cache_list;

    MPI_Finalize();

    return 0;
}



