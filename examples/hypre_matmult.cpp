#include <mpi.h>
#include "timer.hpp"
#include <math.h>
#include "core/types.hpp"
#include "util/linalg/spmv.hpp"
#include "util/linalg/matmult.hpp"
#include "gallery/external/mfem_wrapper.hpp"
#include "gallery/external/hypre_wrapper.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"
#include "gallery/matrix_IO.hpp"
#include "topo_aware/topo_comm.hpp"
#include "topo_aware/topo_spmv.hpp"
#include "topo_aware/topo_matmult.hpp"
#include "topo_aware/topo_wrapper.hpp"
#include "clear_cache.hpp"
#include <unistd.h>

#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "_hypre_parcsr_mv.h"

using namespace raptor;

void compare_results(ParMatrix* S, hypre_ParCSRMatrix* Scsr)
{
    data_t dense[S->global_rows][S->global_cols] = {0};
    for (int j = 0; j < S->global_rows; j++)
        for (int k = 0; k < S->global_cols; k++)
            dense[j][k] = 0;

    int nrows = hypre_ParCSRMatrixNumRows(Scsr);
    hypre_CSRMatrix* diag = hypre_ParCSRMatrixDiag(Scsr);
    hypre_CSRMatrix* offd = hypre_ParCSRMatrixOffd(Scsr);
    int* diag_I = hypre_CSRMatrixI(diag);
    int* diag_J = hypre_CSRMatrixJ(diag);
    double* diag_data = hypre_CSRMatrixData(diag);
    int* offd_I = hypre_CSRMatrixI(offd);
    int* offd_J = hypre_CSRMatrixJ(offd);
    double* offd_data = hypre_CSRMatrixData(offd);
    int first_row = hypre_ParCSRMatrixFirstRowIndex(Scsr);
    int first_col_diag = hypre_ParCSRMatrixFirstColDiag(Scsr);
    int* col_map_offd = hypre_ParCSRMatrixColMapOffd(Scsr);

    int row_start, row_end;
    int global_row, global_col;
    double value;
    for (int j = 0; j < nrows; j++)
    {
        row_start = diag_I[j];
        row_end = diag_I[j+1];
        for (int k = row_start; k < row_end; k++)
        {
            dense[j+first_row][diag_J[k] + first_col_diag] = diag_data[k];
        }
        row_start = offd_I[j];
        row_end = offd_I[j+1];
        for (int k = row_start; k < row_end; k++)
        {
            dense[j+first_row][col_map_offd[offd_J[k]]] = offd_data[k];
        }
    }

    if (S->offd_num_cols)
        S->offd->convert(CSR);
    for (int row = 0; row < S->local_rows; row++)
    {
        row_start = S->diag->indptr[row];
        row_end = S->diag->indptr[row+1];
        global_row = row + S->first_row;

        for (int j = row_start; j < row_end; j++)
        {
            global_col = S->diag->indices[j] + S->first_col_diag;
            value = S->diag->data[j];
            //assert(fabs(value - dense[global_row][global_col]) < 1e-06);
            if (fabs(value - dense[global_row][global_col] > 1e-06))
                printf("Value = %e, Dense[%d][%d] = %e\n", value, global_row,
                    global_col, dense[global_row][global_col]);
        }
        if (S->offd_num_cols)
        {
            row_start = S->offd->indptr[row];
            row_end = S->offd->indptr[row+1];
            for (int j = row_start; j < row_end; j++)
            {
                global_col = S->local_to_global[S->offd->indices[j]];
                value = S->offd->data[j];
                //assert(fabs(value - dense[global_row][global_col]) < 1e-06);
                if (fabs(value - dense[global_row][global_col] > 1e-06))
                    printf("Value = %e, Dense[%d][%d] = %e\n", value, global_row,
                        global_col, dense[global_row][global_col]);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get Local Process Rank, Number of Processes
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    TopoManager* tm = new TopoManager();

    // Get Command Line Argument (For type of system to solve)
    // Iso: 0, Aniso: 1, IO: 2, MFEM: 3
    int system = 0;
    if (argc > 1)
    {
        system = atoi(argv[1]);
    }

    // Variables for AMG hierarchy (raptor)
    ParMatrix* A_rap;
    ParVector* x_rap;
    ParVector* b_rap;
    Hierarchy* ml;
    Level* level;
    ParMatrix* A_l;
    ParMatrix* P_l;
    ParMatrix* S_l;
    ParMatrix* S_l_topo;
    int num_levels;

    // Variables for AMG hierarchy (hypre)
    HYPRE_IJMatrix Aij;
    HYPRE_IJVector xij;
    HYPRE_IJVector bij;
    hypre_ParCSRMatrix* A_par;
    hypre_ParVector* x_par;
    hypre_ParVector* b_par;
    HYPRE_Solver amg_data;    

    // Variables for Iso/Aniso Problems
    int n;
    int dim;
    int* grid;
    data_t* stencil;
 
    data_t max_diff = 0.0;
    data_t total_max_diff;

    // Variable for IO 
    char* filename;

    // Variables for MFEM
    char* mesh;
    int num_elements;
    int order;
    int mfem_choice;

    // Timing variables
    double t0, t1;
    double torig, tTAP, thypre;
    int num_tests = 10;

    // Variables to clear cache between tests
    int cache_len = 10000;
    double* cache_array = new double[cache_len];

    // Create System to be Solved
    if (system < 3)
    {
        if (system == 0)
        {
            n = 435;
            if (argc > 2) n = atoi(argv[2]);
            grid = new int[3];
            grid[0] = n;
            grid[1] = n;
            grid[2] = n;
            dim = 3;
            stencil = laplace_stencil_27pt();
        }
        else if (system == 1)
        {
            n = 9051;
            if (argc > 2) n = atoi(argv[2]);
            grid = new int[2];
            grid[0] = n;
            grid[1] = n;
            dim = 2;
            stencil = diffusion_stencil_2d(0.1, 0.0);
        }

        if (system < 2)
        {
            A_rap = stencil_grid(stencil, grid, dim);
            delete[] stencil;
            delete[] grid;
        }
        else
        {
            filename = argv[2];
            int sym = 1;
            if (argc > 3) sym = atoi(argv[3]);
            A_rap = readParMatrix(filename, MPI_COMM_WORLD, 1, sym);
        }

        b_rap = new ParVector(A_rap->global_rows, A_rap->local_rows, A_rap->first_row);
        x_rap = new ParVector(A_rap->global_cols, A_rap->local_cols, A_rap->first_col_diag);
        b_rap->set_const_value(0.0);
        x_rap->set_const_value(1.0);
    }
    else
    {
        mesh = "/u/sciteam/bienz/mfem/data/beam-tet.mesh";
        //mesh = "/Users/abienz/Documents/Parallel/mfem/data/beam-tet.mesh";

        mfem_choice = 0;
        num_elements = 0;
        order = 0;
        if (argc > 2)
        {
            mfem_choice = atoi(argv[2]);
            if (argc > 3)
            {
                num_elements = atoi(argv[3]);
                if (argc > 4)
                    order = atoi(argv[4]);
            }
        }
        if (mfem_choice == 0)
        {
            if (num_elements == 0) num_elements = 3;
            if (order == 0) order = 3;
            mfem_linear_elasticity(&A_rap, &x_rap, &b_rap, mesh, num_elements, order);
        }
        else if (mfem_choice == 1)
        {
            if (num_elements == 0) num_elements = 50000;
            if (order == 0) order = 5;
            mfem_laplace(&A_rap, &x_rap, &b_rap, mesh, num_elements, order);
        }
    }

    // Create AMG Hierarchy
    Aij = convert(A_rap);
    HYPRE_IJMatrixGetObject(Aij, (void**) &A_par);
    xij = convert(x_rap);
    HYPRE_IJVectorGetObject(xij, (void**) &x_par);
    bij = convert(b_rap);
    HYPRE_IJVectorGetObject(bij, (void**) &b_par);
    if (system == 0)
    {
        amg_data = hypre_create_hierarchy(A_par, x_par, b_par, 10, 6, 0, 1, 0.35);
    }
    else
    {
        amg_data = hypre_create_hierarchy(A_par, x_par, b_par);
    }
    ml = convert((hypre_ParAMGData*) amg_data);

    hypre_ParCSRMatrix** A_array = hypre_ParAMGDataAArray((hypre_ParAMGData*) amg_data);
    hypre_ParCSRMatrix** P_array = hypre_ParAMGDataPArray((hypre_ParAMGData*) amg_data);

    // Initialize Variables
    num_levels = ml->num_levels;
    MPI_Barrier(MPI_COMM_WORLD);

    hypre_ParCSRMatrix* Scsr_l;

    MPI_Datatype custom_type;
    create_coo_type(&custom_type);
    MPI_Type_commit(&custom_type);

    TopoComm* tc;

    for (int i = 0; i < ml->num_levels-1; i++)
    {
        level = ml->levels[i];
        A_l = level->A;
        P_l = level->P;
        
        if (A_l->offd_num_cols) A_l->offd->convert(CSR);
        if (P_l->offd_num_cols) P_l->offd->convert(CSR);

        tc = new TopoComm(tm, A_l->first_row, A_l->first_col_diag,
                A_l->local_to_global.data(), A_l->global_col_starts.data(),
                A_l->offd_num_cols);

        parallel_matmult(A_l, P_l, &S_l, custom_type);
        parallel_matmult_topo(A_l, P_l, &S_l_topo, custom_type, tc);
        Scsr_l = hypre_ParMatmul(A_array[i], P_array[i]);

        compare_results(S_l, Scsr_l);
        compare_results(S_l_topo, Scsr_l);

        delete S_l;
        delete S_l_topo;
        hypre_ParCSRMatrixDestroy(Scsr_l);

        for (int test = 0; test < 5; test++)
        {
            if (rank == 0) printf("Test %d\n", test);

            tTAP = 0.0;
            torig = 0.0;
            thypre = 0.0;

            // Test HYPRE Matmult 
            MPI_Barrier(MPI_COMM_WORLD);
            for (int j = 0; j < num_tests; j++)
            {
                get_ctime(t0);
                parallel_matmult(A_l, P_l, &S_l, custom_type);
                get_ctime(t1);
                torig += (t1 - t0);
                delete S_l;
                clear_cache(cache_len, cache_array);
            }
            torig /= num_tests;

            // Test HYPRE Matmult 
            MPI_Barrier(MPI_COMM_WORLD);
            for (int j = 0; j < num_tests; j++)
            {
                get_ctime(t0);
                parallel_matmult_topo(A_l, P_l, &S_l_topo, custom_type, tc);
                get_ctime(t1);
                tTAP += (t1 - t0);
                delete S_l_topo;
                clear_cache(cache_len, cache_array);
            }
            tTAP /= num_tests;

            // Test Simple Matmult 
            MPI_Barrier(MPI_COMM_WORLD);
            for (int j = 0; j < num_tests; j++)
            {
                get_ctime(t0);
                Scsr_l = hypre_ParMatmul(A_array[i], P_array[i]);
                get_ctime(t1);
                thypre += (t1 - t0);
                hypre_ParCSRMatrixDestroy(Scsr_l);
                clear_cache(cache_len, cache_array);
            }
            thypre /= num_tests;

            // Print Timing Information
            MPI_Reduce(&torig, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Level %d Time Original = %2.5e\n", i, t0);
            MPI_Reduce(&tTAP, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Level %d Time TAP = %2.5e\n", i, t0);
            MPI_Reduce(&thypre, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Level %d Time HYPRE = %2.5e\n", i, t0);
        }
        delete tc;
    }
    // Clean up
    MPI_Type_free(&custom_type);

    delete ml;

    hypre_BoomerAMGDestroy(amg_data);     
    HYPRE_IJMatrixDestroy(Aij);
    HYPRE_IJVectorDestroy(xij);
    HYPRE_IJVectorDestroy(bij);

    delete A_rap;
    delete x_rap;
    delete b_rap;

    delete[] cache_array;

    delete tm;

    MPI_Finalize();

    return 0;
}


