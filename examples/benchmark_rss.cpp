#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "clear_cache.hpp"

#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "core/types.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/external/hypre_wrapper.hpp"
//#include "gallery/external/mfem_wrapper.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "multilevel/multilevel.hpp"

void compare(hypre_ParCSRMatrix* A_h, ParCSRMatrix* A, bool has_data = true, 
        bool hyp_diag = true)
{
    int start, end, size;

    hypre_CSRMatrix* A_h_diag = hypre_ParCSRMatrixDiag(A_h);
    hypre_CSRMatrix* A_h_offd = hypre_ParCSRMatrixOffd(A_h);
    int A_h_nrows = hypre_CSRMatrixNumRows(A_h_diag);
    int* A_h_diag_i = hypre_CSRMatrixI(A_h_diag);
    int* A_h_diag_j = hypre_CSRMatrixJ(A_h_diag);
    double* A_h_diag_data = hypre_CSRMatrixData(A_h_diag);
    int* A_h_offd_i = hypre_CSRMatrixI(A_h_offd);
    int* A_h_offd_j = hypre_CSRMatrixJ(A_h_offd);
    double* A_h_offd_data = hypre_CSRMatrixData(A_h_offd);

    start = A->on_proc->idx1[0];
    end = A->on_proc->idx1[1];
    if (A->on_proc->idx2[start] == 0) start++;
    size = end - start;
    printf("A->size[0] = %d, A_h %d\n", size, A_h_diag_i[1] - A_h_diag_i[0]);
    for (int i = start; i < end; i++)
    {
        printf("Col[%d] = %d\n", i, A->on_proc->idx2[i]);
    }
    for (int i = A_h_diag_i[0]; i < A_h_diag_i[1]; i++)
    {
        printf("COL H [%d] = %d\n", i, A_h_diag_j[i]);
    }

    printf("HypreMax[10, -12] = %d\n", hypre_max(10, -12));

/*    assert(A_h_nrows == A->local_num_rows);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        start = A->on_proc->idx1[i];
        end = A->on_proc->idx1[i+1];
        if (!hyp_diag && A->on_proc->idx2[start] == i) start++;
        size = end - start;
        assert(A_h_diag_i[i+1] - A_h_diag_i[i] == size);

        hypre_qsort1(A_h_diag_j, A_h_diag_data, start+1, end-1);
        for (int j = 0; j < size; j++)
        {
            assert(A->on_proc->idx2[start + j] == A_h_diag_j[A_h_diag_i[i] + j]);
            if (has_data) 
            {
                assert(fabs(A->on_proc->vals[start + j] - 
                            A_h_diag_data[A_h_diag_i[i] + j]) < 1e-05);
            }
        }

        start = A->off_proc->idx1[i];
        end = A->off_proc->idx1[i+1];
        assert(A->off_proc->idx1[i+1] == A_h_offd_i[i+1]);
        
        hypre_qsort1(A_h_offd_j, A_h_offd_data, start, end - 1);
        for (int j = start; j < end; j++)
        {
            assert(A->off_proc->idx2[j] == A_h_offd_j[j]);
            if (has_data) assert(fabs(A->off_proc->vals[j] - A_h_offd_data[j]) < 1e-05);
        }
    }
    */
}

//using namespace raptor;
int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    int dim;
    int n = 5;
    int system = 0;

    if (argc > 1)
    {
        system = atoi(argv[1]);
    }

    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

    double t0;
    double hypre_setup, hypre_solve;
    double raptor_setup, raptor_solve;

    //int coarsen_type = 0; // CLJP
    int coarsen_type = 6; // FALGOUT
    //int interp_type = 3; 
    int interp_type = 0;
    double strong_threshold = 0.25;
    int agg_num_levels = 0;
    int p_max_elmts = 0;

    int cache_len = 10000;
    int num_tests = 2;

    std::vector<double> cache_array(cache_len);

    if (system < 2)
    {
        double* stencil = NULL;
        std::vector<int> grid;
        if (argc > 2)
        {
            n = atoi(argv[2]);
        }

        if (system == 0)
        {
            dim = 3;
            grid.resize(dim, n);
            stencil = laplace_stencil_27pt();
        }
        else if (system == 1)
        {
            dim = 2;
            grid.resize(dim, n);
            double eps = 0.001;
            double theta = M_PI/8.0;
            if (argc > 3)
            {
                eps = atof(argv[3]);
                if (argc > 4)
                {
                    theta = atof(argv[4]);
                }
            }
            stencil = diffusion_stencil_2d(eps, theta);
        }
        A = par_stencil_grid(stencil, grid.data(), dim);
        delete[] stencil;
    }
    else if (system == 2)
    {
        /*char* mesh_file = "/u/sciteam/bienz/mfem/data/beam-tet.mesh";
        int num_elements = 2;
        int order = 3;
        if (argc > 2)
        {
            num_elements = atoi(argv[2]);
            if (argc > 3)
            {
                order = atoi(argv[3]);
                if (argc > 4)
                {
                    mesh_file = argv[4];
                }
            }
        }
        A = mfem_linear_elasticity(mesh_file, num_elements, order);*/
    }
    else if (system == 3)
    {
        char* file = "/Users/abienz/Documents/Parallel/raptor_topo/examples/LFAT5.mtx";
        A = readParMatrix(file, MPI_COMM_WORLD, 1, 1);
    }

    x = ParVector(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);
    b = ParVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    x.set_const_value(1.0);
    A->mult(x, b);

    // Convert system to Hypre format 
    HYPRE_IJMatrix A_h_ij = convert(A);
    HYPRE_IJVector x_h_ij = convert(&x);
    HYPRE_IJVector b_h_ij = convert(&b);
    hypre_ParCSRMatrix* A_h;
    HYPRE_IJMatrixGetObject(A_h_ij, (void**) &A_h);
    hypre_ParVector* x_h;
    HYPRE_IJVectorGetObject(x_h_ij, (void **) &x_h);
    hypre_ParVector* b_h;
    HYPRE_IJVectorGetObject(b_h_ij, (void **) &b_h);

    for (int i = 0; i < num_tests; i++)
    {
        HYPRE_Solver solver_data;
        Multilevel* ml;

        x.set_const_value(0.0);
        double* x_h_data = hypre_VectorData(hypre_ParVectorLocalVector(x_h));
        for (int i = 0; i < A->local_num_rows; i++)
        {
            x_h_data[i] = 0.0;
        }

        clear_cache(cache_array);

        // Create Hypre Hierarchy
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        solver_data = hypre_create_hierarchy(A_h, x_h, b_h, 
                                coarsen_type, interp_type, p_max_elmts, agg_num_levels, 
                                strong_threshold);
        hypre_setup = MPI_Wtime() - t0;
        clear_cache(cache_array);

        // Solve Hypre Hierarchy
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        HYPRE_BoomerAMGSolve(solver_data, A_h, b_h, x_h);
        hypre_solve = MPI_Wtime() - t0;
        clear_cache(cache_array);

        // Delete hypre hierarchy
        hypre_BoomerAMGDestroy(solver_data);     

        // Setup Raptor Hierarchy
        MPI_Barrier(MPI_COMM_WORLD);    
        t0 = MPI_Wtime();
        ml = new Multilevel(A, strong_threshold);
        raptor_setup = MPI_Wtime() - t0;
        clear_cache(cache_array);

        long lcl_nnz;
        long nnz;
        if (rank == 0) printf("Level\tNumRows\tNNZ\n");
        for (int i = 0; i < ml->num_levels; i++)
        {
            ParCSRMatrix* Al = ml->levels[i]->A;
            lcl_nnz = Al->local_nnz;
            MPI_Reduce(&lcl_nnz, &nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("%d\t%d\t%ld\n", i, Al->global_num_rows, nnz);
        }   

        // Solve Raptor Hierarchy
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        ml->solve(x, b);
        raptor_solve = MPI_Wtime() - t0;
        clear_cache(cache_array);

        MPI_Reduce(&hypre_setup, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Hypre Setup Time: %e\n", t0);
        MPI_Reduce(&hypre_solve, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Hypre Solve Time: %e\n", t0);

        MPI_Reduce(&raptor_setup, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Raptor Setup Time: %e\n", t0);
        MPI_Reduce(&raptor_solve, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Raptor Solve Time: %e\n", t0);

        // Delete raptor hierarchy
        delete ml;
    }

    delete A;
    HYPRE_IJMatrixDestroy(A_h_ij);
    HYPRE_IJVectorDestroy(x_h_ij);
    HYPRE_IJVectorDestroy(b_h_ij);
    MPI_Finalize();

    return 0;
}

