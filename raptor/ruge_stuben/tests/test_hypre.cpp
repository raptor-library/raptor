#include <assert.h>

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/laplacian27pt.hpp"
#include "tests/hypre_compare.hpp"
#include "gallery/external/hypre_wrapper.hpp"
#include "multilevel/par_multilevel.hpp"
#include "multilevel/multilevel.hpp"

using namespace raptor;

void form_hypre_weights(std::vector<double>& weights, int n_rows)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (n_rows)
    {
        weights.resize(n_rows);
        int seed = 2747 + rank;
        int a = 16807;
        int m = 2147483647;
        int q = 127773;
        int r = 2836;
        for (int i = 0; i < n_rows; i++)
        {
            int high = seed / q;
            int low = seed % q;
            int test = a * low - r * high;
            if (test > 0) seed = test;
            else seed = test + m;
            weights[i] = ((double)(seed) / m);
        }
    }
}

int main(int argc, char* argv[])
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

    int coarsen_type = 0; // CLJP
    //int coarsen_type = 6; // FALGOUT
    //int interp_type = 3; // Direct Interp
    int interp_type = 0; // Classical Mod Interp
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
        int sym = 1;
        if (argc > 2)
        {
            file = argv[2];
            if (argc > 3)
            {
                sym = atoi(argv[3]);
            }
        }
        A = readParMatrix(file, MPI_COMM_WORLD, 1, sym);
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

    HYPRE_Solver solver_data = hypre_create_hierarchy(A_h, x_h, b_h, 
                                coarsen_type, interp_type, p_max_elmts, agg_num_levels, 
                                strong_threshold);
    ParMultilevel* ml = new ParMultilevel(A, strong_threshold);

    HYPRE_Int num_levels = hypre_ParAMGDataNumLevels((hypre_ParAMGData*) solver_data);
    hypre_ParCSRMatrix** A_array = hypre_ParAMGDataAArray((hypre_ParAMGData*) solver_data);
    hypre_ParCSRMatrix** P_array = hypre_ParAMGDataPArray((hypre_ParAMGData*) solver_data);

    assert(ml->num_levels == num_levels);
    ParCSRMatrix* A6 = ml->levels[6]->A;
    hypre_ParCSRMatrix* A6_h = A_array[6];
    compare(A6, A6_h);

    ParCSRMatrix* S6 = A6->strength(0.25);
    hypre_ParCSRMatrix* S6_h;
    hypre_BoomerAMGCreateS(A6_h, 0.25, 1, 1, NULL, &S6_h);
    compareS(S6, S6_h);

    std::vector<double> weights;
    form_hypre_weights(weights, A6->local_num_rows);
    std::vector<int> states;
    std::vector<int> off_proc_states;
    int* CF_marker;
    split_cljp(S6, states, off_proc_states, weights.data());
    hypre_BoomerAMGCoarsen(S6_h, A6_h, 0, 0, &CF_marker);
    for (int i = 0; i < A6->local_num_rows; i++)
    {
        if (states[i])
            assert(CF_marker[i] == states[i]);
        else
            assert(CF_marker[i] == -1);
        if (states[i] == -3) printf("States[%d] = %d, CFMarker %d\n", i, states[i], CF_marker[i]);
    }

    // NOT RIGHT YET
    int* coarse_pnts_global = NULL;
    hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, A6->local_num_rows, 1, NULL,
            CF_marker, NULL, &coarse_pnts_global);
    hypre_ParCSRMatrix* P6_h;
    hypre_BoomerAMGBuildInterp(A6_h, CF_marker, S6_h, 
            coarse_pnts_global, 1, NULL, 0, 0.0, 0, NULL, &P6_h);
    ParCSRMatrix* P6 = mod_classical_interpolation(A6, S6, states, off_proc_states);
    compare(P6, P6_h);
    printf("P6 %d, %d P6_h %d, %d\n", P6->global_num_rows, P6->global_num_cols,
            hypre_ParCSRMatrixGlobalNumRows(P6_h),
            hypre_ParCSRMatrixGlobalNumCols(P6_h));
    free(CF_marker);

    delete S6;
    delete P6;
    hypre_ParCSRMatrixDestroy(S6_h);
    hypre_ParCSRMatrixDestroy(P6_h);

    hypre_BoomerAMGDestroy(solver_data); 
    delete ml;

    delete A;
    HYPRE_IJMatrixDestroy(A_h_ij);
    HYPRE_IJVectorDestroy(x_h_ij);
    HYPRE_IJVectorDestroy(b_h_ij);
    MPI_Finalize();

    return 0;
}

