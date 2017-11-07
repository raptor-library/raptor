#include <assert.h>

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "tests/par_compare.hpp"
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

    ParCSRMatrix* A;
    HYPRE_IJMatrix A_h;
    hypre_ParCSRMatrix* parcsr_A;

    ParCSRMatrix* S;
    hypre_ParCSRMatrix* parcsr_S;

    ParCSRMatrix* P;
    hypre_ParCSRMatrix* parcsr_P;

    ParCSRMatrix* AP;
    ParCSRMatrix* Ac;
    ParCSCMatrix* P_csc;
    hypre_ParCSRMatrix* parcsr_AP;
    hypre_ParCSRMatrix* parcsr_Ac;

    std::vector<int> states;
    std::vector<int> off_proc_states;
    std::vector<double> weights;
    int* CF_marker = NULL;
    

    // Read in fine level, and convert to hypre format
    A = readParMatrix("../../../../test_data/rss_A0.mtx", MPI_COMM_WORLD, 1, 1);
    A_h = convert(A);
    HYPRE_IJMatrixGetObject(A_h, (void**) &parcsr_A);
    compare(A, parcsr_A);

    // Form strength matrices and compare
    S = A->strength(0.25);
    hypre_BoomerAMGCreateS(parcsr_A, 0.25, 1, 1, NULL, &parcsr_S);
    compareS(S, parcsr_S);

    // Form CLJP splittings and compare
    form_hypre_weights(weights, A->local_num_rows);

    split_cljp(S, states, off_proc_states, weights.data());
    hypre_BoomerAMGCoarsen(parcsr_S, parcsr_A, 0, 0, &CF_marker);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        if (states[i])
            assert(states[i] == CF_marker[i]);
        else assert(CF_marker[i] == -1);
    }

    // Form Modified Classical Interpolation and Compare
    int* coarse_pnts_global = NULL;
    hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, A->local_num_rows, 1, NULL,
            CF_marker, NULL, &coarse_pnts_global);
    hypre_BoomerAMGBuildInterp(parcsr_A, CF_marker, parcsr_S, 
            coarse_pnts_global, 1, NULL, 0, 0.0, 0, NULL, &parcsr_P);
    int first_row = hypre_ParCSRMatrixFirstRowIndex(parcsr_P);
    int first_col = hypre_ParCSRMatrixFirstColDiag(parcsr_P);
    int local_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(parcsr_P));
    int local_cols = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(parcsr_P));
    P = mod_classical_interpolation(A, S, states, off_proc_states);

    compare(P, parcsr_P);
    free(CF_marker);

    // Form Coarse Grid Ac
    AP = A->mult(P);
    AP->sort();
    parcsr_AP = hypre_ParMatmul(parcsr_A, parcsr_P);
    P_csc = new ParCSCMatrix(P);
    Ac = AP->mult_T(P_csc);
    if (!Ac->comm)
        Ac->comm = new ParComm(Ac->partition, Ac->off_proc_column_map, 
                Ac->on_proc_column_map);
    parcsr_Ac = hypre_ParTMatmul(parcsr_P, parcsr_AP);
    compare(Ac, parcsr_Ac);

    // Form Strength Matrices
    ParCSRMatrix* Sc = Ac->strength(0.25);
    hypre_ParCSRMatrix* parcsr_Sc;
    hypre_BoomerAMGCreateS(parcsr_Ac, 0.25, 1, 1, NULL, &parcsr_Sc);
    compareS(Sc, parcsr_Sc);

    // Form CLJP splittings and compare
    split_cljp(Sc, states, off_proc_states, weights.data());
    hypre_BoomerAMGCoarsen(parcsr_Sc, parcsr_Ac, 0, 0, &CF_marker);
    for (int i = 0; i < Ac->local_num_rows; i++)
    {
        if (states[i])
            assert(states[i] == CF_marker[i]);
        else assert(CF_marker[i] == -1);
    }

    // Form Modified Classical Interpolation and Compare
    hypre_ParCSRMatrix* parcsr_Pc;
    int* coarse_pnts_global_c = NULL;
    hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, Ac->local_num_rows, 1, NULL,
            CF_marker, NULL, &coarse_pnts_global_c);
    hypre_BoomerAMGBuildInterp(parcsr_Ac, CF_marker, parcsr_Sc, 
            coarse_pnts_global_c, 1, NULL, 0, 0.0, 0, NULL, &parcsr_Pc);
    ParCSRMatrix* Pc = mod_classical_interpolation(Ac, Sc, states, off_proc_states);
    compare(Pc, parcsr_Pc);
    free(CF_marker);

    hypre_ParCSRMatrixDestroy(parcsr_Pc);
    delete Pc;

    hypre_ParCSRMatrixDestroy(parcsr_Sc);
    delete Sc;

    hypre_ParCSRMatrixDestroy(parcsr_Ac);
    delete P_csc;
    delete Ac;

    hypre_ParCSRMatrixDestroy(parcsr_AP);
    delete AP; 

    hypre_ParCSRMatrixDestroy(parcsr_P);
    delete P;

    hypre_ParCSRMatrixDestroy(parcsr_S);
    delete S;

    HYPRE_IJMatrixDestroy(A_h);
    delete A;

    MPI_Finalize();
    return 0;

}

