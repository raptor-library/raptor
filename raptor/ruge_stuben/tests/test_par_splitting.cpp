#include <assert.h>

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "ruge_stuben/par_cf_splitting.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/par_stencil.hpp"
#include <iostream>
#include <fstream>


#include "gallery/stencil.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FILE* f;
    ParCSRBoolMatrix* S;
    std::vector<int> splitting;
    std::vector<int> splitting_rap;
    std::vector<int> off_proc_states;

    int grid[3] = {10, 10, 10};
    double* stencil = laplace_stencil_27pt();

    CSRMatrix* S_seq;
    CSRMatrix* A_seq = stencil_grid(stencil, grid, 3);
    S_seq = A_seq->strength(0.25);
    std::vector<int> splitting_seq;
    split_cljp(S_seq, splitting_seq);
    f = fopen("../../tests/cljp_laplace_10.txt", "r");
    for (int i = 0; i < S_seq->n_rows; i++)
    {
        int cf;
        fscanf(f, "%d\n", &cf);
        assert(cf == splitting_seq[i]);
    }

    fclose(f);

    delete S_seq;
    delete A_seq;


    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 3);
    S = A->strength(0.25);
    split_cljp(S, splitting, off_proc_states);

    for (int i = 0; i < S->local_num_rows; i++)
    {
    //    printf("Splitting[%d] = %d\n", i + S->partition->first_local_row, splitting[i]);
    }
    f = fopen("../../tests/cljp_laplace_10.txt", "r");
    for (int i = 0; i < A->partition->first_local_row; i++)
    {
        int cf;
        fscanf(f, "%d\n", &cf);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        int cf;
        fscanf(f, "%d\n", &cf);
        assert(cf == splitting[i]);
        assert(splitting[i] == splitting_seq[i + A->partition->first_local_row]);
    }

    fclose(f);

    delete S;
    delete A;

    delete[] stencil;

    MPI_Finalize();

    return 0;
}
