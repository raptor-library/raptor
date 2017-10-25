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
    std::vector<int> splitting;
    std::vector<int> splitting_rap;
    std::vector<int> off_proc_states;

    int grid[3] = {10, 10, 10};
    double* stencil = laplace_stencil_27pt();

    CSRBoolMatrix* S_seq;
    CSRMatrix* S_seq_py;

    S_seq_py = readMatrix("../../tests/rss_laplace_S0.mtx", 1);
    S_seq = new CSRBoolMatrix(S_seq_py);
    std::vector<int> splitting_seq;
    split_cljp(S_seq, splitting_seq);
    f = fopen("../../tests/rss_laplace_cf0.txt", "r");
    for (int i = 0; i < S_seq->n_rows; i++)
    {
        int cf;
        fscanf(f, "%d\n", &cf);
        //assert(cf == splitting_seq[i]);
    }

    fclose(f);

    delete S_seq;
    delete S_seq_py;

    ParCSRMatrix* S_py;
    ParCSRBoolMatrix* S;
    S_py = readParMatrix("../../tests/rss_laplace_S0.mtx", MPI_COMM_WORLD, 1, 1);
    S = new ParCSRBoolMatrix(S_py);
    split_cljp(S, splitting, off_proc_states);
    f = fopen("../../tests/rss_laplace_cf0.txt", "r");
    for (int i = 0; i < S->partition->first_local_row; i++)
    {
        int cf;
        fscanf(f, "%d\n", &cf);
    }
    for (int i = 0; i < S->local_num_rows; i++)
    {
        int cf;
        fscanf(f, "%d\n", &cf);
        assert(cf == splitting[i]);
        //assert(splitting[i] == splitting_seq[i + A->partition->first_local_row]);
    }

    fclose(f);

    delete S;
    delete S_py;

    delete[] stencil;

    MPI_Finalize();

    return 0;
}
