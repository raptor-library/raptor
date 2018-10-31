#include <assert.h>

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "aorthonormalization/par_mgs.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_stencil.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_vectors = 4;
    int grid[2] = {50, 50};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    aligned_vector<ParVector> W;
    aligned_vector<aligned_vector<ParVector>> P_list;

    for (int i = 0; i < num_vectors; i++) {
        ParVector w(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
        w.set_rand_values();
        W.push_back(w);
    }

    for (int i = 0; i < 2; i++) {
        aligned_vector<Vector> P_sublist;
        for (int j = 0; j < num_vectors; j++) {
            ParVector p(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
            p.set_rand_values();
            P_sublist.push_back(p);
        }
        P_list.push_back(P_sublist);
    }

    MGS(A, W, P_list);

    // Insert check
    
    MGS(A, W);

    // Insert check

    delete[] stencil;
    delete A;
    
    MPI_Finalize();

    return 0;
}
