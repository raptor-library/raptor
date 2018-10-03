#include <assert.h>
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "krylov/partial_inner.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_stencil.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm inner_comm, roots_comm; 
    MPI_Comm half_inner_comm, half_roots_comm; 

    int grid[2] = {50, 50};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    ParVector x(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector y(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);

    //x.set_const_value(1.0);
    //y.set_const_value(1.0);
    x.set_rand_values();
    y.set_rand_values();

    data_t inner, other_half;
    int other_root;
    int color, root_color, root, procs_in_group, part_global, groups;
    /*std::vector<double> fracs = {0.5, 0.333, 0.25, 0.2, 0.1};

    for (int i=0; i<fracs.size(); i++) {
        inner_comm = MPI_COMM_NULL;
        roots_comm = MPI_COMM_NULL;
        create_partial_inner_comm(inner_comm, roots_comm, fracs[i], x, color, root_color, root, procs_in_group, part_global);
       
        groups = 1 / fracs[i];
        for (int j=0; j<groups; j++) {
            inner = partial_inner(inner_comm, roots_comm, x, y, color, j, root, procs_in_group, part_global);
            assert(fabs(inner - x.global_n) < 1e-01);
            MPI_Barrier(MPI_COMM_WORLD); 
        }
    }*/

    int my_ind;
    std::vector<int> roots;
    bool am_root;
    //std::vector<double> fracs = {0.5, 0.333, 0.25, 0.2, 0.1};
    std::vector<double> fracs = {0.5};

    data_t partial_other_half, half_other_half;

    for (int i=0; i<fracs.size(); i++) {
        inner_comm = MPI_COMM_NULL;
        roots_comm = MPI_COMM_NULL;
        create_partial_inner_comm_v2(inner_comm, roots_comm, fracs[i], x, my_ind, roots, am_root);
        create_partial_inner_comm(half_inner_comm, half_roots_comm, fracs[i], x, color, root_color, root, procs_in_group, part_global);
        if (root == 0) other_root = num_procs/2;
        else other_root = 0;

        inner = half_inner(inner_comm, x, y);
        partial_other_half = partial_inner_communicate(inner_comm, roots_comm, inner, my_ind, roots, am_root);
        half_other_half = half_inner_communicate(half_inner_comm, inner, root, other_root);

        partial_other_half += inner;
        half_other_half += inner;

        for (int j = 0; j < num_procs; j++) {
            if (rank == j) printf("%d half %lf partial %lf\n", rank, half_other_half, partial_other_half);
            MPI_Barrier(MPI_COMM_WORLD);
        }

        assert(fabs(inner - x.global_n) < 1e-01);
        MPI_Barrier(MPI_COMM_WORLD); 
    }

    MPI_Comm_free(&inner_comm);
    MPI_Comm_free(&roots_comm);
    delete[] stencil;
    delete A;

    MPI_Finalize();

    return 0;
}
