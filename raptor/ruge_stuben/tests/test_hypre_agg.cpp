// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause


#include "gtest/gtest.h"
#include "mpi.h"
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/external/hypre_wrapper.hpp"
#include "ruge_stuben/par_cf_splitting.hpp"
#include "ruge_stuben/par_interpolation.hpp"
#include "tests/hypre_compare.hpp"
#include <iostream>
#include <fstream>

using namespace raptor;

void form_hypre_weights(double** weight_ptr, int n_rows)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    hypre_SeedRand(2747 + rank);
    double* weights;
    if (n_rows)
    {
        weights = new double[n_rows];
        for (int i = 0; i < n_rows; i++)
        {
            weights[i] = hypre_Rand();
        }
    }

    *weight_ptr = weights;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp = RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(TestHypreAgg, TestsInRuge_Stuben)
{ 
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FILE* f;
    double* weights;
    int cf;

    int n = 25;
    std::vector<int> grid(3, n);
    double* stencil = laplace_stencil_27pt();

    std::vector<ParCSRMatrix*> A_array;
    std::vector<ParCSRMatrix*> P_array;

    ParCSRMatrix* A = par_stencil_grid(stencil, grid.data(), 3);
    A_array.push_back(A);
    delete[] stencil;
    form_hypre_weights(&weights, A->local_num_rows);

    std::vector<int> states;
    std::vector<int> off_proc_states;

    int nrows = A_array[0]->global_num_rows;
    int level = 0;
//    while (nrows > 500)
    {
        ParCSRMatrix* Al = A_array[level];
        printf("Al->global_num_rows, %d\n", Al->global_num_rows);
        ParCSRMatrix* Sl = Al->strength(0.25);
        split_pmis(Sl, states, off_proc_states, weights);

        ParCSRMatrix* Pl = mod_classical_interpolation(Al, Sl, states, off_proc_states, Al->comm);
        //ParCSRMatrix* Pl = extended_interpolation(Al, Sl, states, off_proc_states, Al->comm);
        P_array.push_back(Pl);

        ParCSRMatrix* APl = Al->mult(Pl);
        ParCSCMatrix* Pcsc = new ParCSCMatrix(Pl);
        ParCSRMatrix* Ac = APl->mult_T(Pcsc);
        Ac->comm = new ParComm(Ac->partition, Ac->off_proc_column_map, Ac->on_proc_column_map);
        A_array.push_back(Ac);


        nrows = Ac->global_num_rows;
        level++;

        ParCSRMatrix* Al2 = A_array[level];
        printf("Al2->global_num_rows, %d\n", Al2->global_num_rows);
        ParCSRMatrix* Sl2 = Al2->strength(0.25);
        split_pmis(Sl2, states, off_proc_states, weights);
        ParCSRMatrix* Pl2 = extended_interpolation(Al2, Sl2, states, off_proc_states, Al2->comm);


        delete Sl;
        delete APl;
        delete Pcsc;
    }

    for (std::vector<ParCSRMatrix*>::iterator it = A_array.begin(); it != A_array.end(); ++it)
        delete *it;

    for (std::vector<ParCSRMatrix*>::iterator it = P_array.begin(); it != P_array.end(); ++it)
        delete *it;

    delete[] weights;


} // end of TEST(TestParSplitting, TestsInRuge_Stuben) //






