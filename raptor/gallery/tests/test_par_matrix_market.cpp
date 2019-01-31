// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor.hpp"
#include "tests/par_compare.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp = RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(ParAnisoTest, TestsInGallery)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    const char* f_in = "../../../../test_data/sas_P0.mtx";
    const char* f_out = "../../../../test_data/sas_P0_out.mtx";

    ParCSRMatrix* Amm = read_par_mm(f_in);

    MPI_Barrier(MPI_COMM_WORLD);
    write_par_mm(Amm, f_out);

    MPI_Barrier(MPI_COMM_WORLD);
    ParCSRMatrix* Amm_out = read_par_mm(f_out);
    
    MPI_Barrier(MPI_COMM_WORLD);
    compare(Amm, Amm_out);

    // Diff the two mtx files 
    if (rank == 0)
    {
        remove(f_out);
    }

    delete Amm_out;
    delete Amm;

 } // end of TEST(ParAnisoTest, TestsInGallery) //


