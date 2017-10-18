// EXPECT_EQ and ASSERT_EQ are macros
// EXPECT_EQ test execution and continues even if there is a failure
// ASSERT_EQ test execution and aborts if there is a failure
// The ASSERT_* variants abort the program execution if an assertion fails 
// while EXPECT_* variants continue with the run.


#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/vector.hpp"
#include "core/par_vector.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
    MPI_Finalize();

} // end of main() //

TEST(ParVectorTest, TestsInCore)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int global_n = 100;
    int local_n = global_n / num_procs;
    int first_n = rank * ( global_n / num_procs);

    if (global_n % num_procs > rank)
    {
        local_n++;
        first_n += rank;
    }
    else
    {
        first_n += (global_n % num_procs);
    }

    Vector v(global_n);
    ParVector v_par(global_n, local_n, first_n);

    v.set_const_value(1.0);
    v_par.set_const_value(1.0);

    Vector& v_par_l = v_par.local;
    for (int i = 0; i < local_n; i++)
    {
        ASSERT_EQ( v[first_n+i], v_par_l[i] );
        //EXPECT_EQ( v[first_n+i], v_par_l[i] );
        //EXPECT_DOUBLE_EQ( v[first_n+i], v_par_l[i] );
        //EXPECT_FLOAT_EQ( v[first_n+i], v_par_l[i] );
    }
    
    for (int i = 0; i < global_n; i++)
    {
        srand(i);
        v[i] = ((double)rand()) / RAND_MAX;
    }
    for (int i = 0; i < local_n; i++)
    {
        srand(i+first_n);
        v_par_l[i] = ((double)rand()) / RAND_MAX;
    }

    for (int i = 0; i < local_n; i++)
    {
        ASSERT_EQ(v[first_n+i], v_par_l[i]);
    }
    
} // end of TEST(ParVectorTest, TestsInCore) //

