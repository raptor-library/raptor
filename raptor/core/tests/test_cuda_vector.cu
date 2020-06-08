// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor.hpp"

using namespace raptor;
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

} // end of main() //

TEST(CudaVectorTest, TestsInCore)
{
    int n = 100;
    Vector v(n);

    printf("num values %d\n", v.num_values);
    printf("b_vecs %d\n", v.b_vecs);
    printf("values size %d\n", v.values.size());
    
    printf("tblocks %d\n", v.tblocks);
    printf("blocksize %d\n", v.blocksize);

    if (v.dev_ptr) printf("dev_ptr not NULL\n");

    // Need to set vector on host to 0 and then copy to device?
    v.print();

    v.set_const_value(1.0);
    v.copy_from_device();

    v.print();

    /*for (int i = 0; i < global_n; i++)
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
    }*/
    
} // end of TEST(CudaVectorTest, TestsInCore) //

