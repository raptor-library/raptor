// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor.hpp"
#include "tests/compare.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} // end of main() //

TEST(AnisoTest, TestsInGallery)
{
    const char* f_in = "../../../../test_data/sas_P0.mtx";
    const char* f_out = "../../../../test_data/sas_P0_out.mtx";
    CSRMatrix* Amm = read_mm(f_in);
    write_mm(Amm, f_out);
    CSRMatrix* Amm_out = read_mm(f_out);
    compare(Amm, Amm_out);

    // Diff the two mtx files 
    std::string command = "diff ";
    command += f_in;
    command += " ";
    command += f_out;
    int err = system(command.c_str());


    remove(f_out);
     
    delete Amm;
} // end of TEST(AnisoTest, TestsInGallery) //


