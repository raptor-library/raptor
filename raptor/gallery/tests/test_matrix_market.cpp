// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.

#include "gtest/gtest.h"
#include "raptor/raptor.hpp"
#include "raptor/tests/compare.hpp"

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
    ASSERT_EQ(err, 0);


    remove(f_out);
     
    delete Amm_out;
    delete Amm;
} // end of TEST(AnisoTest, TestsInGallery) //


