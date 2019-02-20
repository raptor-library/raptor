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

TEST(TestRugeStuben, TestsInRuge_Stuben)
{ 
    CSRMatrix *A;
    CSRMatrix* S;
    CSRMatrix* P;
    CSRMatrix* AP;
    CSCMatrix* P_csc;
    CSRMatrix* Ac_rap;
    CSRMatrix* Ac;
    aligned_vector<int> splitting;

    const char* weight_fn = "../../../../test_data/weights.txt";
    const char* A0_fn = "../../../../test_data/rss_A0.pm";
    const char* A1_fn = "../../../../test_data/rss_A1.pm";
    const char* A2_fn = "../../../../test_data/rss_A2.pm";

    // Read in weights (for max num rows)
    FILE* f;
    int max_n = 5000;
    aligned_vector<double> weights(max_n);
    f = fopen(weight_fn, "r");
    for (int i = 0; i < max_n; i++)
    {
        fscanf(f, "%lf\n", &weights[i]);
    }
    fclose(f);

    // TEST LEVEL 0
    A = readMatrix(A0_fn);
    S = A->strength(Classical, 0.25);
    split_cljp(S, splitting, weights.data());
    P = direct_interpolation(A, S, splitting);
    AP = A->mult(P);
    P_csc = P->to_CSC();
    Ac_rap = AP->mult_T(P_csc);
    Ac = readMatrix(A1_fn);
    compare(Ac, Ac_rap);
    delete Ac;
    delete P_csc;
    delete AP;
    delete P;
    delete S;
    delete A;

    // TEST LEVEL 1
    A = Ac_rap;
    Ac_rap = NULL;
    S = A->strength(Classical, 0.25);
    split_cljp(S, splitting, weights.data());
    P = direct_interpolation(A, S, splitting);
    AP = A->mult(P);
    P_csc = P->to_CSC();
    Ac_rap = AP->mult_T(P_csc);
    Ac = readMatrix(A2_fn);
    compare(Ac, Ac_rap);
    delete Ac;
    delete Ac_rap;
    delete P_csc;
    delete AP;
    delete P;
    delete S;
    delete A;
} // end of TEST(TestRugeStuben, TestsInRuge_Stuben) //

