// EXPECT_EQ and ASSERT_EQ are macros
// EXPECT_EQ test execution and continues even if there is a failure
// ASSERT_EQ test execution and aborts if there is a failure
// The ASSERT_* variants abort the program execution if an assertion fails 
// while EXPECT_* variants continue with the run.


#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

} // end of main() //

TEST(TestRAP, TestsInRuge_Stuben)
{ 
    char name[128];
    int ctr = 0;
    int start, end;

    CSRMatrix* A;
    CSRMatrix* P;
    CSRMatrix* Ac;
    CSRMatrix* AP;
    CSCMatrix* AP_csc;
    CSRMatrix* Ac_rap;

    snprintf(name, sizeof(name), "../../tests/rss_laplace_P%d.mtx", ctr);

    while (FILE *file = fopen(name, "r")) 
    {
        fclose(file);
        
        snprintf(name, sizeof(name), "../../tests/rss_laplace_A%d.mtx", ctr);
        if (ctr == 0)
        {
            A = readMatrix(name, 1);
        }
        else
        {
            A = readMatrix(name, 0);
        }

        snprintf(name, sizeof(name), "../../tests/rss_laplace_P%d.mtx", ctr);        
        P = readMatrix(name, 0);

        snprintf(name, sizeof(name), "../../tests/rss_laplace_A%d.mtx", ctr+1);        
        Ac = readMatrix(name, 0);

        AP = A->mult(P);
        AP_csc = new CSCMatrix(AP);
        Ac_rap = P->mult_T(AP_csc);

        Ac->sort();
        Ac_rap->sort();
        ASSERT_EQ(Ac->n_rows, Ac_rap->n_rows);
        ASSERT_EQ(Ac->n_cols, Ac_rap->n_cols);
        ASSERT_EQ(Ac->nnz, Ac_rap->nnz);

        ASSERT_EQ(Ac->idx1[0], Ac_rap->idx1[0]);
        for (int i = 0; i < Ac->n_rows; i++)
        {
            ASSERT_EQ(Ac->idx1[i+1], Ac_rap->idx1[i+1]);
            start = Ac->idx1[i];
            end = Ac->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                ASSERT_EQ(Ac->idx2[j], Ac_rap->idx2[j]);
                ASSERT_NEAR(Ac->vals[j], Ac_rap->vals[j], 1e-06);
            }
        }

        ctr++;
        snprintf(name, sizeof(name), "../../tests/rss_laplace_P%d.mtx", ctr);

        delete A;
        delete P;
        delete Ac;
        delete AP;
        delete AP_csc;
        delete Ac_rap;
    }

    ctr = 0;
    snprintf(name, sizeof(name), "../../tests/rss_aniso_P%d.mtx", ctr);
    while (FILE *file = fopen(name, "r")) 
    {
        fclose(file);
        
        snprintf(name, sizeof(name), "../../tests/rss_aniso_A%d.mtx", ctr);
        if (ctr == 0)
        {
            A = readMatrix(name, 1);
        }
        else
        {
            A = readMatrix(name, 0);
        }

        snprintf(name, sizeof(name), "../../tests/rss_aniso_P%d.mtx", ctr);        
        P = readMatrix(name, 0);

        snprintf(name, sizeof(name), "../../tests/rss_aniso_A%d.mtx", ctr+1);        
        Ac = readMatrix(name, 0);

        AP = A->mult(P);
        AP_csc = new CSCMatrix(AP);
        Ac_rap = P->mult_T(AP_csc);

        Ac->sort();
        Ac_rap->sort();
        ASSERT_EQ(Ac->n_rows, Ac_rap->n_rows);
        ASSERT_EQ(Ac->n_cols, Ac_rap->n_cols);
        ASSERT_EQ(Ac->nnz, Ac_rap->nnz);

        ASSERT_EQ(Ac->idx1[0], Ac_rap->idx1[0]);
        for (int i = 0; i < Ac->n_rows; i++)
        {
            ASSERT_EQ(Ac->idx1[i+1], Ac_rap->idx1[i+1]);
            start = Ac->idx1[i];
            end = Ac->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                ASSERT_EQ(Ac->idx2[j], Ac_rap->idx2[j]);
                ASSERT_NEAR(Ac->vals[j], Ac_rap->vals[j], 1e-06);
            }
        }

        ctr++;
        snprintf(name, sizeof(name), "../../tests/rss_aniso_P%d.mtx", ctr);

        delete A;
        delete P;
        delete Ac;
        delete AP;
        delete AP_csc;
        delete Ac_rap;
    }

} // end of TEST(TestRAP, TestsInRuge_Stuben) //

