#include <assert.h>

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"
#include "ruge_stuben/cf_splitting.hpp"
#include "ruge_stuben/interpolation.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"
#include "tests/compare.hpp"

#include <iostream>
#include <fstream>

using namespace raptor;
int main(int argc, char* argv[])
{
    CSRMatrix* A;
    CSRMatrix* S;
    CSRBoolMatrix* S_bool;
    CSRMatrix* P;
    CSRMatrix* P_rap;
    std::vector<int> splitting;
    FILE* f;

    // TEST LEVEL 0
    A = readMatrix("../../../../test_data/rss_A0.mtx", 1);
    S = readMatrix("../../../../test_data/rss_S0.mtx", 1);
    P = readMatrix("../../../../test_data/rss_P0.mtx", 0);
    S_bool = new CSRBoolMatrix(S);
    splitting.resize(A->n_rows);
    f = fopen("../../../../test_data/rss_cf0", "r");
    for (int i = 0; i < A->n_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);

    P_rap = direct_interpolation(A, S_bool, splitting);
    compare(P, P_rap);

    delete P_rap;
    delete S_bool;
    delete S;
    delete P;
    delete A;


    // TEST LEVEL 1
    A = readMatrix("../../../../test_data/rss_A1.mtx", 0);
    P = readMatrix("../../../../test_data/rss_P1.mtx", 0);
    S = readMatrix("../../../../test_data/rss_S1.mtx", 0);
    S_bool = new CSRBoolMatrix(S);
    splitting.resize(A->n_rows);
    f = fopen("../../../../test_data/rss_cf1", "r");
    for (int i = 0; i < A->n_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);

    P_rap = direct_interpolation(A, S_bool, splitting);
    compare(P, P_rap);

    delete P_rap;
    delete S_bool;
    delete S;
    delete P;
    delete A;



    return 0;
}