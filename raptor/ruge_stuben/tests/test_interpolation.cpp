#include <assert.h>

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"
#include "ruge_stuben/cf_splitting.hpp"
#include "ruge_stuben/interpolation.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"

#include <iostream>
#include <fstream>

using namespace raptor;

void compare(CSRMatrix* P, CSRMatrix* P_rap)
{
    int start, end;

    assert(P->n_rows == P_rap->n_rows);
    assert(P->n_cols == P_rap->n_cols);
    assert(P->nnz == P_rap->nnz);
    assert(P->idx1[0] == P_rap->idx1[0]);
    for (int i = 0; i < P->n_rows; i++)
    {
        assert(P->idx1[i+1] == P_rap->idx1[i+1]);
        start = P->idx1[i];
        end = P->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            assert(P->idx2[j] == P_rap->idx2[j]);
            assert(fabs(P->vals[j] - P_rap->vals[j]) < 1e-06);
        }
    }
}

int main(int argc, char* argv[])
{
    CSRMatrix* A;
    CSRMatrix* S;
    CSRBoolMatrix* S_bool;
    CSRMatrix* P;
    CSRMatrix* P_rap;
    std::vector<int> splitting;
    A = readMatrix("../../tests/rss_laplace_A0.mtx", 1);
    P = readMatrix("../../tests/rss_laplace_P0.mtx", 0);
    S = readMatrix("../../tests/rss_laplace_S0.mtx", 1);
    S_bool = new CSRBoolMatrix(S);
    split_rs(S_bool, splitting);
    P_rap = direct_interpolation(A, S_bool, splitting);
    compare(P, P_rap);
    delete A;
    delete P;
    delete S;
    delete S_bool;
    delete P_rap;

    A = readMatrix("../../tests/rss_laplace_A1.mtx", 0);
    P = readMatrix("../../tests/rss_laplace_P1.mtx", 0);
    //S = A->strength(0.25);
    //split_rs(S, splitting);
    //P_rap = direct_interpolation(A, S, splitting);
    //compare(P, P_rap);
    delete A;
    delete P;
    //delete S;
    //delete P_rap;

    return 0;
}
