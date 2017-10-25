#include <assert.h>

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"

using namespace raptor;

void compare(CSRMatrix* S, CSRBoolMatrix* S_rap)
{
    int start, end;

    S->sort();
    S->move_diag();
    S_rap->sort();
    S_rap->move_diag();

    assert(S->n_rows == S_rap->n_rows);
    assert(S->n_cols == S_rap->n_cols);
    assert(S->nnz == S_rap->nnz);

    assert(S->idx1[0] == S_rap->idx1[0]);
    for (int i = 0; i < S->n_rows; i++)
    {
        assert(S->idx1[i+1] == S_rap->idx1[i+1]);
        start = S->idx1[i];
        end = S->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            assert(S->idx2[j] == S_rap->idx2[j]);
        }
    }
}

int main(int argc, char* argv[])
{
    CSRMatrix* A;
    CSRMatrix* S;
    CSRBoolMatrix* S_rap;

    A = readMatrix("rss_laplace_A0.mtx", 1);
    S = readMatrix("rss_laplace_S0.mtx", 1);
    S_rap = A->strength(0.25);
    compare(S, S_rap);
    delete A;
    delete S;
    delete S_rap;


    A = readMatrix("rss_laplace_A1.mtx", 0);
    S = readMatrix("rss_laplace_S1.mtx", 0);
    S_rap = A->strength(0.25);
    compare(S, S_rap);
    delete A;
    delete S;
    delete S_rap;

    A = readMatrix("rss_aniso_A0.mtx", 1);
    S = readMatrix("rss_aniso_S0.mtx", 1);
    S_rap = A->strength();
    compare(S, S_rap);
    delete A;
    delete S;
    delete S_rap;

    A = readMatrix("rss_aniso_A1.mtx", 0);
    S = readMatrix("rss_aniso_S1.mtx", 0);
    S_rap = A->strength();
    compare(S, S_rap);
    delete A;
    delete S;
    delete S_rap;

}
