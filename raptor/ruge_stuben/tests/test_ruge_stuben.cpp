#include <assert.h>

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"
#include "ruge_stuben/cf_splitting.hpp"
#include "ruge_stuben/interpolation.hpp"
#include "tests/compare.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    CSRMatrix *A;
    CSRMatrix* S;
    CSRMatrix* P;
    CSRMatrix* AP;
    CSCMatrix* P_csc;
    CSRMatrix* Ac_rap;
    CSRMatrix* Ac;
    std::vector<int> splitting;

    // Read in weights (for max num rows)
    FILE* f;
    int max_n = 5000;
    std::vector<double> weights(max_n);
    f = fopen("../../../../test_data/weights.txt", "r");
    for (int i = 0; i < max_n; i++)
    {
        fscanf(f, "%lf\n", &weights[i]);
    }
    fclose(f);

    // TEST LEVEL 0
    A = readMatrix("../../../../test_data/rss_A0.mtx", 1);
    S = A->strength(0.25);
    split_cljp(S, splitting, weights.data());
    P = direct_interpolation(A, S, splitting);
    AP = A->mult(P);
    P_csc = new CSCMatrix(P);
    Ac_rap = AP->mult_T(P_csc);
    Ac = readMatrix("../../../../test_data/rss_A1.mtx", 0);
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
    S = A->strength(0.25);
    split_cljp(S, splitting, weights.data());
    P = direct_interpolation(A, S, splitting);
    AP = A->mult(P);
    P_csc = new CSCMatrix(P);
    Ac_rap = AP->mult_T(P_csc);
    Ac = readMatrix("../../../../test_data/rss_A2.mtx", 0);
    compare(Ac, Ac_rap);
    delete Ac;
    delete Ac_rap;
    delete P_csc;
    delete AP;
    delete P;
    delete S;
    delete A;

    return 0;
}
