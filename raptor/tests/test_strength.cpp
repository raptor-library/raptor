#include <assert.h>

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"
#include "tests/compare.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    CSRMatrix* A;
    CSRMatrix* S;
    CSRMatrix* S_rap;

    A = readMatrix("../../../test_data/rss_A0.mtx", 1);
    S = readMatrix("../../../test_data/rss_S0.mtx", 1);
    S_rap = A->strength(0.25);
    compare_pattern(S, S_rap);
    delete A;
    delete S;
    delete S_rap;

    A = readMatrix("../../../test_data/rss_A1.mtx", 0);
    S = readMatrix("../../../test_data/rss_S1.mtx", 0);
    S_rap = A->strength(0.25);
    compare_pattern(S, S_rap);
    delete A;
    delete S;
    delete S_rap;

}
