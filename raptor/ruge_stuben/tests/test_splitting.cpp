#include <assert.h>

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"
#include "ruge_stuben/cf_splitting.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"

#include <iostream>
#include <fstream>

using namespace raptor;

int main(int argc, char* argv[])
{
    FILE* f;
    CSRMatrix* S;
    std::vector<int> splitting;
    std::vector<int> splitting_rap;
   
    // TEST LAPLACIAN SPLITTINGS ON LEVEL 0 
    S = readMatrix("../../../../test_data/rss_S0.mtx", 1);
    splitting.resize(S->n_rows);;

    // Test RugeStuben Splitting
    split_rs(S, splitting_rap);
    f = fopen("../../../../test_data/rss_cf0_rs", "r");
    for (int i = 0; i < S->n_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);
    assert(splitting_rap.size() == splitting.size());
    for (int i = 0; i < S->n_rows; i++)
    {
        assert(splitting[i] == splitting_rap[i]);
    }


    // Test CLJP Splittings
    f = fopen("../../../../test_data/weights.txt", "r");
    std::vector<double> weights(S->n_rows);
    for (int i = 0; i < S->n_rows; i++)
    {
        fscanf(f, "%lf\n", &weights[i]);
    }
    fclose(f);
    split_cljp(S, splitting_rap, weights.data());
    f = fopen("../../../../test_data/rss_cf0", "r");
    for (int i = 0; i < S->n_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);
    assert(splitting_rap.size() == splitting.size());
    for (int i = 0; i < S->n_rows; i++)
    {
        assert(splitting[i] == splitting_rap[i]);
    }

    delete S;



    // TEST LAPLACIAN SPLITTINGS ON LEVEL 1 
    S = readMatrix("../../../../test_data/rss_S1.mtx", 0);
    splitting.resize(S->n_rows);;

    // Test RugeStuben Splitting
    split_rs(S, splitting_rap);
    f = fopen("../../../../test_data/rss_cf1_rs", "r");
    for (int i = 0; i < S->n_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);
    assert(splitting_rap.size() == splitting.size());
    for (int i = 0; i < S->n_rows; i++)
    {
        assert(splitting[i] == splitting_rap[i]);
    }


    // Test CLJP Splittings
    f = fopen("../../../../test_data/weights.txt", "r");
    weights.resize(S->n_rows);
    for (int i = 0; i < S->n_rows; i++)
    {
        fscanf(f, "%lf\n", &weights[i]);
    }
    fclose(f);
    split_cljp(S, splitting_rap, weights.data());
    f = fopen("../../../../test_data/rss_cf1", "r");
    for (int i = 0; i < S->n_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);
    assert(splitting_rap.size() == splitting.size());
    for (int i = 0; i < S->n_rows; i++)
    {
        assert(splitting[i] == splitting_rap[i]);
    }

    delete S;


    return 0;
}


