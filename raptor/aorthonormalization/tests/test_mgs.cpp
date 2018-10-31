#include <assert.h>

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "aorthonormalization/mgs.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{    
    int num_vectors = 4;
    int grid[2] = {50, 50};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    CSRMatrix* A = stencil_grid(stencil, grid, 2);
    aligned_vector<Vector> W;
    aligned_vector<aligned_vector<Vector>> P_list;

    for (int i = 0; i < num_vectors; i++) {
        Vector w(A->n_rows);
        w.set_rand_values();
        W.push_back(w);
    }

    for (int i = 0; i < 2; i++) {
        aligned_vector<Vector> P_sublist;
        for (int j = 0; j < num_vectors; j++) {
            Vector p(A->n_rows);
            p.set_rand_values();
            P_sublist.push_back(p);
        }
        P_list.push_back(P_sublist);
    }

    MGS(A, W, P_list);

    // Insert check 

    MGS(A, W);

    // Check for correctness
    double one;
    Vector Aw(A->n_rows);

    for (int i = 0; i < num_vectors; i++) {
        A->mult(W[i], Aw);
        one = Aw.inner_product(W[i]);
        printf("%lg\n", one);
        assert(fabs(1.0 - one) < 1e-02);
    }   
 
    delete[] stencil;
    delete A;

    return 0;
}
