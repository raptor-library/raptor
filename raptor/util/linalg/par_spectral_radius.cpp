// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_spectral_radius.hpp"
#include "raptor/core/matrix_traits.hpp"
#include <stdexcept>
#include <cmath>

namespace raptor{

double power_iteration(ParBSRMatrix* A,
    const std::vector<double>& block_diag_inv,
    int max_iterations, double tol)
{

    const int scalar_global_row = total_global_num_rows(*A);
    const int scalar_local_row = total_local_num_rows(*A);
    const int scalar_global_col = total_global_num_cols(*A);
    const int scalar_local_col = total_local_num_cols(*A);
    const BSRMatrix* A_on = static_cast<BSRMatrix*>(A->on_proc);
    const int block_row = A_on->b_rows;
    const int block_col = A_on->b_cols;
    const int block_size = A_on->b_size;

    if (block_diag_inv.size() != A->local_num_rows*block_size)
    {
        throw std::invalid_argument("D^-1 and A sizes do not match");
    }

    if (block_row != block_col ||
        scalar_global_col != scalar_global_row ||
        scalar_local_col != scalar_local_row)
    {
        throw std::invalid_argument("Power iteration requires a square ParBSRMatrix");
    }

    ParVector q(scalar_global_col, scalar_local_col); 
    q.set_rand_values();
    auto qnorm = q.norm(2);
    if (qnorm <= 0 && !std::isfinite(qnorm))
    {   
        std::runtime_error("invalid random vector initialization in power iteration");
    }
    q.scale(1.0/qnorm);
    double lambda = 0.0;

    for (int i = 0; i < max_iterations; i++)
    {
        ParVector y(scalar_global_row, scalar_local_row);
        A->mult(q, y);

        // z = D-1 * y
        ParVector z(scalar_global_row, scalar_local_row);
        for (int j = 0; j < A->local_num_rows; j++)
        {
            const double* pinv_block = block_diag_inv.data() + j*block_size;
            for (int br = 0; br < block_row; br++)
            { 
                double val = 0.0;
                for (int bc = 0; bc < block_col; bc++)
                {
                    val += y[block_col*j + bc]* pinv_block[br*block_col+bc];
                }
                z[block_row*j+br] = val;
            }
        }

        // rayleigh quotient
        lambda = static_cast<double>(q.inner_product(z));
        
        ParVector x(z);

        auto xnorm = x.norm(2);
        if (xnorm <= 0 && !std::isfinite(xnorm))
        {   
            std::runtime_error("invalid vector normalization in power iteration");
        }
        x.scale(1.0/xnorm);

        // check D-1 A q - lambda q 
        ParVector tmp(z);
        tmp.axpy(q, -lambda);
        if (tmp.norm(2)/xnorm < tol) break;

        q = x;
    }

    return std::abs(lambda);

}

}
