// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "prolongation.hpp"

// Assuming weighting = local (not getting approx spectral radius)
void jacobi_prolongation(CSRMatrix& A, CSCMatrix& T, CSCMatrix& P,
        double omega, int num_smooth_steps)
{
    CSRMatrix scaled_A(&A);
    P.copy(&T);

    // Get absolute row sum for each row
    int row_start, row_end;
    std::vector<double> inv_sums(A.n_rows, 0);
    double row_sum = 0;
    for (int row = 0; row < A.n_rows; row++)
    {
        row_start = A.idx1[row];
        row_end = A.idx1[row+1];
        row_sum = 0.0;
        for (int j = row_start; j < row_end; j++)
        {
            row_sum += fabs(A.vals[j]);
        }

        if (row_sum)
        {
            inv_sums[row] = (1.0 / fabs(row_sum)) * omega;
        }

        for (int j = row_start; j < row_end; j++)
        {
            scaled_A.vals[j] *= inv_sums[row];
        }
    }

    // P = P - (scaled_A*P)
    CSCMatrix P_tmp;
    for (int i = 0; i < num_smooth_steps; i++)
    {
        // TODO-- need to implement this to also return CSC
        scaled_A.mult(P, &P_tmp);
        P.subtract(P_tmp, T);
        P.copy(&T);
    }
}

