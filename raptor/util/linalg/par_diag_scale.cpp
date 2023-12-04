// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_diag_scale.hpp"

namespace raptor {

void row_scale(ParCSRMatrix* A, ParVector& rhs)
{
    int start, end;
    double scale;
    
    A->on_proc->move_diag();

    for (int i = 0; i < A->local_num_rows; i++)
    {
        start = A->on_proc->idx1[i];
        end = A->on_proc->idx1[i+1];
	    scale = 0.0;
	    if (A->on_proc->idx2[start] == i)
	    {
	        scale = 1.0 / A->on_proc->vals[start];
	    }
	    for (int j = start; j < end; j++)
	    {
	        A->on_proc->vals[j] *= scale;
	    }
	    rhs[i] *= scale;
    }
}

void diagonally_scale(ParCSRMatrix* A, ParVector& rhs, std::vector<double>& row_scales)
{
    int start, end, col;

    A->on_proc->move_diag();
    double* rhs_vals = rhs.local.data();

    if (A->local_num_rows) row_scales.resize(A->local_num_rows, 0);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        start = A->on_proc->idx1[i];
        if (A->on_proc->idx2[start] == i)
        {
            row_scales[i] = 1.0 / sqrt(fabs(A->on_proc->vals[start]));
        }
    }

    std::vector<double> off_proc_scales = A->comm->communicate(row_scales);

    for (int i = 0; i < A->local_num_rows; i++)
    {
        double row_scale = row_scales[i];

        start = A->on_proc->idx1[i];
        end = A->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A->on_proc->idx2[j];
            A->on_proc->vals[j] *= row_scale * row_scales[col];
        }

        start = A->off_proc->idx1[i];
        end = A->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A->off_proc->idx2[j];
            A->off_proc->vals[j] *= row_scale * off_proc_scales[col];
        }

        rhs_vals[i] *= row_scale;
    }
}

void diagonally_unscale(ParVector& sol, const std::vector<double>& row_scales)
{
    double* vals = sol.local.data();
    for (int i = 0; i < sol.local_n; i++)
    {
        vals[i] *= row_scales[i];
    }
}

}
