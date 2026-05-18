// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "par_relax.hpp"

namespace raptor {


/**************************************************************
 *****   Hybrid Gauss-Seidel / Jacobi Parallel Relaxation
 **************************************************************
 ***** Performs Jacobi along the off-diagonal block and
 ***** symmetric Gauss-Seidel along the diagonal block
 ***** The tmp array is used as a place-holder, but the result
 ***** is returned put in the x-vector. 
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to relax over
 ***** x : data_t*
 *****    Vector to be relaxed, will contain result
 ***** y : data_t*
 *****    Right hand side vector
 ***** tmp: data_t*
 *****    Vector needed for inner steps
 ***** dist_x : data_t*
 *****    Vector of distant x-values recvd from other processes
 **************************************************************/

template<typename ParMatrixType>
void relax_row(ParMatrixType* A, ParVector& b, ParVector& x,
        ParVector& tmp, std::vector<double>& dist_x, double omega, int row, 
        double* rsum, double* tmp_rsum, double* D_inv, float S);

template<>
void relax_row<ParCSRMatrix>(ParCSRMatrix* A, ParVector& b, ParVector& x,
        ParVector& tmp, std::vector<double>& dist_x, double omega, int row, 
        double* rsum, double* tmp_rsum, double* D_inv, float S)
{
    double diag = 0;
    double row_sum = 0;

    int row_start = A->on_proc->idx1[row];
    int row_end = A->on_proc->idx1[row+1];
    if (row_start < row_end && A->on_proc->idx2[row_start] == row)
        diag = A->on_proc->vals[row_start++];
    else                
        return;

    calc_row_sum((CSRMatrix*)A->on_proc, tmp.local.data(), row_start, row_end, &row_sum, row, S);
    
    row_start = A->off_proc->idx1[row];
    row_end = A->off_proc->idx1[row+1];
    calc_row_sum((CSRMatrix*)A->off_proc, dist_x.data(), row_start, row_end,
            &row_sum, row, S);

    update_row((CSRMatrix*)A->on_proc, &(x[row]), &(b[row]), &(tmp[row]), &diag, &row_sum,
            omega, NULL, S);
}

template<>
void relax_row<ParBSRMatrix>(ParBSRMatrix* A, ParVector& b, ParVector& x,
        ParVector& tmp, std::vector<double>& dist_x, double omega, int row, 
        double* rsum, double* tmp_rsum, double* D_inv, float S)
{
    int n = A->on_proc->b_rows;

    int row_start = A->on_proc->idx1[row];
    int row_end = A->on_proc->idx1[row+1];
    if (row_start == row_end) return;

    memset(rsum, 0, n*sizeof(double));

    calc_row_sum((BSRMatrix*)A->on_proc, tmp.local.data(), row_start, row_end, 
            rsum, row, S);

    row_start = A->off_proc->idx1[row];
    row_end = A->off_proc->idx1[row+1];
    calc_row_sum((BSRMatrix*)A->off_proc, dist_x.data(), row_start, row_end, 
            rsum, row, S);

    update_row((BSRMatrix*)A->on_proc, &(x[row*n]), &(b[row*n]), &(tmp[row*n]),
            &(D_inv[row*A->on_proc->b_size]), rsum, omega, tmp_rsum, S);
}

template <typename ParMatrixType>
void relax_incr(ParMatrixType* A, ParVector& b, ParVector& x, ParVector& tmp,
        std::vector<double>& dist_x, double omega, double* D_inv = NULL, 
        double* rsum = NULL, double* tmp_rsum = NULL, int* points = NULL, 
        int points_len = 0, float S = 1.0)
{
    for (int row = 0; row < A->local_num_rows; row++)
    {    
        relax_row(A, b, x, tmp, dist_x, omega, row, rsum, tmp_rsum, D_inv, S);
    }

}

template <typename ParMatrixType>
void relax_points(ParMatrixType* A, ParVector& b, ParVector& x, ParVector& tmp,
        std::vector<double>& dist_x, double omega, double* D_inv = NULL, 
        double* rsum = NULL, double* tmp_rsum = NULL, int* points = NULL, 
        int points_len = 0, float S = 1.0)
{
    int idx = 0;
    while (idx < points_len)
    {
        int row = points[idx++];
        relax_row(A, b, x, tmp, dist_x, omega, row, rsum, tmp_rsum, D_inv, S);
    }
}

template <typename ParMatrixType, typename SweepType, typename CopyType>
void relax(SweepType relax_sweep, CopyType copy, ParMatrixType* A, 
        ParVector& b, ParVector& x, ParVector& tmp, CommPkg* comm,
        int num_sweeps, double omega, double* D_inv = NULL, 
        int* points = NULL, int points_len = 0, float S = 1.0)
{
    A->on_proc->sort();
    A->off_proc->sort();
    A->on_proc->move_diag();

    double* rsum = new double[A->on_proc->b_rows];
    double* tmp_rsum = new double[A->on_proc->b_rows];

    for (int iter = 0; iter < num_sweeps; iter++)
    {
        comm->communicate(x);
        std::vector<double>& dist_x = comm->get_buffer<double>();

        copy(tmp.local, x.local);

        relax_sweep(A, b, x, tmp, dist_x, omega, D_inv, rsum, tmp_rsum, 
                points, points_len, S);
    }

    delete[] rsum;
    delete[] tmp_rsum;
}
    

/**************************************************************
 *****  Relaxation Method 
 **************************************************************
 ***** Performs jacobi along the diagonal block of the matrix
 *****
 ***** Parameters
 ***** -------------
 ***** l: Level*
 *****    Level in hierarchy to be relaxed
 ***** num_sweeps : int
 *****    Number of relaxation sweeps to perform
 **************************************************************/
void set_comm(ParCSRMatrix* A, CommPkg** comm, bool tap)
{
    if (tap)
    {
        if (!A->tap_comm) 
        {
            A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map,
                    A->on_proc_column_map);
        }
        *comm = A->tap_comm;
    }
    else
    {
        if (!A->comm) 
        {
            A->comm = new ParComm(A->partition, A->off_proc_column_map,
                    A->on_proc_column_map);
        }
        *comm = A->comm;
    }
}

template <typename ParMatrixType>
void jacobi(ParMatrixType* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega, bool tap, double* D_inv,int* points,
        int points_len, float S)
{
    CommPkg* comm;
    set_comm(A, &comm, tap);
    auto F = relax_incr<ParMatrixType>;
    if (points_len > 0 && points != NULL)
        F = relax_points<ParMatrixType>;
    relax(F, jacobi_copy, A, b, x, tmp, comm, num_sweeps, omega, D_inv, 
            points, points_len, S);
}

template <typename ParMatrixType>
void sor(ParMatrixType* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega, bool tap, double* D_inv, int* points,
        int points_len, float S)
{
    CommPkg* comm;
    set_comm(A, &comm, tap);
    auto F = relax_incr<ParMatrixType>;
    if (points_len > 0 && points != NULL)
        F = relax_points<ParMatrixType>;
    relax(F, sor_copy, A, b, x, x, comm, num_sweeps, omega, D_inv, 
            points, points_len, S);
}


template void jacobi(ParCSRMatrix*, ParVector&, ParVector&, ParVector&, 
        int, double, bool, double*, int*, int, float);
template void jacobi(ParBSRMatrix*, ParVector&, ParVector&, ParVector&, 
        int, double, bool, double*, int*, int, float);


template void sor(ParCSRMatrix*, ParVector&, ParVector&, ParVector&, 
        int, double, bool, double*, int*, int, float);
template void sor(ParBSRMatrix*, ParVector&, ParVector&, ParVector&, 
        int, double, bool, double*, int*, int, float);



}
