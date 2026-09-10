// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "par_prolongation.hpp"
#include "raptor/util/linalg/lapack_wrapper.hpp"
#include "raptor/util/linalg/par_spectral_radius.hpp"

#include <iostream>
#include <cmath>
#include <stdexcept>

#ifdef USING_OPENMP
#include <omp.h>
#endif

namespace raptor {

// Assuming weighting = local (not getting approx spectral radius)
static ParCSRMatrix* jacobi_prolongation_csr(ParCSRMatrix* A, ParCSRMatrix* T, bool tap_comm,
        double omega, int num_smooth_steps)
{
    ParCSRMatrix* AP_tmp;
    ParCSRMatrix* P_tmp;
    ParCSRMatrix* P = T->copy();
    ParCSRMatrix* scaled_A = A->copy();

    // Get absolute row sum for each row
    int row_start_on, row_end_on;
    int row_start_off, row_end_off;
    std::vector<double> inv_sums;
    if (A->local_num_rows)
    {
        inv_sums.resize(A->local_num_rows, 0);
    }

    double row_sum = 0;
    for (int row = 0; row < A->local_num_rows; row++)
    {
        row_start_on = A->on_proc->idx1[row];
        row_end_on = A->on_proc->idx1[row+1];
        row_sum = 0.0;
        for (int j = row_start_on; j < row_end_on; j++)
        {
            row_sum += fabs(A->on_proc->vals[j]);
        }
        row_start_off = A->off_proc->idx1[row];
        row_end_off = A->off_proc->idx1[row+1];
        for (int j = row_start_off; j < row_end_off; j++)
        {
            row_sum += fabs(A->off_proc->vals[j]);
        }

        if (row_sum)
        {
            inv_sums[row] = (1.0 / fabs(row_sum)) * omega;
        }

        for (int j = row_start_on; j < row_end_on; j++)
        {
            scaled_A->on_proc->vals[j] *= inv_sums[row];
        }
        for (int j = row_start_off; j < row_end_off; j++)
        {
            scaled_A->off_proc->vals[j] *= inv_sums[row];
        }
    }

    // P = P - (scaled_A*P)
    for (int i = 0; i < num_smooth_steps; i++)
    {
        if (tap_comm)
        {
            if (scaled_A->tap_mat_comm == NULL)
            {
                scaled_A->tap_mat_comm = new TAPComm(scaled_A->partition, 
                        scaled_A->off_proc_column_map,
                        scaled_A->on_proc_column_map, 
                        false, RAPtor_MPI_COMM_WORLD);
            }
            AP_tmp = scaled_A->tap_mult(P);
        }
        else
        {
            if (scaled_A->comm == NULL)
            {
                scaled_A->comm = new ParComm(scaled_A->partition, 
                        scaled_A->off_proc_column_map,
                        scaled_A->on_proc_column_map, 
                        9283, RAPtor_MPI_COMM_WORLD);
            }

            AP_tmp = scaled_A->mult(P);
        }

        P_tmp = P->subtract(AP_tmp);
        delete AP_tmp;
        delete P;
        P = P_tmp;
        P_tmp = NULL;
    }

    if (tap_comm)
    {
        P->init_tap_communicators();
    }
    else
    {
        P->comm = new ParComm(P->partition, P->off_proc_column_map, 
                P->on_proc_column_map, 9283, RAPtor_MPI_COMM_WORLD);
    }

    delete scaled_A;
    
    return P;
} // end of jacobi_prolongation_csr()

static ParBSRMatrix* jacobi_prolongation_bsr_local(ParBSRMatrix* A, ParBSRMatrix* T, bool tap_comm,
        double omega, int num_smooth_steps)
{
    int rank;
    MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);

    ParBSRMatrix* AP_tmp;
    ParBSRMatrix* P_tmp;
    ParBSRMatrix* P = T->copy();
    ParBSRMatrix* scaled_A = A->copy();
    BSRMatrix* A_on = static_cast<BSRMatrix*>(A->on_proc);
    BSRMatrix* A_off = static_cast<BSRMatrix*>(A->off_proc);
    BSRMatrix* scaled_A_on = static_cast<BSRMatrix*>(scaled_A->on_proc);
    BSRMatrix* scaled_A_off = static_cast<BSRMatrix*>(scaled_A->off_proc);

    // Get absolute row sum for each row
    int row_start_on, row_end_on;
    int row_start_off, row_end_off;
    std::vector<double> inv_sums(A->local_num_rows * A_on->b_rows, 0.0);

    for (int row = 0; row < A->local_num_rows; row++)
    {
        row_start_on = A_on->idx1[row];
        row_end_on = A_on->idx1[row+1];
        for (int j = row_start_on; j < row_end_on; j++)
        {
            const double* block = A_on->block_vals[j];
            for (int br = 0; br < A_on->b_rows; br++)
            {
                for (int bc = 0; bc < A_on->b_cols; bc++)
                {
                    // block is stored row major 
                    inv_sums[row*A_on->b_rows+br] += fabs(block[br*A_on->b_cols+bc]);
                }
            }

        }
        row_start_on = A_off->idx1[row];
        row_end_on = A_off->idx1[row+1];
        for (int j = row_start_on; j < row_end_on; j++)
        {
            const double* block = A_off->block_vals[j];
            for (int br = 0; br < A_off->b_rows; br++)
            {
                for (int bc = 0; bc < A_off->b_cols; bc++)
                {
                    // block is stored row major 
                    inv_sums[row*A_off->b_rows+br] += fabs(block[br*A_off->b_cols+bc]);
                }
            }

        }
    }

    // w * D^{-1}
    for (auto& r: inv_sums)
    {
        if (r)
        {
            r = (1.0 / r) * omega;
        }
    }
    
    // w * D^{-1} * A
    for (int row = 0; row < A->local_num_rows; row++)
    {
        row_start_on = scaled_A_on->idx1[row];
        row_end_on = scaled_A_on->idx1[row+1];
        for (int j = row_start_on; j < row_end_on; j++)
        {
            double* block = scaled_A_on->block_vals[j];
            for (int br = 0; br < scaled_A_on->b_rows; br++)
            {
                for (int bc = 0; bc < scaled_A_on->b_cols; bc++)
                {
                    block[br*scaled_A_on->b_cols+bc] *= inv_sums[row*scaled_A_on->b_rows+br];
                }
            }

        }
        row_start_on = scaled_A_off->idx1[row];
        row_end_on = scaled_A_off->idx1[row+1];
        for (int j = row_start_on; j < row_end_on; j++)
        {
            double* block = scaled_A_off->block_vals[j];
            for (int br = 0; br < scaled_A_off->b_rows; br++)
            {
                for (int bc = 0; bc < scaled_A_off->b_cols; bc++)
                {
                    // block is stored row major 
                    block[br*scaled_A_off->b_cols+bc] *= inv_sums[row*scaled_A_off->b_rows+br];
                }
            }

        }
    }

    // TODO: tap_mult does not support parBSR yet
    if (tap_comm && rank == 0)
    {
        std::cerr << "Warning: tap_mult is not implemented for ParBSRMatrix; " << "falling back to standard mult().\n";
    }

    // P = P - (scaled_A*P)
    for (int i = 0; i < num_smooth_steps; i++)
    {
        if (scaled_A->comm == NULL)
        {
            scaled_A->comm = new ParComm(scaled_A->partition, 
                    scaled_A->off_proc_column_map,
                    scaled_A->on_proc_column_map, 
                    9283, RAPtor_MPI_COMM_WORLD);
        }

        AP_tmp = scaled_A->mult(P);

        P_tmp = P->subtract(AP_tmp);
        delete AP_tmp;
        delete P;
        P = P_tmp;
        P_tmp = NULL;
    }

    if (tap_comm)
    {
        P->init_tap_communicators();
    }
    else
    {
        P->comm = new ParComm(P->partition, P->off_proc_column_map, 
                P->on_proc_column_map, 9283, RAPtor_MPI_COMM_WORLD);
    }

    delete scaled_A;
    
    return P;
} // end of jacobi_prolongation_bsr_local()

static ParBSRMatrix* jacobi_prolongation_bsr_block(ParBSRMatrix* A, ParBSRMatrix* T, bool tap_comm,
        double omega, int num_smooth_steps, bool spectral_scaling)
{
    int rank;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);

    ParBSRMatrix* AP_tmp;
    ParBSRMatrix* P_tmp;
    ParBSRMatrix* P = T->copy();
    ParBSRMatrix* scaled_A = A->copy();
    BSRMatrix* scaled_A_on = static_cast<BSRMatrix*>(scaled_A->on_proc);
    BSRMatrix* scaled_A_off = static_cast<BSRMatrix*>(scaled_A->off_proc);

    assert(scaled_A_on->b_rows == scaled_A_on->b_cols);

    int row_start, row_end;
    // D_1 
    scaled_A -> sort();
    scaled_A_on -> move_diag(); 
    std::vector<double> diag_inv(A->local_num_rows * scaled_A_on -> b_size, 0.0);
    for (int row = 0; row < A-> local_num_rows; row ++)
    {
        row_start = scaled_A_on->idx1[row];
        row_end = scaled_A_on->idx1[row+1];

        // empty row or no diagonal block that row
        if (row_start == row_end || scaled_A_on -> idx2[row_start] != row)
        {
            // filled with 0s
            continue;
        }
        
        double* diag_block = diag_inv.data() + row * scaled_A_on->b_size;

        std::copy(scaled_A_on->block_vals[row_start],scaled_A_on->block_vals[row_start] + scaled_A_on->b_size, diag_block);
    }

    for (int row = 0; row < A-> local_num_rows; row ++)
    {
        double* block = diag_inv.data() + row*scaled_A_on->b_size;
        auto pinv_block = pinv(block, scaled_A_on->b_rows, scaled_A_on->b_cols, -1.0);
        std::copy(pinv_block.begin(), pinv_block.end(), block);
    }

    // optionally apply spectral radius
    if (spectral_scaling)
    {
        double lambda = power_iteration(A, diag_inv);
        if (lambda <= 0 || !std::isfinite(lambda))
        {
            throw std::runtime_error("Power iteration produced an invalid spectral radius");
        }
        omega *= 1.0 / lambda;
    }

    // w * D^{-1} * A
    auto multiply_block_diagonal = [&](BSRMatrix* matrix)
    {
    #ifdef USING_OPENMP
    #pragma omp parallel
    #endif
        {
            std::vector<double> product(matrix->b_size);
    #ifdef USING_OPENMP
    #pragma omp for schedule(static)
    #endif
            for (int row = 0; row < matrix->n_rows; row++)
            {
                const double* pinv_block = diag_inv.data() + row * matrix->b_size;

                for (int j = matrix->idx1[row]; j < matrix->idx1[row + 1]; j++)
                {
                    double* block = matrix->block_vals[j];

                    for (int br = 0; br < matrix->b_rows; br++)
                    {
                        for (int bc = 0; bc < matrix->b_cols; bc++)
                        {
                            double value = 0.0;

                            for (int k = 0; k < matrix->b_cols; k++)
                            {
                                value += pinv_block[br * matrix->b_cols + k]* block[k * matrix->b_cols + bc];
                            }

                            product[br * matrix->b_cols + bc] = omega * value;
                        }
                    }

                    std::copy(product.begin(), product.end(), block);
                }
            }
        }
    };

    multiply_block_diagonal(scaled_A_on);
    multiply_block_diagonal(scaled_A_off);

    // tap_mult does not support parBSR yet
    if (tap_comm && rank == 0)
    {
        std::cerr << "Warning: tap_mult is not implemented for ParBSRMatrix; " << "falling back to standard mult().\n";
    }

    // P = P - (scaled_A*P)
    for (int i = 0; i < num_smooth_steps; i++)
    {
        if (scaled_A->comm == NULL)
        {
            scaled_A->comm = new ParComm(scaled_A->partition, 
                    scaled_A->off_proc_column_map,
                    scaled_A->on_proc_column_map, 
                    9283, RAPtor_MPI_COMM_WORLD);
        }

        AP_tmp = scaled_A->mult(P);

        P_tmp = P->subtract(AP_tmp);
        delete AP_tmp;
        delete P;
        P = P_tmp;
        P_tmp = NULL;
    }

    if (tap_comm)
    {
        P->init_tap_communicators();
    }
    else
    {
        P->comm = new ParComm(P->partition, P->off_proc_column_map, 
                P->on_proc_column_map, 9283, RAPtor_MPI_COMM_WORLD);
    }

    delete scaled_A;
    
    return P;
} // end of jacobi_prolongation_bsr_block()

template <class T, is_bsr_or_csr<T>>
T* jacobi_prolongation(T* A, T* tentative, bool tap_comm,
        double omega, int num_smooth_steps, prolongation_weighting weighting)
{
    if constexpr (is_bsr_v<T>)
    {
       switch (weighting)
        {
            case prolongation_weighting::local:
                return jacobi_prolongation_bsr_local(A, tentative, tap_comm, omega, num_smooth_steps);

            case prolongation_weighting::block:
                return jacobi_prolongation_bsr_block(A, tentative, tap_comm, omega, num_smooth_steps, false);

            case prolongation_weighting::block_spectral:
                return jacobi_prolongation_bsr_block(
                        A, tentative, tap_comm,
                        omega, num_smooth_steps, true);
            default:
                throw std::invalid_argument("unsupported BSR prolongation weighting");
        }
    }
    else
    {
        assert(weighting == prolongation_weighting::local);

        return jacobi_prolongation_csr(A, tentative, tap_comm, omega, num_smooth_steps);
    }
}

template ParCSRMatrix* jacobi_prolongation(ParCSRMatrix*, ParCSRMatrix*, bool, double, int, prolongation_weighting);

template ParBSRMatrix* jacobi_prolongation(ParBSRMatrix*, ParBSRMatrix*, bool, double, int, prolongation_weighting);

} // raptor namespace
