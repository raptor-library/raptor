// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "par_hip_cg.hpp"
#include "raptor/core/hip_error.hpp"
#include "raptor/core/mpi_types.hpp"
#include "raptor/util/linalg/par_hip_spmv.hpp"
#include "raptor/util/linalg/par_hip_vector.hpp"
#include <cmath>

namespace raptor
{
void CG(ParHipCSRMatrix* A, double* x, const double* b, std::vector<double>& res,
        rocsparse_handle sparse_handle, rocblas_handle blas_handle,
        double tol, int max_iter, double* comm_t)
{
    int rank;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);

    double *r, *p, *Ap;
    int n = A->on_proc.n_rows;
    int iter, recompute_r;
    data_t alpha, beta;
    data_t rr_inner, next_inner, App_inner;
    double norm_r;
    double b_norm = par_hip_norm(b, n, blas_handle);
    if (b_norm < zero_tol) b_norm = 1.0;

    if (max_iter <= 0)
    {
        max_iter = static_cast<int>(1.3 * A->global_num_rows) + 2;
    }

    // allocate device vector
    std::size_t bytes = static_cast<std::size_t>(n) * sizeof(double);
    HIP_CHECK(hipMalloc(&r, bytes));
    HIP_CHECK(hipMalloc(&p, bytes));
    HIP_CHECK(hipMalloc(&Ap, bytes));

    // r0 = b - A*x0; p0 = r0
    par_hip_residual(*A, x, b, r, sparse_handle);
    hip_copy(p, r, n);

if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
    rr_inner = par_hip_inner_product(r, r, n, blas_handle);
if (comm_t) *comm_t += RAPtor_MPI_Wtime();
    norm_r = std::sqrt(rr_inner);
    res.emplace_back(norm_r / b_norm);

    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    recompute_r = 8;
    iter = 0;

    // Main CG Loop
    while (norm_r > tol && iter < max_iter)
    {
        // alpha_i = (r_i, r_i) / (A*p_i, p_i)
        par_hip_spmv(1.0, *A, p, 0.0, Ap, sparse_handle);
if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
        App_inner = par_hip_inner_product(Ap, p, n, blas_handle);
if (comm_t) *comm_t += RAPtor_MPI_Wtime();
        if (App_inner < 0.0)
        {
            if (rank == 0)
            {
                std::printf("Indefinite matrix detected in CG! Aborting...\n");
            }
            RAPTOR_HIP_ABORT();
        }
        alpha = rr_inner / App_inner;

        hip_axpy(alpha, x, p, n, blas_handle);

        // x_{i+1} = x_i + alpha_i * p_i
        if ((iter % recompute_r) && iter > 0)
        {
            hip_axpy(-alpha, r, Ap, n, blas_handle);
        }
        else
        {
            par_hip_residual(*A, x, b, r, sparse_handle);
        }

        // beta_i = (r_{i+1}, r_{i+1}) / (r_i, r_i)
if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
        next_inner = par_hip_inner_product(r, r, n, blas_handle);
if (comm_t) *comm_t += RAPtor_MPI_Wtime();
        beta = next_inner / rr_inner;

        // p_{i+1} = r_{i+1} + beta_i * p_i
        hip_scale(beta, p, n, blas_handle);
        hip_axpy(1.0, p, r, n, blas_handle);

        // Update next inner product
        rr_inner = next_inner;
        norm_r = std::sqrt(rr_inner);
        res.emplace_back(norm_r / b_norm);
        iter++;
    }

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(r));
    HIP_CHECK(hipFree(p));
    HIP_CHECK(hipFree(Ap));

    if (rank == 0)
    {
        if (iter == max_iter)
        {
            std::printf("Max Iterations Reached.\n");
        }
        else
        {
            std::printf("%d Iteration required to converge\n", iter);
        }
        std::printf("2 Norm of Residual: %lg\n\n", norm_r);
    }
}
} // namespace raptor
