// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor/raptor.hpp"
#include "raptor/core/hip_error.hpp"
#include "raptor_mfem_config.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int local_result = RUN_ALL_TESTS();
    int result;
    MPI_Allreduce(&local_result, &result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}

// 2D MFEM elasticity example; correctness check against CPU reference
TEST(ParHipCGTest, MFEMLinearElasticity2DMatchesCPU)
{
    int rank, devices = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    hipError_t device_status = hipGetDeviceCount(&devices);
    int available = device_status == hipSuccess && devices > 0;
    int all_available;
    MPI_Allreduce(&available, &all_available, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all_available)
        GTEST_SKIP() << "An AMD GPU must be accessible on every MPI rank.";

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
            MPI_INFO_NULL, &node_comm);
    int local_rank;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);
    HIP_CHECK(hipSetDevice(local_rank % devices));

    // create the 2D MFEM elasticity example
    ParVector initial_x, b;
    int num_variables;
    std::unique_ptr<ParCSRMatrix> A(mfem_linear_elasticity(initial_x, b,
            &num_variables, MFEM_MESH_DIR "/beam-tri.mesh", 1, 1, 1));
    ASSERT_EQ(num_variables, 2);
    initial_x.set_const_value(0.0);
    ParVector cpu_x(initial_x), gpu_x(initial_x);
    double b_norm = b.norm(2);
    ASSERT_GT(b_norm, 0.0);
    ASSERT_TRUE(std::isfinite(b_norm));

    // CPU run
    const double tol = 1e-10;
    const int max_iter = 2000;
    std::vector<double> cpu_residuals, gpu_residuals;
    CG(A.get(), cpu_x, b, cpu_residuals, tol, max_iter);

    rocsparse_handle sparse_handle;
    rocblas_handle blas_handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&sparse_handle));
    ROCBLAS_CHECK(rocblas_create_handle(&blas_handle));
    ParHipCSRMatrix device_A;
    upload_hip_matrix(*A, device_A, sparse_handle);

    // allocate vector GPU storage
    double *device_x, *device_b;
    std::size_t bytes = static_cast<std::size_t>(b.local_n) * sizeof(double);
    HIP_CHECK(hipMalloc(&device_x, bytes));
    HIP_CHECK(hipMalloc(&device_b, bytes));
    if (b.local_n)
    {
        HIP_CHECK(hipMemcpy(device_x, initial_x.local.data(), b.local_n * sizeof(double),
                hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(device_b, b.local.data(), b.local_n * sizeof(double),
                hipMemcpyHostToDevice));
    }
    CG(&device_A, device_x, device_b, gpu_residuals,
            sparse_handle, blas_handle, tol, max_iter);
    if (b.local_n) // copy x back to host for correctness check
        HIP_CHECK(hipMemcpy(gpu_x.local.data(), device_x, b.local_n * sizeof(double),
                hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(device_x));
    HIP_CHECK(hipFree(device_b));
    free_hip_matrix(device_A);
    ROCBLAS_CHECK(rocblas_destroy_handle(blas_handle));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(sparse_handle));

    // Check the GPU solution against the original system and CPU solution
    ParVector residual(b.global_n, b.local_n);
    A->residual(gpu_x, b, residual);
    double gpu_relative_residual = residual.norm(2) / b_norm;
    ParVector difference = gpu_x - cpu_x;
    double cpu_norm = cpu_x.norm(2);
    ASSERT_TRUE(std::isfinite(cpu_norm));
    ASSERT_GT(cpu_norm, 0.0);
    double relative_solution_difference = difference.norm(2) / cpu_norm;

    EXPECT_LE(gpu_relative_residual, tol);
    EXPECT_LE(relative_solution_difference, 1e-6);

    if (rank == 0)
        std::printf("2D MFEM elasticity CG: CPU/GPU iterations=%zu/%zu, "
                    "GPU true relative residual=%.6e, relative solution difference=%.6e\n",
                cpu_residuals.size() - 1, gpu_residuals.size() - 1,
                gpu_relative_residual, relative_solution_difference);
}
