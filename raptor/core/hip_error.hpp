// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
// Reference: Communication-Performance-Tutorial/cg-solver-example

#ifndef RAPTOR_CORE_HIP_ERROR_HPP
#define RAPTOR_CORE_HIP_ERROR_HPP

#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>

#ifdef USING_MPI
#include <mpi.h>
#define RAPTOR_HIP_ABORT() MPI_Abort(MPI_COMM_WORLD, -1)
#else
#define RAPTOR_HIP_ABORT() std::abort()
#endif

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t status = (call);                                             \
        if (status != hipSuccess) {                                            \
            std::fprintf(stderr, "HIP error '%s' at %s:%d\n",                    \
                    hipGetErrorString(status), __FILE__, __LINE__);             \
            RAPTOR_HIP_ABORT();                                                 \
        }                                                                      \
    } while (0)

#define ROCSPARSE_CHECK(call)                                                   \
    do {                                                                       \
        rocsparse_status status = (call);                                       \
        if (status != rocsparse_status_success) {                               \
            std::fprintf(stderr, "rocSPARSE error %d at %s:%d\n",                \
                    static_cast<int>(status), __FILE__, __LINE__);              \
            RAPTOR_HIP_ABORT();                                                 \
        }                                                                      \
    } while (0)

#define ROCBLAS_CHECK(call)                                                     \
    do {                                                                       \
        rocblas_status status = (call);                                         \
        if (status != rocblas_status_success) {                                 \
            std::fprintf(stderr, "rocBLAS error %d at %s:%d\n",                  \
                    static_cast<int>(status), __FILE__, __LINE__);              \
            RAPTOR_HIP_ABORT();                                                 \
        }                                                                      \
    } while (0)

#endif // RAPTOR_CORE_HIP_ERROR_HPP
