// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "lapack_wrapper.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>


extern "C" void dgesvd_(char* jobu, char* jobvt, int* m, int* n,
        double* a, int* lda, double* singular_values, double* u, int* ldu,
        double* vt, int* ldvt, double* work, int* lwork, int* info);

namespace raptor
{

std::vector<double> pinv(const double* matrix, int rows, int cols,
        double rcond)
{
    if (!matrix || rows <= 0 || cols <= 0)
    {
        throw std::invalid_argument("Invalid pseudoinverse input");
    }

    const int rank = std::min(rows, cols);
    int lda = rows;
    int ldu = rows;
    int ldvt = rank;

    // LAPACK uses column-major storage
    std::vector<double> a_col(static_cast<std::size_t>(rows) * cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            a_col[j * rows + i] = matrix[i * cols + j];
        }
    }

    std::vector<double> singular_values(rank);
    std::vector<double> u(static_cast<std::size_t>(rows) * rank);
    std::vector<double> vt(static_cast<std::size_t>(rank) * cols);

    char jobu = 'S';
    char jobvt = 'S';
    int m = rows;
    int n = cols;
    int info = 0;
    int lwork = -1;
    double work_query = 0.0;

    dgesvd_(&jobu, &jobvt, &m, &n, a_col.data(), &lda,
            singular_values.data(), u.data(), &ldu, vt.data(), &ldvt,
            &work_query, &lwork, &info);

    if (info != 0)
    {
        throw std::runtime_error("LAPACK dgesvd workspace query failed");
    }
    lwork = std::max(1, static_cast<int>(work_query));
    std::vector<double> work(lwork);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            a_col[j * rows + i] = matrix[i * cols + j];
        }
    }

    dgesvd_(&jobu, &jobvt, &m, &n, a_col.data(), &lda,
            singular_values.data(), u.data(), &ldu, vt.data(), &ldvt,
            work.data(), &lwork, &info);

    if (info != 0)
    {
        throw std::runtime_error("LAPACK dgesvd failed");
    }
    const double relative_cutoff = rcond < 0.0
        ? std::max(rows, cols) * std::numeric_limits<double>::epsilon()
        : rcond;
    const double cutoff = relative_cutoff * singular_values[0];

    // Convert the result back to row-major
    std::vector<double> result(static_cast<std::size_t>(cols) * rows, 0.0);
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            for (int k = 0; k < rank; k++)
            {
                if (singular_values[k] > cutoff)
                {
                    result[i * rows + j] +=
                            vt[k + i * ldvt] *
                            u[j + k * ldu] /
                            singular_values[k];
                }
            }
        }
    }

    return result;
}

} // namespace raptor
