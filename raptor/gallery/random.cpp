// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.

#include "random.hpp"

namespace raptor {
CSRMatrix* random(int rows, int cols, int nnz_per_row)
{
    CSRMatrix* A;
    COOMatrix* Atmp = new COOMatrix(rows, cols, nnz_per_row);

    int nnz = nnz_per_row * rows;
    for (int i = 0; i < nnz; i++)
    {
        Atmp->idx1.emplace_back(rand() % rows);
        Atmp->idx2.emplace_back(rand() % cols);
        Atmp->vals.emplace_back(1.0);
    }
    Atmp->nnz = nnz;

    A = Atmp->to_CSR();
    delete Atmp;

    return A;

}

}
