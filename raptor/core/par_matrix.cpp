// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_matrix.hpp"

ParMatrix::ParMatrix(index_t _globalRows, index_t _globalCols, Matrix* _diag, Matrix* _offd)
{
    this->global_rows = _globalRows;
    this->global_cols = _globalCols;
    this->diag = _diag;
    this->offd = _offd;
}

ParMatrix::ParMatrix(ParMatrix* A)
{
    this->global_rows = A->global_rows;
    this->global_cols = A->global_cols;
    this->diag = A->diag; // should we mark as not owning? (we should think about move semantics or if people really love pointers we could use smart pointers).
    this->offd = A->offd;
}

ParMatrix::ParMatrix()
{
    this->local_rows = 0;
    this->local_cols = 0;
    this->offd_num_cols = 0;
}

ParMatrix::~ParMatrix()
{
    if (this->offd_num_cols)
    {
        delete this->offd;
    }
    if (this->local_rows)
    {
        delete this->diag;
        delete this->comm;
    }
    //delete[] this-> global_row_starts;
}
