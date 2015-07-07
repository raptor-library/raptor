// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_matrix.hpp"

ParMatrix::ParMatrix(index_t _globalRows, index_t _globalCols, Matrix* _diag, Matrix* _offd)
{
    this->globalRows = _globalRows;
    this->globalCols = _globalCols;
    this->diag = _diag;
    this->offd = _offd;
}

ParMatrix::ParMatrix(ParMatrix* A)
{
    this->globalRows = A->global_rows;
    this->globalCols = A->global_cols;
    this->diag = A->diag; // should we mark as not owning?
    this->offd = A->offd;
}

ParMatrix::~ParMatrix()
{
    delete this->diag;
    delete this->offd;
}
