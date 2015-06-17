/******** ParMatrix.cpp **********/
#include "ParMatrix.hpp"


ParMatrix::ParMatrix(int _globalRows, int _globalCols, Matrix* _diag)
{
    this.globalRows = _globalRows;
    this.globalCols = _globalCols;
    this.diag = _diag;
    this.off = _offd;
}

ParMatrix::ParMatrix(ParMatrix* A)
{
    this.globalRows = _globalRows;
    this.globalCols = _globalCols;
    this.diag = _diag; // should we mark as not owning?
    this.off = _offd;
}

ParMatrix::~ParMatrix()
{
    delete this->diag;
    delete this->offd;
}

