#ifndef PARMATRIX_HPP
#define PARMATRIX_HPP


#include <mpi.h>
#include <math.h>
#include <Eigen/Dense>
using Eigen::VectorXd;

#include "Matrix.hpp"


class ParMatrix
{
    public:
        ParMatrix(_GlobRows, _GlobCols, Matrix* _diag, Matrix* _offd);
        ParMatrix(ParMatrix* A);
        ~ParMatrix();

    //private:
        int globalRows;
        int globalCols;
        Matrix* diag;
	    Matrix* offd;
};
#endif
