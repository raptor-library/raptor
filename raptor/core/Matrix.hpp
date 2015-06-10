#ifndef CSRMATRIX_H
#define CSR_MATRIX_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
typedef Eigen::SparseMatrix<double, RowMajor> SpMat;
typedef Eigen::Triplet<double> T;

class Matrix {
};

class CSR_Matrix : public Matrix {

    public:
        CSR_Matrix(std::vector<T> _triplets, int _nRows, int _nCols)
            {
                m = new SpMat (_nRows, _nCols);
                m->setFromTriplets(_triplets.begin(), _triplets.end());
                nRows = _nRows;   nCols = _nCols;
            }

        ~CSR_Matrix() { delete m; }
    private:
        SpMat* m;
        int nRows;
        int nCols;
};





