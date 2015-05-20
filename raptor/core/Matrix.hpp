#ifndef CSRMATRIX_H
#define CSR_MATRIX_H

#include <Eigen/Dense>
using Eigen::VectorXd;

class Matrix {
    public:
        virtual void spmv(VectorXd* x, VectorXd* y, double alpha, double beta);
};

class CSR_Matrix : public Matrix {

    public:
        CSR_MATRIX(VectorXd* _rowPtr, VectorXd* _cols, VectorXd* _data, 
                    int _nRows, int _nCols)
            {
                rowPtr = _rowPtr; cols = _cols;   data = _data; 
                nRows = _nRows;   nCols = _nCols;
            }

        void spmv(VectorXd* x, VectorXd* y, double alpha, double beta)
        {
            // y = \alpha * Ax + \beta * b
            double sum;
            for(int i = 0; i < nRows; i++) {
                sum = 0;
                for(int jj = rowPtr[i]; jj < rowPtr[i+1]; jj++) {
                    sum += data[jj] * x[cols[jj]];
                }
                y[i] = alpha * sum + beta * y[i];
            }
        }
    private:
        VectorXd* rowPtr;
        VectorXd* cols;
        VectorXd* data;
        int nRows;
        int nCols;
};





