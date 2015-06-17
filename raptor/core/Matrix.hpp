#ifndef CSRMATRIX_H
#define CSR_MATRIX_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
typedef Eigen::SparseMatrix<double, RowMajor> SpMat;
typedef Eigen::Triplet<double> Triplet;

class Matrix {
};

class CSR_Matrix : public Matrix {

  public:
    CSR_Matrix(std::vector<Triplet>* _triplets, int _nRows, int _nCols)
        {
            m = new SpMat (_nRows, _nCols);
            m->setFromTriplets(_triplets->begin(), _triplets->end());
            nRows = _nRows;
            nCols = _nCols;
            nnz = _triplests->size();
        }
    CSR_Matrix(int* I, int* J, double* data, int _nRows, int _nCols, unsigned long _nnz)
    {
        m = new SpMat (_nRows, _nCols);
        std::vector<Triplet> _triplets(_nnz);

        // assumes COO format
        for (int i = 0; i < _nnz)
        {
            _triplets.push_back(Triplet(I[i], J[i], data[i]));
        }
        m->setFromTriplets(_triplets.begin(), _triplets.end());
        
        // TODO: allow for CSR format
        /*  1.) reserve approx nnz per row
        m->reserve(VectorXi::Constant(_nRows, nnzRow));
            2.) direct insertion
        for(int i = 0; i < _nRows; i++)
        {
            for(int jj = I[i]; jj < I[i+1]; jj++)
            {
                m->insert(i, J[jj]) = data[jj];
            }
        }
        */
        nRows = _nRows;
        nCols = _nCols;
        nnz = _nnz;
        
    }
    ~CSR_Matrix() { delete m; }
    
    SpMat* m;
    int nRows;
    int nCols;
    unsigned long nnz;
};





