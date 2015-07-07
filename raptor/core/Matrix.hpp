// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_MATRIX_H
#define RAPTOR_CORE_MATRIX_H

#include <Eigen/Sparse>

#include "Types.hpp"

// TODO: Do not hard code RowMajor as 1; use the enum that exists somewhwere
typedef Eigen::SparseMatrix<data_t, 1> SpMat;
//typedef Eigen::SparseMatrix<data_t, RowMajor> SpMat;
typedef Eigen::Triplet<data_t> Triplet;

class Matrix
{
public:
    SpMat* m;
    index_t nRows;
    index_t nCols;
    unsigned long nnz;
};

class CSR_Matrix : public Matrix
//class Matrix
{

public:
    CSR_Matrix(std::vector<Triplet>* _triplets, index_t _nRows, index_t _nCols)
    {
        m = new SpMat (_nRows, _nCols);
        m->setFromTriplets(_triplets->begin(), _triplets->end());
        nRows = _nRows;
        nCols = _nCols;
        nnz = _triplets->size();
    }
    CSR_Matrix(index_t* I, index_t* J, data_t* data, index_t _nRows, index_t _nCols, unsigned long _nnz)
    {
        m = new SpMat (_nRows, _nCols);
        std::vector<Triplet> _triplets(_nnz);

        // assumes COO format
        //index_t ctr = 0;
        //for (int i = 0; i < _nnz; i++)
        //{
        //    _triplets[ctr++] = (Triplet(I[i], J[i], data[i]));
        //}

        //Assumes CSR Format
        index_t ctr = 0;
        for (index_t i = 0; i < _nRows; i++)
        {
            for (index_t j = I[i]; j < I[i+1]; j++)
            {
                _triplets[ctr++] = (Triplet(i, J[j], data[j]));
            }
        }

        m->setFromTriplets(_triplets.begin(), _triplets.end());
        nRows = _nRows;
        nCols = _nCols;
        nnz = _nnz;

    }
    ~CSR_Matrix() { delete m; }
};

#endif
