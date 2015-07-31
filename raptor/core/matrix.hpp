// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_MATRIX_H
#define RAPTOR_CORE_MATRIX_H

#include <Eigen/Sparse>

#include "types.hpp"

// TODO: Do not hard code RowMajor as 1; use the enum that exists somewhwere
typedef Eigen::SparseMatrix<data_t, 1> CSRMat;
typedef Eigen::SparseMatrix<data_t, 0> CSCMat;
typedef Eigen::Triplet<data_t> Triplet;


template<int MatType>
class Matrix
{
public:
    Eigen::SparseMatrix<data_t, MatType, index_t>* m;
    index_t n_rows;
    index_t n_cols;
    unsigned long nnz;
};

class CSR_Matrix : public Matrix <1>
{

public:
    CSR_Matrix(std::vector<Triplet>* _triplets, index_t _nrows, index_t _ncols)
    {
        m = new CSRMat (_nrows, _ncols);
        m->setFromTriplets(_triplets->begin(), _triplets->end());
        n_rows = _nrows;
        n_cols = _ncols;
        nnz = _triplets->size();
    }
    CSR_Matrix(index_t* I, index_t* J, data_t* data, index_t _nrows, index_t _ncols, index_t _nnz, format_t format = CSR)
    {
        m = new CSRMat (_nrows, _ncols);
        std::vector<Triplet> _triplets(_nnz);

        if (format == CSR)
        {
            // Assumes CSR Format
            index_t ctr = 0;
            for (index_t i = 0; i < _nrows; i++)
            {
                for (index_t j = I[i]; j < I[i+1]; j++)
                {
                    _triplets[ctr++] = (Triplet(i, J[j], data[j]));
                }
            }
        }
        else if (format == COO)
        {
            // Assumes COO format
            index_t ctr = 0;
            for (int i = 0; i < _nnz; i++)
            {
                _triplets[ctr++] = (Triplet(I[i], J[i], data[i]));
            }
        }

        m->setFromTriplets(_triplets.begin(), _triplets.end());
        n_rows = _nrows;
        n_cols = _ncols;
        nnz = _nnz;

    }
    ~CSR_Matrix() { delete m; }
};


class CSC_Matrix : public Matrix <0>
{

public:
    CSC_Matrix(std::vector<Triplet>* _triplets, index_t _nrows, index_t _ncols)
    {
        m = new CSCMat (_nrows, _ncols);
        m->setFromTriplets(_triplets->begin(), _triplets->end());
        n_rows = _nrows;
        n_cols = _ncols;
        nnz = _triplets->size();
    }
    CSC_Matrix(index_t* I, index_t* J, data_t* data, index_t _nrows, index_t _ncols, index_t _nnz, format_t format = CSR)
    {
        m = new CSCMat (_nrows, _ncols);
        std::vector<Triplet> _triplets(_nnz);

        if (format == CSR)
        {
            // Assumes CSR Format
            index_t ctr = 0;
            for (index_t i = 0; i < _nrows; i++)
            {
                for (index_t j = I[i]; j < I[i+1]; j++)
                {
                    _triplets[ctr++] = (Triplet(i, J[j], data[j]));
                }
            }
        }
        else if (format == COO)
        {
            // Assumes COO format
            index_t ctr = 0;
            for (int i = 0; i < _nnz; i++)
            {
                _triplets[ctr++] = (Triplet(I[i], J[i], data[i]));
            }
        }

        m->setFromTriplets(_triplets.begin(), _triplets.end());
        n_rows = _nrows;
        n_cols = _ncols;
        nnz = _nnz;

    }
    ~CSC_Matrix() { delete m; }
};

// TODO -- ADD COO_Matrix ... cannot find a Eigen implementation for this...

#endif
