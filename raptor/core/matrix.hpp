// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_MATRIX_H
#define RAPTOR_CORE_MATRIX_H

#include "types.hpp"

template <int MatType>
class Matrix
{
public:

    Matrix(index_t _nrows, index_t _ncols)
    {
        n_rows = _nrows;
        n_cols = _ncols;
        nnz = 0;
    }

    ~Matrix()
    {
        //delete[] indptr;
        //delete[] indices;
        //delete[] data;
    }

    void add_values(index_t i, index_t* j, data_t* values, index_t num_values);
    void add_value(index_t row, index_t col, data_t value);

    void resize(index_t _nrows, index_t _ncols)
    {
        n_rows = _nrows;
        n_cols = _ncols;
    }

    void finalize();

    std::vector<index_t> indptr;
    std::vector<index_t> indices;
    std::vector<data_t> data;

    index_t n_rows;
    index_t n_cols;
    index_t nnz;

};

class CSR_Matrix : public Matrix<1>
{

public:
    CSR_Matrix(index_t _nrows, index_t _ncols) : Matrix(_nrows, _ncols)
    {
        row_nnz = new index_t[_nrows];
        for (index_t i = 0; i < _nrows; i++)
        {
            row_nnz[i] = 0;
        }
        init_csr = 0;
    }
    CSR_Matrix(index_t _nrows, index_t _ncols, index_t nnz_per_row) : Matrix(_nrows, _ncols)
    {
        index_t _nnz = nnz_per_row * _nrows;
       
        indptr.resize(_nrows + 1);
        indices.resize(_nnz);
        data.resize(_nnz);

        row_nnz = new index_t[_nrows];
        for (index_t i = 0; i < _nrows; i++)
        {
            row_nnz[i] = 0;
            indptr[i] = i*nnz_per_row;
        }
        indptr[_nrows] = _nrows*nnz_per_row;
        init_csr = 1;
    }

    ~CSR_Matrix() {}

    void add_value(index_t row, index_t col, data_t value)
    {
        if (init_csr)
        {
            index_t pos = indptr[row] + row_nnz[row]++;
            indices[pos] = col;
            data[pos] = value;
            nnz++;
        }
        else
        {
            indptr.push_back(row);
            indices.push_back(col);
            data.push_back(value);
            row_nnz[row]++;
            nnz++;
        }
    }

    void add_values(index_t row, index_t* cols, data_t* values, index_t num_values)
    {
        for (index_t i = 0; i < num_values; i++)
        {
            index_t pos = indptr[row] + row_nnz[row]++;
            indices[pos] = cols[i];
            data[pos] = values[i];
        }
        nnz += num_values;
    }

    void finalize()
    {
        if (init_csr) // Remove zeros from inital CSR
        {
            index_t ctr = 0;

            for (index_t row = 0; row < n_rows; row++)
            {
                index_t row_start = indptr[row];
                index_t row_end = row_start + row_nnz[row];

                indptr[row] = ctr;

                for (index_t j = row_start; j < row_end; j++)
                {
                    indices[ctr] = indices[j];
                    data[ctr] = data[j];
                    ctr++;
                }
            }
            indptr[n_rows] = ctr;

            indices.resize(nnz);
            data.resize(nnz);
        }
        else // Convert COO to CSR
        {
            index_t indptr_a[n_rows+1];
            index_t indices_a[nnz];
            data_t data_a[nnz];

            indptr_a[0] = 0;
            for (index_t row = 0; row < n_rows; row++)
            {
                indptr_a[row+1] = indptr_a[row] + row_nnz[row];
                row_nnz[row] = 0;
            }

            for (index_t ctr = 0; ctr < nnz; ctr++)
            {
                index_t row = indptr[ctr];
                index_t pos = indptr_a[row] + row_nnz[row]++;
                indices_a[pos] = indices[ctr];
                data_a[pos] = data[ctr];
            }

            indptr.assign(indptr_a, indptr_a + n_rows+1);
            indices.assign(indices_a, indices_a + nnz);
            data.assign(data_a, data_a + nnz);
        }
        delete[] row_nnz;
    }

    index_t* row_nnz;
    index_t init_csr;
};


class CSC_Matrix : public Matrix<0>
{

public:
    CSC_Matrix(index_t _nrows, index_t _ncols) : Matrix(_nrows, _ncols)
    {
        init_csc = 0;
    }
    CSC_Matrix(index_t _nrows, index_t _ncols, index_t nnz_per_col) : Matrix(_nrows, _ncols)
    { 
        index_t _nnz = nnz_per_col * _ncols;

        indptr.resize(_ncols + 1);
        indices.resize(_nnz);
        data.resize(_nnz);
        col_nnz.resize(_ncols);

        for (index_t i = 0; i < _ncols; i++)
        {
            col_nnz[i] = 0;
            indptr[i] = i*nnz_per_col;
        }
        indptr[_ncols] = _ncols*nnz_per_col;
        init_csc = 1;
    }

    ~CSC_Matrix() {}

    void add_value(index_t row, index_t col, data_t value)
    {
        if (init_csc)
        {
            index_t pos = indptr[col] + col_nnz[col]++;
            indices[pos] = row;
            data[pos] = value;
            nnz++;
        }
        else
        {
            indptr.push_back(col);
            indices.push_back(row);
            data.push_back(value);
            if (col >= col_nnz.size())
            {
                col_nnz.push_back(1);
            }
            else
            {
                col_nnz[col]++;
            }
            nnz++;
        }
    }

    void add_values(index_t col, index_t* rows, data_t* values, index_t num_values)
    {
        for (index_t i = 0; i < num_values; i++)
        {
            index_t pos = indptr[col] + col_nnz[col]++;
            indices[pos] = rows[i];
            data[pos] = values[i];
        }
        nnz += num_values;
    }

    void finalize()
    {
        if (init_csc)
        {
            index_t ctr = 0;

            for (index_t col = 0; col < n_cols; col++)
            {
                index_t col_start = indptr[col];
                index_t col_end = col_start + col_nnz[col];

                indptr[col] = ctr;

                for (index_t j = col_start; j < col_end; j++)
                {
                    indices[ctr] = indices[j];
                    data[ctr] = data[j];
                    ctr++;
                }
            }
            indptr[n_cols] = ctr;
            
            data.resize(nnz);
            indices.resize(nnz);
        }
        else
        {
            index_t indptr_a[n_cols+1];
            index_t indices_a[nnz];
            data_t data_a[nnz];

            indptr_a[0] = 0;
            for (index_t col = 0; col < n_cols; col++)
            {
                indptr_a[col+1] = indptr_a[col] + col_nnz[col];
                col_nnz[col] = 0;
            }

            for (index_t ctr = 0; ctr < nnz; ctr++)
            {
                index_t col = indptr[ctr];
                index_t pos = indptr_a[col] + col_nnz[col]++;
                indices_a[pos] = indices[ctr];
                data_a[pos] = data[ctr];
            }

            indptr.assign(indptr_a, indptr_a + n_cols+1);
            indices.assign(indices_a, indices_a + nnz);
            data.assign(data_a, data_a + nnz);
        }
        col_nnz.clear();
    }

    std::vector<index_t> col_nnz;
    index_t init_csc;
};

//template <int MatType>
//class COO_Matrix : public Matrix<MatType>
//{

//public:
//    COO_Matrix(index_t _nrows, index_t _ncols, index_t _nnz) : Matrix(_nrows, _ncols)
//    {
//        indptr = new index_t[_nnz];
//        indices = new index_t[_nnz];
//        data = new data_t[_nnz];

//        for (index_t i = 0; i < _ncols; i++)
//        {
//            col_starts[i] = i*nnz_per_col;
//            indptr[i] = 0;
//        }
//        indptr[_ncols] = 0;
//    }

//    ~COO_Matrix() {}

//    void add_value(index_t row, index_t col, data_t value)
//    {
//        index_t pos = col_starts[col]++;
//        indices[pos] = row;
//        data[pos] = value;
//        nnz++;
//    }

//    void add_values(index_t* rows, index_t* cols, data_t* values, index_t num_values)
//    {
//        for (index_t i = 0; i < num_values; i++)
//        {
//            indptr[nnz] = rows[i];
//            indices[nnz] = cols[i];
//            data[nnz] = values[i];
//        }
//        nnz += num_values;
//    }

//    CSR_Matrix convert(Matrix<1>* A);

//    CSC_Matrix convert(Matrix<0>* A);

//    index_t* nnz_per_ind;
//};


#endif
