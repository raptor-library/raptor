// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_MATRIX_H
#define RAPTOR_CORE_MATRIX_H

#include "types.hpp"

template <int MatType>
class Matrix<int MatType>
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
        delete[] indptr;
        delete[] indices;
        delete[] data;
    }

    index_t add_values(index_t i, index_t* j, data_t* values, index_t num_values);

    index_t* indptr;
    index_t* indices;
    data_t* data;

    index_t n_rows;
    index_t n_cols;
    index_t nnz;

};

class CSR_Matrix : public Matrix<1>
{

public:
    CSR_Matrix(index_t _nrows, index_t _ncols, index_t nnz_per_row) : Matrix(_nrows, _ncols)
    {
        index_t _nnz = nnz_per_row * _nrows;
       
        indptr = new index_t[_nrows + 1];
        indices = new index_t[_nnz];
        data = new data_t[_nnz];

        row_starts = new index_t[_nrows];
        for (index_t i = 0; i < _nrows; i++)
        {
            row_starts[i] = i*nnz_per_row;
            indptr[i] = 0;
        }
        indptr[_nrows] = 0;
    }

    ~CSR_Matrix() {}

    index_t add_value(index_t row, index_t col, data_t value)
    {
        index_t pos = row_starts[row]++;
        indices[pos] = col;
        data[pos] = value;
        indptr[row]++;
        nnz++;
    }

    index_t add_values(index_t row, index_t* cols, data_t* values, index_t num_values)
    {
        for (index_t i = 0; i < num_values; i++)
        {
            index_t pos = row_starts[row]++;
            indices[pos] = cols[i];
            data[pos] = values[i];
        }
        indptr[row] += num_values;
        nnz += num_values;
    }

    index_t finalize()
    {
        index_t* temp_idx = new index_t[nnz];
        data_t* temp_data = new data_t[nnz];
        index_t ctr = 0;

        for (index_t row = 0; row < n_rows; row++)
        {
            index_t row_start = indptr[row];
            index_t row_end = row_starts[row];

            indptr[row] = ctr;

            for (index_t j = row_start; j < row_end; j++)
            {
                temp_idx[ctr] = indices[j];
                temp_data[ctr] = data[j];
                ctr++;
            }
        }
        indptr[n_rows] = ctr;

        delete[] indices;
        delete[] data;
        delete[] row_starts;

        indices = temp_idx;
        data = temp_data;
    }

    index_t* row_starts;

};


class CSC_Matrix : public Matrix<0>
{

public:
    CSC_Matrix(index_t _nrows, index_t _ncols, index_t nnz_per_col) : Matrix(_nrows, _ncols)
    {
        index_t _nnz = nnz_per_col * _ncols;

        indptr = new index_t[_ncols + 1];
        indices = new index_t[_nnz];
        data = new data_t[_nnz];

        col_starts = new index_t[_ncols];
        for (index_t i = 0; i < _ncols; i++)
        {
            col_starts[i] = i*nnz_per_col;
            indptr[i] = 0;
        }
        indptr[_ncols] = 0;
    }

    ~CSR_Matrix() {}

    index_t add_value(index_t row, index_t col, data_t value)
    {
        index_t pos = col_starts[col]++;
        indices[pos] = row;
        data[pos] = value;
        indptr[col]++;
        nnz++;
    }

    index_t add_values(index_t col, index_t* rows, data_t* values, index_t num_values)
    {
        for (index_t i = 0; i < num_values; i++)
        {
            index_t pos = col_starts[col]++;
            indices[pos] = rows[i];
            data[pos] = values[i];
        }
        indptr[col] += num_values;
        nnz += num_values;
    }

    index_t resize(index_t _nrows, index_t _ncols)
    {
        n_rows = _nrows;
        n_cols = _ncols;
    }

    index_t finalize()
    {
        index_t* temp_idx = new index_t[nnz];
        data_t* temp_data = new data_t[nnz];
        index_t ctr = 0;

        for (index_t col = 0; col < n_cols; col++)
        {
            index_t col_start = indptr[col];
            index_t col_end = col_starts[col];

            indptr[col] = ctr;

            for (index_t j = col_start; j < col_end; j++)
            {
                temp_idx[ctr] = indices[j];
                temp_data[ctr] = data[j];
                ctr++;
            }
        }
        indptr[n_cols] = ctr;

        delete[] indices;
        delete[] data;
        delete[] col_starts;

        indices = temp_idx;
        data = temp_data;
    }

    index_t* col_starts;
};


#endif
