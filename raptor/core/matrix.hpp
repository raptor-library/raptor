// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_MATRIX_H
#define RAPTOR_CORE_MATRIX_H

#include "types.hpp"

template <int MatType>
class Matrix
{
//    Stores sequential sparse matrix
//
//    The class constructs a sparse matrix in either the compressed sparse row (CSR)
//    or compressed sparse column (CSC) storage format.  A empty matrix can be 
//    initialized directly into a compressed format if a bound on the number
//    of nonzeros per row/column is known.  Otherwise, the matrix is initialized
//    into coordinate (COO) format, and then converted to compressed form.
//
//    Attributes
//    ----------
//    indptr : vector<index_t>
//        Array of pointers to the outer indices (Row in CSR, Col in CSC)
//    indices : vector<index_t>
//        Indices of the inner array (Col in CSR, Row in CSC)
//    data : vector<data_t>
//        Values of the matrix data
//    ptr_nnz : vector<index_t>
//        Number of nonzeros added to each value in outer array.  Only used to initialize
//        matrix, and cleared when the finalize() method is called.
//    n_rows : index_t
//        Number of rows in the matrix
//    n_cols : index_t 
//        Number of columns in the matrix
//    n_outer : index_t
//        Number of outer values in matrix (Rows in CSR, Cols in CSC)
//    nnz : index_t
//        Number of nonzero values in the matrix
//    init_coo : index_t
//        Boolean whether to initialize matrix as COO or add values
//        directly into compressed storage format.
//
//    Methods
//    -------
//    add_values()
//        Insert multiple values into the matrix.
//    add_value()
//        Insert a single value into the matrix.
//    resize()
//        Change the matrix dimensions.
//    finalize()
//        Convert the matrix to compressed form, removing any zero values.
//
public:
    Matrix(index_t _nrows, index_t _ncols, index_t _nouter)
    {
        //    Class constructor responsible for initializing an empty matrix
        //    in COO format.
        //
        //    Parameters
        //    ----------
        //    _nrows : index_t
        //        Number of rows in the matrix.
        //    _ncols : index_t 
        //        Number of columns in the matrix.
        //    _nouter : index_t
        //        Number of indices in the outer array (Rows if CSR and
        //        Cols if CSC).

        n_rows = _nrows;
        n_cols = _ncols;
        n_outer = _nouter;
        nnz = 0;

        ptr_nnz.resize(n_outer);
        init_coo = 1;
    }

    Matrix(index_t _nrows, index_t _ncols, index_t _nouter, index_t nnz_per_outer)
    {
        //    Class constructor responsible for initializing an empty matrix
        //    directly into compressed format.  Initalizes each outer index
        //    to hold the given number of elements (nnz_per_outer).
        //
        //    Parameters
        //    ----------
        //    _nrows : index_t
        //        Number of rows in the matrix.
        //    _ncols : index_t 
        //        Number of columns in the matrix.
        //    _nouter : index_t
        //        Number of indices in the outer array (Rows if CSR and
        //        Cols if CSC).
        //    nnz_per_outer : index_t
        //        Maximum number of nonzeros per outer index

        n_rows = _nrows;
        n_cols = _ncols;
        n_outer = _nouter;
        nnz = 0;

        index_t max_nnz = nnz_per_outer * n_outer;
       
        indptr.resize(n_outer + 1);
        indices.resize(max_nnz);
        data.resize(max_nnz);
        ptr_nnz.resize(n_outer);

        for (index_t i = 0; i < n_outer; i++)
        {
            ptr_nnz[i] = 0;
            indptr[i] = i*nnz_per_outer;
        }
        indptr[n_outer] = n_outer*nnz_per_outer;
        init_coo = 0;
    }

    ~Matrix(){}

    void add_values(index_t ptr, index_t* idxs, data_t* values, index_t num_values)
    {
        //    Insert multiple values into a single outer index of the matrix.
        //
        //    Parameters
        //    ----------
        //    ptr : index_t
        //        Outer index to insert values into.
        //    idxs : index_t* 
        //        Array of inner indices to be inserted into matrix.
        //    values : data_t*
        //        Array of values to be inserted into the matrix.
        //    num_values : index_t
        //        The number of values to be inserted into the matrix.

        for (index_t i = 0; i < num_values; i++)
        {
            index_t pos = indptr[ptr] + ptr_nnz[ptr]++;
            indices[pos] = idxs[i];
            data[pos] = values[i];
        }
        nnz += num_values;
    }

    void add_value(index_t ptr, index_t idx, data_t value)
    {
        //    Insert a single value into the matrix.
        //
        //    Parameters
        //    ----------
        //    ptr : index_t
        //        Outer index of nonzero.
        //    idx : index_t*
        //        Inner index of nonzero.
        //    value : data_t
        //        Data value of nonzero.

        if (init_coo)
        {
            indptr.push_back(ptr);
            indices.push_back(idx);
            data.push_back(value);
            if (ptr >= ptr_nnz.size())
            {
                ptr_nnz.push_back(1);
            }
            else
            {
                ptr_nnz[ptr]++;
            }
            nnz++;
        }
        else
        {
            index_t pos = indptr[ptr] + ptr_nnz[ptr]++;
            indices[pos] = idx;
            data[pos] = value;
            nnz++;
        }
    }

    void resize(index_t _nrows, index_t _ncols, index_t _nouter)
    {
        //    Resize the dimensions of the matrix.
        //
        //    Parameters
        //    ----------
        //    _nrows : index_t
        //        New number of rows in the matrix.
        //    _ncols : index_t
        //        New number of columns in the matrix.
        //    _nouter : index_t
        //        New number of outer indices in the matrix.

        n_rows = _nrows;
        n_cols = _ncols;
        n_outer = _nouter;
    }

    void finalize()
    {
        //    Converts the matrix into a useable, compressed form.  If
        //    the matrix is initialized as a COO matrix (init_coo == 1)
        //    the matrix is converted into the compressed format, and 
        //    the COO matrix is cleared.  If the matrix was initialized 
        //    directly into a compressed format, zeros are removed.  The
        //    vector ptr_nnz is cleared.

        if (init_coo) // Convert COO to CSR
        {
            index_t indptr_a[n_outer+1];
            index_t indices_a[nnz];
            data_t data_a[nnz];

            indptr_a[0] = 0;
            for (index_t ptr = 0; ptr < n_outer; ptr++)
            {
                indptr_a[ptr+1] = indptr_a[ptr] + ptr_nnz[ptr];
                ptr_nnz[ptr] = 0;
            }

            for (index_t ctr = 0; ctr < nnz; ctr++)
            {
                index_t ptr = indptr[ctr];
                index_t pos = indptr_a[ptr] + ptr_nnz[ptr]++;
                indices_a[pos] = indices[ctr];
                data_a[pos] = data[ctr];
            }

            indptr.assign(indptr_a, indptr_a + n_outer+1);
            indices.assign(indices_a, indices_a + nnz);
            data.assign(data_a, data_a + nnz);
        }
        else // Remove zeros from inital CSR
        {
            index_t ctr = 0;

            for (index_t ptr = 0; ptr < n_outer; ptr++)
            {
                index_t ptr_start = indptr[ptr];
                index_t ptr_end = ptr_start + ptr_nnz[ptr];

                indptr[ptr] = ctr;

                for (index_t j = ptr_start; j < ptr_end; j++)
                {
                    indices[ctr] = indices[j];
                    data[ctr] = data[j];
                    ctr++;
                }
            }
            indptr[n_outer] = ctr;

            data.resize(nnz);
            indices.resize(nnz);
        }
        ptr_nnz.clear();
    }

    std::vector<index_t> indptr;
    std::vector<index_t> indices;
    std::vector<index_t> ptr_nnz;
    std::vector<data_t> data;

    index_t n_rows;
    index_t n_cols;
    index_t n_outer;
    index_t nnz;
    index_t init_coo;

};

class CSR_Matrix : public Matrix<1>
{
//    The class constructs a compressed sparse row matrix, extending the
//    Matrix class, with rows being the outer index.

public:
    CSR_Matrix(index_t _nrows, index_t _ncols) : Matrix(_nrows, _ncols, _nrows)
    {
        //    Class constructor responsible for initializing an empty matrix
        //    in COO format.  Calls the Matrix constructor, with number of 
        //    outer indices equal to the number of rows.
    }
    CSR_Matrix(index_t _nrows, index_t _ncols, index_t nnz_per_row) : Matrix(_nrows, _ncols, _nrows, nnz_per_row)
    {
        //    Class constructor responsible for initializing an empty matrix
        //    directly into CSR format.  Calls the Matrix constructor, with number of 
        //    outer indices equal to the number of rows.
    }

    ~CSR_Matrix() {}

    void add_value(index_t row, index_t col, data_t value)
    {
        Matrix::add_value(row, col, value);
    }

    void add_values(index_t row, index_t* cols, data_t* values, index_t num_values)
    {
        Matrix::add_values(row, cols, values, num_values);
    }

    void resize(index_t _nrows, index_t _ncols)
    {
        Matrix::resize(_nrows, _ncols, _nrows);
    }
};


class CSC_Matrix : public Matrix<0>
{

public:
    CSC_Matrix(index_t _nrows, index_t _ncols) : Matrix(_nrows, _ncols, _ncols)
    {
        //    Class constructor responsible for initializing an empty matrix
        //    in COO format.  Calls the Matrix constructor, with number of 
        //    outer indices equal to the number of columns.
    }
    CSC_Matrix(index_t _nrows, index_t _ncols, index_t nnz_per_col) : Matrix(_nrows, _ncols, _ncols, nnz_per_col)
    { 
        //    Class constructor responsible for initializing an empty matrix
        //    directly into CSC format.  Calls the Matrix constructor, with number of 
        //    outer indices equal to the number of columns.
    }

    ~CSC_Matrix() {}

    void add_value(index_t row, index_t col, data_t value)
    {
        Matrix::add_value(col, row, value);
    }

    void add_values(index_t col, index_t* rows, data_t* values, index_t num_values)
    {
        Matrix::add_values(col, rows, values, num_values);
    }

    void resize(index_t _nrows, index_t _ncols)
    {
        Matrix::resize(_nrows, _ncols, _ncols);
    }

};

#endif
