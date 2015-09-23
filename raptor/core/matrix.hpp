// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_MATRIX_H
#define RAPTOR_CORE_MATRIX_H

#include "types.hpp"
#include <vector>
#include <algorithm>

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
//    n_inner : index_t
//        Number of inner values in matrix (Cols in CSR, Rows in CSC)
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
    Matrix(index_t _nrows, index_t _ncols, format_t _format = COO)
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
        nnz = 0;
        format = _format;

        if (format == CSR)
        {
            //TODO - implement option for straight to CSR
        }
        else if (format == CSC)
        {
            //TODO - implement option for straight to CSC
        }
    }

    Matrix();
    Matrix(Matrix* A);
    ~Matrix();

    void reserve(index_t nnz_per_outer);
    void add_value(index_t ptr, index_t idx, data_t value);
    void resize(index_t _nrows, index_t _ncols);
    void finalize(format_t _format);
    void convert(format_t _format);

    //COO - row/col idx, CSR/CSC - indptr/indices
    std::vector<index_t> row_idx;
    std::vector<index_t> col_idx;
    std::vector<index_t> indptr;
    std::vector<index_t> indices;
    std::vector<data_t> data;

    index_t n_rows;
    index_t n_cols;
    index_t n_outer;
    index_t n_inner;
    index_t nnz;

    format_t format;

};

#endif
