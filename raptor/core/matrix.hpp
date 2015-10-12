// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_MATRIX_H
#define RAPTOR_CORE_MATRIX_H

#include "types.hpp"
#include <vector>
#include <map>
#include <algorithm>

/**************************************************************
 *****   Matrix Class
 **************************************************************
 ***** This class constructs a sparse matrix in either coordinate 
 ***** (COO), compressed sparse row (CSR), or compressed sparse
 ***** column format.  The matrix is initilized into COO format 
 ***** and then can be converted.
 ***** //TODO - add directly into compressed format
 *****
 ***** Attributes
 ***** -------------
 ***** row_idx : vector<index_t>
 *****    Array of row indices (only in COO)
 ***** col_idx : vector<index_t>
 *****    Array of column indices (only in COO)
 ***** indptr : vector<index_t>
 *****    Array of pointers to the outer indices (Row in CSR, Col in CSC)
 ***** indices : vector<index_t>
 *****    Indices of the inner array (Col in CSR, Row in CSC)
 ***** data : vector<data_t>
 *****    Values of the matrix data
 ***** n_rows : index_t
 *****    Number of rows in the matrix
 ***** n_cols : index_t 
 *****    Number of columns in the matrix
 ***** n_outer : index_t
 *****    Number of outer values in matrix (Rows in CSR, Cols in CSC)
 ***** n_inner : index_t
 *****    Number of inner values in matrix (Cols in CSR, Rows in CSC)
 ***** nnz : index_t
 *****    Number of nonzero values in the matrix
 ***** format : format_t
 *****    Format of Matrix (CSR, CSC, COO)
 ***** 
 ***** Methods
 ***** -------
 ***** add_value()
 *****    Insert a single value into the matrix.
 ***** reserve()
 *****    Reserve nnz/row (or nnz/col)
 *****    TODO -- Currently doesn't affect anything
 ***** resize()
 *****    Change the matrix dimensions.
 ***** finalize()
 ****    Convert the matrix to compressed form,
 *****    removing any zero values.
 *****    TODO -- implement this for other options than
 *****    COO to compressed
 ***** convert()
 *****    Convert between formats
 **************************************************************/
namespace raptor
{
class Matrix
{

public:
    /**************************************************************
    *****   Matrix Class Constructor
    **************************************************************
    ***** Initializes an empty Matrix of the given format
    *****
    ***** Parameters
    ***** -------------
    ***** _nrows : index_t
    *****    Number of rows in the matrix
    ***** _ncols : index_t 
    *****    Number of columns in the matrix
    ***** _format : format_t
    *****    Format of the Matrix (COO, CSR, CSC)
    **************************************************************/
    Matrix(index_t _nrows, index_t _ncols, format_t _format = COO)
    {
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

    /**************************************************************
    *****   Matrix Class Constructor
    **************************************************************
    ***** Initializes an empty COO Matrix (without setting dimensions)
    *****
    **************************************************************/
    Matrix();

    /**************************************************************
    *****   Matrix Class Constructor
    **************************************************************
    ***** TODO -- copy Matrix?
    *****
    ***** Parameters
    ***** -------------
    ***** A : Matrix*
    *****    Matrix to copy
    **************************************************************/
    Matrix(Matrix* A);

    /**************************************************************
    *****   Matrix Class Destructor
    **************************************************************
    ***** Deletes all arrays/vectors
    *****
    **************************************************************/
    ~Matrix();

    /**************************************************************
    *****   Matrix Reserve
    **************************************************************
    ***** Reserves nnz per each outer (row or column) to add entries 
    ***** without re-allocating memory.
    ***** TODO -- implement for COO, add CSR/CSC functionality
    *****
    ***** Parameters
    ***** -------------
    ***** nnz_per_outer : index_t
    *****    Number of nonzeros per each row (or column if CSC)
    **************************************************************/
    void reserve(index_t nnz_per_outer);

    /**************************************************************
    *****   Matrix Add Value
    **************************************************************
    ***** Inserts a value into the Matrix
    ***** TODO -- functionality for CSR/CSC
    *****
    ***** Parameters
    ***** -------------
    ***** ptr : index_t
    *****    Outer index (Row in CSR, Col in CSC)
    ***** idx : index_t 
    *****    Inner index (Col in CSR, Row in CSC)
    ***** value : data_t
    *****    Value to insert
    **************************************************************/
    void add_value(index_t ptr, index_t idx, data_t value);

    /**************************************************************
    *****   Matrix Resize
    **************************************************************
    ***** Resizes the dimensions of the matrix
    *****
    ***** Parameters
    ***** -------------
    ***** _nrows : index_t
    *****    Number of rows in the matrix
    ***** _ncols : index_t 
    *****    Number of columns in the matrix
    **************************************************************/
    void resize(index_t _nrows, index_t _ncols);

    /**************************************************************
    *****   Matrix Finalize
    **************************************************************
    ***** Compresses matrix, sorts the entries, removes any zero
    ***** values, and combines any entries at the same location
    *****
    ***** Parameters
    ***** -------------
    ***** _format : format_t
    *****    Format to convert Matrix to
    **************************************************************/
    void finalize(format_t _format);
    void finalize(format_t _format, std::map<index_t, index_t>& to_local);

    /**************************************************************
    *****   Matrix Convert
    **************************************************************
    ***** Converts matrix into given format.  If format is the same
    ***** as that already set, nothing is done.
    *****
    ***** Parameters
    ***** -------------
    ***** _format : format_t
    *****    Format to convert Matrix to
    **************************************************************/
    void convert(format_t _format);

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
}
#endif
