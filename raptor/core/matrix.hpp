// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_MATRIX_H
#define RAPTOR_CORE_MATRIX_H

#include "types.hpp"
#include "array.hpp"
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
 ***** indptr : Array<index_t>
 *****    Array of pointers to the outer indices (Row in CSR, Col in CSC)
 ***** indices : Array<index_t>
 *****    Indices of the inner array (Col in CSR, Row in CSC)
 ***** data : Array<data_t>
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
 ***** resize()
 *****    Change the matrix dimensions.
 ***** col_to_local()
 *****    Converts columns indices from global to local
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
    *****    Format of the Matrix (CSR, CSC)
    **************************************************************/
    Matrix(index_t _nrows, index_t _ncols, format_t _format = CSR)
    {
        n_rows = _nrows;
        n_cols = _ncols;
        nnz = 0;
        format = _format;  // Adds to COO and then compresses to format
    }

    /**************************************************************
    *****   Matrix Class Constructor
    **************************************************************
    ***** Initializes an empty Matrix (without setting dimensions)
    *****
    ***** Parameters
    ***** -------------
    ***** _format : format_t
    *****    Format of the Matrix (CSR, CSC)
    **************************************************************/
    Matrix(format_t _format = CSR)
    {
        n_rows = 0;
        n_cols = 0;
        nnz = 0;
        format = _format;
    }

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
    Matrix(Matrix* A)
    {

    }

    /**************************************************************
    *****   Matrix Class Destructor
    **************************************************************
    ***** Deletes all arrays/vectors
    *****
    **************************************************************/
    ~Matrix()
    {

    }

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
    *****   Matrix ColToLocal
    **************************************************************
    ***** Converts column indices to be local values
    ***** 0 to number of cols
    *****
    ***** Parameters
    ***** -------------
    ***** map : std::map<index_t, index_t>
    *****    Maps global columns to local columns
    **************************************************************/
    void col_to_local(std::map<index_t, index_t>& map);

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
    void finalize();

    void move_diag_first();

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

    Array<index_t> indptr;
    Array<index_t> indices;
    Array<data_t> data;

    index_t n_rows;
    index_t n_cols;
    index_t n_outer;
    index_t n_inner;
    index_t nnz;

    format_t format;

};
}
#endif
