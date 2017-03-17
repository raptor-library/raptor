// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_MATRIX_HPP
#define RAPTOR_CORE_MATRIX_HPP

#include "vector.hpp"
#include <map>
#include <numeric>
#include <algorithm>
#include <functional>
#include <set>
#include <map>

/**************************************************************
 *****   Matrix Base Class
 **************************************************************
 ***** This class constructs a sparse matrix, supporting simple linear
 ***** algebra operations.
 *****
 ***** Attributes
 ***** -------------
 ***** n_rows : int
 *****    Number of rows
 ***** n_cols : int
 *****    Number of columns
 ***** nnz : int
 *****    Number of nonzeros
 ***** idx1 : std::vector<int>
 *****    List of position indices, specific to type of matrix
 ***** idx2 : std::vector<int>
 *****    List of position indices, specific to type of matrix
 ***** vals : std::vector<double>
 *****    List of values in matrix
 ***** row_list : std::vector<int>
 *****    List of rows containing nonzeros.  Only initialized
 *****    for condensed matrices.
 ***** col_list : std::vector<int>
 *****    List of columns containing nonzeros.  Only initialized
 *****    for condensed matrices.
 *****
 ***** Methods
 ***** -------
 ***** print()
 *****    Prints the nonzeros in the sparse matrix, along with 
 *****    the row and column of the nonzero
 ***** resize(int n_rows, int n_cols)
 *****    Resizes dimension of matrix to passed parameters
 ***** mult(Vector* x, Vector* b)
 *****    Sparse matrix-vector multiplication b = A * x
 ***** residual(Vector* x, Vector* b, Vector* r)
 *****    Calculates the residual r = b - A * x
 *****
 ***** Virtual Methods
 ***** -------
 ***** format() 
 *****    Returns the format of the sparse matrix (COO, CSR, CSC)
 ***** sort()
 *****    Sorts the matrix by position.  Whether row-wise or 
 *****    column-wise depends on matrix format.
 ***** add_value(int row, int col, double val)
 *****     Adds val to position (row, col)
 *****     TODO -- make sure this is working for CSR/CSC
 ***** condense_rows()
 *****     Removes zeros rows from sparse matrix, decreasing the indices
 *****     of remaining rows as needed.  Initializes row_list to contain
 *****     the original rows of the matrix (row_list[i] = orig_row[i])
 ***** condense_cols()
 *****     Removes zeros cols from sparse matrix, decreasing the indices
 *****     of remaining cols as needed.  Initializes col_list to contain
 *****     the original cols of the matrix (col_list[i] = orig_col[i])
 ***** apply_func (std::function<void(int, int, double)> func_ptr)
 *****     Applys function passed as paramter to each position of matrix.
 *****     For example call to this function, see method print()
 ***** apply_func (double* x, double* b, std::function<void(int, int, double ...)>)
 *****     Applys function passed as parameter to each position of matrix,
 *****     where function depends on double* x and double* b.
 *****     For example call to this function, see mult(Vector* x, Vector* b)
 **************************************************************/
namespace raptor
{
  // Forward Declaration of classes so objects can be used
  class COOMatrix;
  class CSRMatrix;
  class CSCMatrix;

  class Matrix
  {

  public:

    /**************************************************************
    *****   Matrix Base Class Constructor
    **************************************************************
    ***** Sets matrix dimensions, and sets nnz to 0
    *****
    ***** Parameters
    ***** -------------
    ***** _nrows : int
    *****    Number of rows in matrix
    ***** _ncols : int
    *****    Number of cols in matrix
    **************************************************************/
    Matrix(int _nrows, int _ncols)
    {
        n_rows = _nrows;
        n_cols = _ncols;
        nnz = 0;
    }

    /**************************************************************
    *****   Matrix Base Class Constructor
    **************************************************************
    ***** Sets matrix dimensions and nnz based on Matrix* A
    *****
    ***** Parameters
    ***** -------------
    ***** A : Matrix*
    *****    Matrix to be copied
    **************************************************************/
    Matrix()
    {
        n_rows = 0;
        n_cols = 0;
        nnz = 0;
    }

    virtual ~Matrix(){}

    virtual format_t format() = 0;
    virtual void sort() = 0;
    virtual void add_value(int row, int col, double val) = 0;
    virtual void condense_rows() = 0;
    virtual void condense_cols() = 0;

    void print();

    void mult_append(Vector& x, Vector& b);
    void mult_append_neg(Vector& x, Vector& b);
    void mult_append_T(Vector& x, Vector& b);
    void mult_append_neg_T(Vector& x, Vector& b);
    void residual(Vector& x, Vector& b, Vector& r);

    virtual void apply_func( std::function<void(int, int, double)> func_ptr) = 0;
    virtual void apply_func( Vector& x, Vector& b, 
            std::function<void(int, int, double, Vector&, Vector&)> func_ptr) = 0;

    virtual void copy(const COOMatrix* A) = 0;
    virtual void copy(const CSRMatrix* A) = 0;
    virtual void copy(const CSCMatrix* A) = 0;

    void jacobi(Vector& x, Vector& b, Vector& tmp, double omega = .667);
    void gauss_seidel(Vector& x, Vector& b);
    void SOR(Vector& x, Vector& b, double omega = .667);

    void classical_strength(CSRMatrix* S, double theta = 0.0);
    void symmetric_strength(CSRMatrix* S, double theta = 0.0);
    void symmetric_strength(CSCMatrix* S, double theta = 0.0);

    void mult(Vector& x, Vector& b);
    void mult_T(Vector& x, Vector& b);
    void mult(const CSRMatrix& B, CSRMatrix* C);
    void mult(const CSCMatrix& B, CSRMatrix* C);
    void mult(const CSCMatrix& B, CSCMatrix* C);
    void mult_T(const CSRMatrix& B, CSRMatrix* C);
    void mult_T(const CSCMatrix& B, CSRMatrix* C);
    void mult_T(const CSCMatrix& B, CSCMatrix* C);

    void RAP(const CSCMatrix& P, CSCMatrix* Ac);
    void RAP(const CSCMatrix& P, CSRMatrix* Ac);

    void subtract(CSRMatrix& B, CSRMatrix& C);
    void subtract(CSCMatrix& B, CSCMatrix& C);

    void resize(int _n_rows, int _n_cols);

    std::vector<index_t>& get_row_list()
    {
        return row_list;
    }

    std::vector<index_t>& get_col_list()
    {
        return col_list;
    }

    std::vector<int>& index1()
    {
        return idx1;
    }

    std::vector<int>& index2()
    {
        return idx2;
    }
    
    std::vector<double>& values()
    {
        return vals;
    }

    std::vector<int> idx1;
    std::vector<int> idx2;
    std::vector<double> vals;

    // Lists of rows with nonzeros
    // Only initialized when matrix is condensed
    std::vector<int> row_list;
    std::vector<int> col_list;

    int n_rows;
    int n_cols;
    int nnz;
  };


/**************************************************************
 *****   COOMatrix Class (Inherits from Matrix Base Class)
 **************************************************************
 ***** This class constructs a sparse matrix in COO format.
 *****
 ***** Methods
 ***** -------
 ***** format() 
 *****    Returns the format of the sparse matrix (COO)
 ***** sort()
 *****    Sorts the matrix by row, and by column within each row.
 ***** add_value(int row, int col, double val)
 *****     Adds val to position (row, col)
 ***** condense_rows()
 *****     Removes zeros rows from sparse matrix, decreasing the indices
 *****     of remaining rows as needed.  Initializes row_list to contain
 *****     the original rows of the matrix (row_list[i] = orig_row[i])
 ***** condense_cols()
 *****     Removes zeros cols from sparse matrix, decreasing the indices
 *****     of remaining cols as needed.  Initializes col_list to contain
 *****     the original cols of the matrix (col_list[i] = orig_col[i])
 ***** apply_func (std::function<void(int, int, double)> func_ptr)
 *****     Applys function passed as paramter to each position of matrix.
 *****     For example call to this function, see method print()
 ***** apply_func (double* x, double* b, std::function<void(int, int, double ...)>)
 *****     Applys function passed as parameter to each position of matrix,
 *****     where function depends on double* x and double* b.
 *****     For example call to this function, see mult(Vector* x, Vector* b)
 ***** rows()
 *****     Returns std::vector<int>& containing the rows corresponding
 *****     to each nonzero
 ***** cols()
 *****     Returns std::vector<int>& containing the cols corresponding
 *****     to each nonzero
 ***** data()
 *****     Returns std::vector<double>& containing the nonzero values
 **************************************************************/
  class COOMatrix : public Matrix
  {

  public:

    /**************************************************************
    *****   COOMatrix Class Constructor
    **************************************************************
    ***** Initializes an empty COOMatrix
    *****
    ***** Parameters
    ***** -------------
    ***** _nrows : int
    *****    Number of rows in Matrix
    ***** _ncols : int
    *****    Number of columns in Matrix
    ***** nnz_per_row : int
    *****    Prediction of (approximately) number of nonzeros 
    *****    per row, used in reserving space
    **************************************************************/
    COOMatrix(int _nrows, int _ncols, int nnz_per_row = 1) : Matrix(_nrows, _ncols)
    {
        if (nnz_per_row)
        {
            int _nnz = nnz_per_row * _nrows;
            idx1.reserve(_nnz);
            idx2.reserve(_nnz);
            vals.reserve(_nnz);
        }        
    }

    COOMatrix(int _nrows, int _ncols, double* _data) : Matrix(_nrows, _ncols)
    {
        nnz = 0;
        int nnz_dense = n_rows*n_cols;

        idx1.reserve(nnz_dense);
        idx2.reserve(nnz_dense);
        vals.reserve(nnz_dense);

        for (int i = 0; i < n_rows; i++)
        {
            for (int j = 0; j < n_cols; j++)
            {
                double val = _data[i*n_cols + j];
                if (fabs(val) > zero_tol)
                {
                    idx1.push_back(i);
                    idx2.push_back(j);
                    vals.push_back(val);
                    nnz++;
                }
            }
        }
    }

    COOMatrix()
    {
    }


    /**************************************************************
    *****   COOMatrix Class Constructor
    **************************************************************
    ***** Constructs a COOMatrix from a CSRMatrix
    *****
    ***** Parameters
    ***** -------------
    ***** A : const CSRMatrix*
    *****    CSRMatrix A, from which to copy data
    **************************************************************/
    explicit COOMatrix(const CSRMatrix* A)
    {
        copy(A);
    }

    /**************************************************************
    *****   COOMatrix Class Constructor
    **************************************************************
    ***** Copies matrix, constructing new COOMatrix from 
    ***** another COOMatrix
    *****
    ***** Parameters
    ***** -------------
    ***** A : const COOMatrix*
    *****    COOMatrix A, from which to copy data
    **************************************************************/
    explicit COOMatrix(const COOMatrix* A)
    {
        copy(A);
    }

    /**************************************************************
    *****   COOMatrix Class Constructor
    **************************************************************
    ***** Constructs a COOMatrix from a CSCMatrix
    *****
    ***** Parameters
    ***** -------------
    ***** A : const CSCMatrix*
    *****    CSCMatrix A, from which to copy data
    **************************************************************/
    explicit COOMatrix(const CSCMatrix* A)
    {
        copy(A);
    }

    ~COOMatrix()
    {

    }

    void copy(const COOMatrix* A);
    void copy(const CSRMatrix* A);
    void copy(const CSCMatrix* A);

    void add_value(int row, int col, double value);
    void condense_rows();
    void condense_cols();
    void sort();
    void apply_func( std::function<void(int, int, double)> func_ptr);
    void apply_func( Vector& x, Vector& b, 
            std::function<void(int, int, double, Vector&, Vector&)> func_ptr);

    format_t format()
    {
        return COO;
    }

    std::vector<int>& rows()
    {
        return idx1;
    }

    std::vector<int>& cols()
    {
        return idx2;
    }

    std::vector<double>& data()
    {
        return vals;

    }
};

/**************************************************************
 *****   CSRMatrix Class (Inherits from Matrix Base Class)
 **************************************************************
 ***** This class constructs a sparse matrix in CSR format.
 *****
 ***** Methods
 ***** -------
 ***** format() 
 *****    Returns the format of the sparse matrix (CSR)
 ***** sort()
 *****    Sorts the matrix.  Already in row-wise order, but sorts
 *****    the columns in each row.
 ***** add_value(int row, int col, double val)
 *****     TODO -- add this functionality
 ***** condense_rows()
 *****     Removes zeros rows from sparse matrix, decreasing the indices
 *****     of remaining rows as needed.  Initializes row_list to contain
 *****     the original rows of the matrix (row_list[i] = orig_row[i])
 ***** condense_cols()
 *****     Removes zeros cols from sparse matrix, decreasing the indices
 *****     of remaining cols as needed.  Initializes col_list to contain
 *****     the original cols of the matrix (col_list[i] = orig_col[i])
 ***** apply_func (std::function<void(int, int, double)> func_ptr)
 *****     Applys function passed as paramter to each position of matrix.
 *****     For example call to this function, see method print()
 ***** apply_func (double* x, double* b, std::function<void(int, int, double ...)>)
 *****     Applys function passed as parameter to each position of matrix,
 *****     where function depends on double* x and double* b.
 *****     For example call to this function, see mult(Vector* x, Vector* b)
 ***** indptr()
 *****     Returns std::vector<int>& row pointer.  The ith element points to
 *****     the index of indices() corresponding to the first column to lie on 
 *****     row i.
 ***** indices()
 *****     Returns std::vector<int>& containing the cols corresponding
 *****     to each nonzero
 ***** data()
 *****     Returns std::vector<double>& containing the nonzero values
 **************************************************************/
  class CSRMatrix : public Matrix
  {

  public:

    /**************************************************************
    *****   CSRMatrix Class Constructor
    **************************************************************
    ***** Initializes an empty CSRMatrix
    *****
    ***** Parameters
    ***** -------------
    ***** _nrows : int
    *****    Number of rows in Matrix
    ***** _ncols : int
    *****    Number of columns in Matrix
    ***** nnz_per_row : int
    *****    Prediction of (approximately) number of nonzeros 
    *****    per row, used in reserving space
    **************************************************************/
    CSRMatrix(int _nrows, int _ncols, int _nnz = 0): Matrix(_nrows, _ncols)
    {
        idx1.reserve(_nrows + 1);
        if (_nnz)
        {
            idx2.reserve(_nnz);
            vals.reserve(_nnz);
        }
    }

    CSRMatrix(int _nrows, int _ncols, double* _data) : Matrix(_nrows, _ncols)
    {
        n_rows = _nrows;
        n_cols = _ncols;
        nnz = 0;

        int nnz_dense = n_rows*n_cols;

        idx1.resize(n_rows + 1);
        idx2.reserve(nnz_dense);
        vals.reserve(nnz_dense);

        idx1[0] = 0;
        for (int i = 0; i < n_rows; i++)
        {
            for (int j = 0; j < n_cols; j++)
            {
                double val = _data[i*n_cols + j];
                if (fabs(val) > zero_tol)
                {
                    idx2.push_back(j);
                    vals.push_back(val);
                    nnz++;
                }
            }
            idx1[i+1] = nnz;
        }
    }


    /**************************************************************
    *****   CSRMatrix Class Constructor
    **************************************************************
    ***** Constructs a CSRMatrix from a COOMatrix
    *****
    ***** Parameters
    ***** -------------
    ***** A : const COOMatrix*
    *****    COOMatrix A, from which to copy data
    **************************************************************/
    explicit CSRMatrix(COOMatrix* A) 
    {
        copy(A);
    }

    /**************************************************************
    *****   CSRMatrix Class Constructor
    **************************************************************
    ***** Constructs a CSRMatrix from a CSCMatrix
    *****
    ***** Parameters
    ***** -------------
    ***** A : const CSCMatrix*
    *****    CSCMatrix A, from which to copy data
    **************************************************************/
    explicit CSRMatrix(CSCMatrix* A)
    {
        copy(A);
    }

    /**************************************************************
    *****   CSRMatrix Class Constructor
    **************************************************************
    ***** Constructs a CSRMatrix from a CSRMatrix
    *****
    ***** Parameters
    ***** -------------
    ***** A : const CSRMatrix*
    *****    CSRMatrix A, from which to copy data
    **************************************************************/
    explicit CSRMatrix(const CSRMatrix* A) 
    {
        copy(A);
    }

    CSRMatrix()
    {
    }

    ~CSRMatrix()
    {

    }

    void copy(const COOMatrix* A);
    void copy(const CSRMatrix* A);
    void copy(const CSCMatrix* A);

    void add_value(int row, int col, double value);
    void condense_rows();
    void condense_cols();
    void sort();
    void apply_func( std::function<void(int, int, double)> func_ptr);
    void apply_func( Vector& x, Vector& b, 
            std::function<void(int, int, double, Vector&, Vector&)> func_ptr);

    void mult(const CSRMatrix& B, CSRMatrix* C);
    void mult(const CSCMatrix& B, CSRMatrix* C);
    void mult(const CSCMatrix& B, CSCMatrix* C);
    void mult(Vector& x, Vector& b);
    void mult_T(Vector& x, Vector& b);

    void classical_strength(CSRMatrix* S, double theta = 0.0);
    void symmetric_strength(CSRMatrix* S, double theta = 0.0);

    format_t format()
    {
        return CSR;
    }

    std::vector<int>& row_ptr()
    {
        return idx1;
    }

    std::vector<int>& cols()
    {
        return idx2;
    }

    std::vector<double>& data()
    {
        return vals;
    }
};

/**************************************************************
 *****   CSCMatrix Class (Inherits from Matrix Base Class)
 **************************************************************
 ***** This class constructs a sparse matrix in CSC format.
 *****
 ***** Methods
 ***** -------
 ***** format() 
 *****    Returns the format of the sparse matrix (CSC)
 ***** sort()
 *****    Sorts the matrix.  Already in col-wise order, but sorts
 *****    the rows in each column.
 ***** add_value(int row, int col, double val)
 *****     TODO -- add this functionality
 ***** condense_rows()
 *****     Removes zeros rows from sparse matrix, decreasing the indices
 *****     of remaining rows as needed.  Initializes row_list to contain
 *****     the original rows of the matrix (row_list[i] = orig_row[i])
 ***** condense_cols()
 *****     Removes zeros cols from sparse matrix, decreasing the indices
 *****     of remaining cols as needed.  Initializes col_list to contain
 *****     the original cols of the matrix (col_list[i] = orig_col[i])
 ***** apply_func (std::function<void(int, int, double)> func_ptr)
 *****     Applys function passed as paramter to each position of matrix.
 *****     For example call to this function, see method print()
 ***** apply_func (double* x, double* b, std::function<void(int, int, double ...)>)
 *****     Applys function passed as parameter to each position of matrix,
 *****     where function depends on double* x and double* b.
 *****     For example call to this function, see mult(Vector* x, Vector* b)
 ***** indptr()
 *****     Returns std::vector<int>& column pointer.  The ith element points to
 *****     the index of indices() corresponding to the first row to lie on 
 *****     column i.
 ***** indices()
 *****     Returns std::vector<int>& containing the rows corresponding
 *****     to each nonzero
 ***** data()
 *****     Returns std::vector<double>& containing the nonzero values
 **************************************************************/
  class CSCMatrix : public Matrix
  {

  public:

    CSCMatrix(int _nrows, int _ncols, int _nnz): Matrix(_nrows, _ncols)
    {
        idx1.reserve(_ncols + 1);
        idx2.reserve(_nnz);
        vals.reserve(_nnz);
        nnz = _nnz;
    }

    CSCMatrix(int _nrows, int _ncols, double* _data) : Matrix(_nrows, _ncols)
    {
        int nnz_dense = n_rows*n_cols;

        idx1.resize(n_cols + 1);
        idx2.reserve(nnz_dense);
        vals.reserve(nnz_dense);

        idx1[0] = 0;
        for (int i = 0; i < n_cols; i++)
        {
            for (int j = 0; j < n_rows; j++)
            {
                double val = _data[i*n_cols + j];
                if (fabs(val) > zero_tol)
                {
                    idx2.push_back(j);
                    vals.push_back(val);
                    nnz++;
                }
            }
            idx1[i+1] = nnz;
        }
    }


    /**************************************************************
    *****   CSCMatrix Class Constructor
    **************************************************************
    ***** Constructs a CSCMatrix from a COOMatrix
    *****
    ***** Parameters
    ***** -------------
    ***** A : const COOMatrix*
    *****    COOMatrix A, from which to copy data
    **************************************************************/
    explicit CSCMatrix(COOMatrix* A) 
    {
        copy(A);
    }

    /**************************************************************
    *****   CSCMatrix Class Constructor
    **************************************************************
    ***** Constructs a CSCMatrix from a CSRMatrix
    *****
    ***** Parameters
    ***** -------------
    ***** A : const CSRMatrix*
    *****    CSRMatrix A, from which to copy data
    **************************************************************/
    explicit CSCMatrix(CSRMatrix* A) 
    {
        copy(A);
    }

    /**************************************************************
    *****   CSCMatrix Class Constructor
    **************************************************************
    ***** Constructs a CSCMatrix from a CSCMatrix
    *****
    ***** Parameters
    ***** -------------
    ***** A : const CSCMatrix*
    *****    CSCMatrix A, from which to copy data
    **************************************************************/
    explicit CSCMatrix(const CSCMatrix* A) 
    {
        copy(A);
    }

    CSCMatrix()
    {
    }

    ~CSCMatrix()
    {

    }

    void copy(const COOMatrix* A);
    void copy(const CSRMatrix* A);
    void copy(const CSCMatrix* A);

    void sort();
    void add_value(int row, int col, double value);
    void condense_rows();
    void condense_cols();
    void apply_func( std::function<void(int, int, double)> func_ptr);
    void apply_func( Vector& x, Vector& b, 
            std::function<void(int, int, double, Vector&, Vector&)> func_ptr);

    void mult(Vector& x, Vector& b);
    void mult_T(Vector& x, Vector& b);

    void mult(const CSCMatrix& B, CSCMatrix* C);
    void mult_T(const CSRMatrix& B, CSRMatrix* C);
    void mult_T(const CSCMatrix& B, CSRMatrix* C);
    void mult_T(const CSCMatrix& B, CSCMatrix* C);

    void jacobi(Vector& x, Vector& b, Vector& tmp, double omega = .667);    

    void symmetric_strength(CSCMatrix* S, double theta = 0.0);

    format_t format()
    {
        return CSC;
    }

    std::vector<int>& col_ptr()
    {
        return idx1;
    }

    std::vector<int>& rows()
    {
        return idx2;
    }

    std::vector<double>& data()
    {
        return vals;
    }

  };


}

#endif

