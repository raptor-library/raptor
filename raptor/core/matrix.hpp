// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_MATRIX_HPP
#define RAPTOR_CORE_MATRIX_HPP

#include "types.hpp"
#include "vector.hpp"

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
 *****
 ***** Methods
 ***** -------
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
 **************************************************************/
namespace raptor
{
  // Forward Declaration of classes so objects can be used
  class COOMatrix;
  class CSRMatrix;
  class CSCMatrix;
  class CSRBoolMatrix;

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
        sorted = false;
        diag_first = false;
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
        sorted = false;
        diag_first = false;
    }

    virtual ~Matrix(){}

    virtual format_t format() = 0;
    virtual void sort() = 0;
    virtual void move_diag() = 0;
    virtual void remove_duplicates() = 0;
    virtual void add_value(int row, int col, double val) = 0;

    virtual void copy(const COOMatrix* A) = 0;
    virtual void copy(const CSRMatrix* A) = 0;
    virtual void copy(const CSCMatrix* A) = 0;

    void jacobi(Vector& x, Vector& b, Vector& tmp, double omega = .667);
    void gauss_seidel(Vector& x, Vector& b);
    void SOR(Vector& x, Vector& b, double omega = .667);

    Matrix* strength(double theta = 0.0);
    Matrix* aggregate();

    void mult(Vector& x, Vector& b)
    {
        mult(x.values, b.values);
    }
    void mult(std::vector<double>& x, Vector& b)
    {
        mult(x, b.values);
    }
    void mult(Vector& x, std::vector<double>& b)
    {
        mult(x.values, b);
    }
    virtual void mult(std::vector<double>& x, std::vector<double>& b) = 0;

    void mult_T(Vector& x, Vector& b)
    {
        mult_T(x.values, b.values);
    }
    void mult_T(std::vector<double>& x, Vector& b)
    {
        mult_T(x, b.values);
    }
    void mult_T(Vector& x, std::vector<double>& b)
    {
        mult_T(x.values, b);
    }
    virtual void mult_T(std::vector<double>& x, std::vector<double>& b) = 0;

    void mult_append(Vector& x, Vector& b)
    {
        mult_append(x.values, b.values);
    }
    void mult_append(std::vector<double>& x, Vector& b)
    {
        mult_append(x, b.values);
    }
    void mult_append(Vector& x, std::vector<double>& b)
    {
        mult_append(x.values, b);
    }
    virtual void mult_append(std::vector<double>& x, std::vector<double>& b) = 0;

    void mult_append_T(Vector& x, Vector& b)
    {
        mult_append_T(x.values, b.values);
    }
    void mult_append_T(std::vector<double>& x, Vector& b)
    {
        mult_append_T(x, b.values);
    }
    void mult_append_T(Vector& x, std::vector<double>& b)
    {
        mult_append_T(x.values, b);
    }
    virtual void mult_append_T(std::vector<double>& x, std::vector<double>& b) = 0;

    void mult_append_neg(Vector& x, Vector& b)
    {
        mult_append_neg(x.values, b.values);
    }
    void mult_append_neg(std::vector<double>& x, Vector& b)
    {
        mult_append_neg(x, b.values);
    }
    void mult_append_neg(Vector& x, std::vector<double>& b)
    {
        mult_append_neg(x.values, b);
    }
    virtual void mult_append_neg(std::vector<double>& x, std::vector<double>& b) = 0;

    void mult_append_neg_T(Vector& x, Vector& b)
    {
        mult_append_neg_T(x.values, b.values);
    }
    void mult_append_neg_T(std::vector<double>& x, Vector& b)
    {
        mult_append_neg_T(x, b.values);
    }
    void mult_append_neg_T(Vector& x, std::vector<double>& b)
    {
        mult_append_neg_T(x.values, b);
    }
    virtual void mult_append_neg_T(std::vector<double>& x, std::vector<double>& b) = 0;

    void residual(const Vector& x, const Vector& b, Vector& r)
    {
        residual(x.values, b.values, r.values);
    }
    void residual(const std::vector<double>& x, const Vector& b, Vector& r)
    {
        residual(x, b.values, r.values);
    }
    virtual void residual(const std::vector<double>& x, const std::vector<double>& b,
            std::vector<double>& r) = 0;
/*
    CSRMatrix* mult(const CSRMatrix* B){}
    CSRMatrix* mult(const CSCMatrix* B){}
    CSRMatrix* mult(const COOMatrix* B){}
    CSRMatrix* mult_T(const CSRMatrix* A){}
    CSRMatrix* mult_T(const CSCMatrix* A){}
    CSRMatrix* mult_T(const COOMatrix* A){}
*/
    void RAP(const CSCMatrix& P, CSCMatrix* Ac);
    void RAP(const CSCMatrix& P, CSRMatrix* Ac);

    Matrix* subtract(Matrix* B);

    void resize(int _n_rows, int _n_cols);

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

    int n_rows;
    int n_cols;
    int nnz;

    bool sorted;
    bool diag_first;

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
            if (_nnz)
            {
                idx1.reserve(_nnz);
                idx2.reserve(_nnz);
                vals.reserve(_nnz);
            }
        }        
    }

    COOMatrix(int _nrows, int _ncols, double* _data) : Matrix(_nrows, _ncols)
    {
        nnz = 0;
        int nnz_dense = n_rows*n_cols;

        if (nnz_dense)
        {
            idx1.reserve(nnz_dense);
            idx2.reserve(nnz_dense);
            vals.reserve(nnz_dense);
        }

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

    void print();

    void copy(const COOMatrix* A);
    void copy(const CSRMatrix* A);
    void copy(const CSCMatrix* A);

    void add_value(int row, int col, double value);
    void sort();
    void move_diag();
    void remove_duplicates();

    template <typename T, typename U> void mult(T& x, U& b)
    { 
        Matrix::mult(x, b);
    }
    template <typename T, typename U> void mult_T(T& x, U& b)
    { 
        Matrix::mult_T(x, b);
    }
    template <typename T, typename U> void mult_append(T& x, U& b)
    { 
        Matrix::mult_append(x, b);
    }
    template <typename T, typename U> void mult_append_T(T& x, U& b)
    { 
        Matrix::mult_append_T(x, b);
    }
    template <typename T, typename U> void mult_append_neg(T& x, U& b)
    { 
        Matrix::mult_append_neg(x, b);
    }
    template <typename T, typename U> void mult_append_neg_T(T& x, U& b)
    { 
        Matrix::mult_append_neg_T(x, b);
    }
    template <typename T, typename U, typename V> 
    void residual(const T& x, const U& b, V& r)
    { 
        Matrix::residual(x, b, r);
    }

    void mult(std::vector<double>& x, std::vector<double>& b)
    {
        for (int i = 0; i < n_rows; i++)
            b[i] = 0.0;
        mult_append(x, b);
    }
    void mult_T(std::vector<double>& x, std::vector<double>& b)
    {
        for (int i = 0; i < n_cols; i++)
            b[i] = 0.0;

        mult_append_T(x, b);
    }
    void mult_append(std::vector<double>& x, std::vector<double>& b)
    { 
        for (int i = 0; i < nnz; i++)
        {
            b[idx1[i]] += vals[i] * x[idx2[i]];
        }
    }
    void mult_append_T(std::vector<double>& x, std::vector<double>& b)
    {
        for (int i = 0; i < nnz; i++)
        {
            b[idx2[i]] += vals[i] * x[idx1[i]];
        }
    }
    void mult_append_neg(std::vector<double>& x, std::vector<double>& b)
    {
        for (int i = 0; i < nnz; i++)
        {
            b[idx1[i]] -= vals[i] * x[idx2[i]];
        }
    }
    void mult_append_neg_T(std::vector<double>& x, std::vector<double>& b)
    {
        for (int i = 0; i < nnz; i++)
        {
            b[idx2[i]] -= vals[i] * x[idx1[i]];
        }
    }
    void residual(const std::vector<double>& x, const std::vector<double>& b, 
            std::vector<double>& r)
    {
        for (int i = 0; i < n_rows; i++)
            r[i] = b[i];
     
        for (int i = 0; i < nnz; i++)
        {
            r[idx1[i]] -= vals[i] * x[idx2[i]];
        }
    }

    CSRMatrix* mult(const CSRMatrix* B);
    CSRMatrix* mult(const CSCMatrix* B);
    CSRMatrix* mult(const COOMatrix* B);
    CSRMatrix* mult_T(const CSRMatrix* A);
    CSRMatrix* mult_T(const CSCMatrix* A);
    CSRMatrix* mult_T(const COOMatrix* A);

    void mult_append(Vector& x, Vector& b);
    void mult_append_neg(Vector& x, Vector& b);
    void mult_append_T(Vector& x, Vector& b);
    void mult_append_neg_T(Vector& x, Vector& b);

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
        idx1.resize(_nrows + 1);
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
        if (nnz_dense)
        {
            idx2.reserve(nnz_dense);
            vals.reserve(nnz_dense);
        }

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
    explicit CSRMatrix(const COOMatrix* A) 
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
    explicit CSRMatrix(const CSCMatrix* A)
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

    void print();

    void copy(const COOMatrix* A);
    void copy(const CSRMatrix* A);
    void copy(const CSCMatrix* A);

    void add_value(int row, int col, double value);
    void sort();
    void move_diag();
    void remove_duplicates();

    template <typename T, typename U> void mult(T& x, U& b)
    { 
        Matrix::mult(x, b);
    }
    template <typename T, typename U> void mult_T(T& x, U& b)
    { 
        Matrix::mult_T(x, b);
    }
    template <typename T, typename U> void mult_append(T& x, U& b)
    { 
        Matrix::mult_append(x, b);
    }
    template <typename T, typename U> void mult_append_T(T& x, U& b)
    { 
        Matrix::mult_append_T(x, b);
    }
    template <typename T, typename U> void mult_append_neg(T& x, U& b)
    { 
        Matrix::mult_append_neg(x, b);
    }
    template <typename T, typename U> void mult_append_neg_T(T& x, U& b)
    { 
        Matrix::mult_append_neg_T(x, b);
    }
    template <typename T, typename U, typename V> 
    void residual(const T& x, const U& b, V& r)
    { 
        Matrix::residual(x, b, r);
    }

    void mult(std::vector<double>& x, std::vector<double>& b)
    {
        for (int i = 0; i < n_rows; i++)
            b[i] = 0.0;
        mult_append(x, b);
    }
    void mult_T(std::vector<double>& x, std::vector<double>& b)

    {
        for (int i = 0; i < n_cols; i++)
            b[i] = 0.0;

        mult_append_T(x, b);    
    }
    void mult_append(std::vector<double>& x, std::vector<double>& b)
    { 
        int start, end;
        for (int i = 0; i < n_rows; i++)
        {
            start = idx1[i];
            end = idx1[i+1];
            for (int j = start; j < end; j++)
            {
                b[i] += vals[j] * x[idx2[j]];
            }
        }
    }
    void mult_append_T(std::vector<double>& x, std::vector<double>& b)
    {
        int start, end;
        for (int i = 0; i < n_rows; i++)
        {
            start = idx1[i];
            end = idx1[i+1];
            for (int j = start; j < end; j++)
            {
                b[idx2[j]] += vals[j] * x[i];
            }
        }
    }
    void mult_append_neg(std::vector<double>& x, std::vector<double>& b)
    {
        int start, end;
        for (int i = 0; i < n_rows; i++)
        {
            start = idx1[i];
            end = idx1[i+1];
            for (int j = start; j < end; j++)
            {
                b[i] -= vals[j] * x[idx2[j]];
            }
        }
    }
    void mult_append_neg_T(std::vector<double>& x, std::vector<double>& b)
    {
        int start, end;
        for (int i = 0; i < n_rows; i++)
        {
            start = idx1[i];
            end = idx1[i+1];
            for (int j = start; j < end; j++)
            {
                b[idx2[j]] -= vals[j] * x[i];
            }
        }
    }
    void residual(const std::vector<double>& x, const std::vector<double>& b, 
            std::vector<double>& r)
    {
        for (int i = 0; i < n_rows; i++)
            r[i] = b[i];
     
        int start, end;
        for (int i = 0; i < n_rows; i++)
        {
            start = idx1[i];
            end = idx1[i+1];
            for (int j = start; j < end; j++)
            {
                r[i] -= vals[j] * x[idx2[j]];
            }
        }
    }


    CSRMatrix* mult(const CSRMatrix* B);
    CSRMatrix* mult(const CSCMatrix* B);
    CSRMatrix* mult(const COOMatrix* B);
    CSRMatrix* mult_T(const CSCMatrix* A);
    CSRMatrix* mult_T(const CSRMatrix* A);
    CSRMatrix* mult_T(const COOMatrix* A);

    CSRMatrix* subtract(CSRMatrix* B);

    CSRBoolMatrix* strength(double theta = 0.0);
    CSRMatrix* aggregate();
    CSRMatrix* fit_candidates(data_t* B, data_t* R, int num_candidates, 
            double tol = 1e-10);

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

    CSCMatrix(int _nrows, int _ncols, int _nnz = 0): Matrix(_nrows, _ncols)
    {
        idx1.resize(_ncols + 1);
        if (_nnz)
        {
            idx2.reserve(_nnz);
            vals.reserve(_nnz);
        }
        nnz = _nnz;
    }

    CSCMatrix(int _nrows, int _ncols, double* _data) : Matrix(_nrows, _ncols)
    {
        int nnz_dense = n_rows*n_cols;

        idx1.resize(n_cols + 1);
        if (nnz_dense)
        {
            idx2.reserve(nnz_dense);
            vals.reserve(nnz_dense);
        }

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
    explicit CSCMatrix(const COOMatrix* A) 
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
    explicit CSCMatrix(const CSRMatrix* A) 
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

    void print();

    void copy(const COOMatrix* A);
    void copy(const CSRMatrix* A);
    void copy(const CSCMatrix* A);

    void sort();
    void move_diag();
    void remove_duplicates();
    void add_value(int row, int col, double value);

    template <typename T, typename U> void mult(T& x, U& b)
    { 
        Matrix::mult(x, b);
    }
    template <typename T, typename U> void mult_T(T& x, U& b)
    { 
        Matrix::mult_T(x, b);
    }
    template <typename T, typename U> void mult_append(T& x, U& b)
    { 
        Matrix::mult_append(x, b);
    }
    template <typename T, typename U> void mult_append_T(T& x, U& b)
    { 
        Matrix::mult_append_T(x, b);
    }
    template <typename T, typename U> void mult_append_neg(T& x, U& b)
    { 
        Matrix::mult_append_neg(x, b);
    }
    template <typename T, typename U> void mult_append_neg_T(T& x, U& b)
    { 
        Matrix::mult_append_neg_T(x, b);
    }
    template <typename T, typename U, typename V> 
    void residual(const T& x, const U& b, V& r)
    { 
        Matrix::residual(x, b, r);
    }

    void mult(std::vector<double>& x, std::vector<double>& b)
    {
        for (int i = 0; i < n_rows; i++)
            b[i] = 0.0;
        mult_append(x, b);
    }
    void mult_T(std::vector<double>& x, std::vector<double>& b)
    {
        for (int i = 0; i < n_cols; i++)
            b[i] = 0.0;

        mult_append_T(x, b);
    }
    void mult_append(std::vector<double>& x, std::vector<double>& b)
    { 
        int start, end;
        for (int i = 0; i < n_cols; i++)
        {
            start = idx1[i];
            end = idx1[i+1];
            for (int j = start; j < end; j++)
            {
                b[idx2[j]] += vals[j] * x[i];
            }
        }
    }
    void mult_append_T(std::vector<double>& x, std::vector<double>& b)
    {
        int start, end;
        for (int i = 0; i < n_cols; i++)
        {
            start = idx1[i];
            end = idx1[i+1];
            for (int j = start; j < end; j++)
            {
                b[i] += vals[j] * x[idx2[j]];
            }
        }
    }
    void mult_append_neg(std::vector<double>& x, std::vector<double>& b)
    {
        int start, end;
        for (int i = 0; i < n_cols; i++)
        {
            start = idx1[i];
            end = idx1[i+1];
            for (int j = start; j < end; j++)
            {
                b[idx2[j]] -= vals[j] * x[i];
            }
        }
    }
    void mult_append_neg_T(std::vector<double>& x, std::vector<double>& b)
    {
        int start, end;
        for (int i = 0; i < n_cols; i++)
        {
            start = idx1[i];
            end = idx1[i+1];
            for (int j = start; j < end; j++)
            {
                b[i] -= vals[j] * x[idx2[j]];
            }
        }
    }
    void residual(const std::vector<double>& x, const std::vector<double>& b, 
            std::vector<double>& r)
    {
        for (int i = 0; i < n_rows; i++)
            r[i] = b[i];

        int start, end;
        for (int i = 0; i < n_cols; i++)
        {
            start = idx1[i];
            end = idx1[i+1];
            for (int j = start; j < end; j++)
            {
                r[idx2[j]] -= vals[j] * x[i];
            }
        }
    }

    CSRMatrix* mult(const CSRMatrix* B);
    CSRMatrix* mult(const CSCMatrix* B);
    CSRMatrix* mult(const COOMatrix* B);
    CSRMatrix* mult_T(const CSRMatrix* A);
    CSRMatrix* mult_T(const CSCMatrix* A);
    CSRMatrix* mult_T(const COOMatrix* A);

    void jacobi(Vector& x, Vector& b, Vector& tmp, double omega = .667);    

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



/**************************************************************
 *****   CSRBoolMatrix Class (Inherits from Matrix Base Class)
 **************************************************************
 ***** This class constructs a sparse boolean matrix in CSR format.
 *****
 ***** Methods
 ***** -------
 ***** format() 
 *****    Returns the format of the sparse matrix (CSR)
 ***** sort()
 *****    Sorts the matrix.  Already in row-wise order, but sorts
 *****    the columns in each row.
 ***** indptr()
 *****     Returns std::vector<int>& row pointer.  The ith element points to
 *****     the index of indices() corresponding to the first column to lie on 
 *****     row i.
 ***** indices()
 *****     Returns std::vector<int>& containing the cols corresponding
 *****     to each nonzero
 **************************************************************/
  class CSRBoolMatrix : public Matrix
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
    CSRBoolMatrix(int _nrows, int _ncols, int _nnz = 0): Matrix(_nrows, _ncols)
    {
        idx1.resize(_nrows + 1);
        if (_nnz)
        {
            idx2.reserve(_nnz);
        }
    }

    CSRBoolMatrix(int _nrows, int _ncols, bool* _data) : Matrix(_nrows, _ncols)
    {
        n_rows = _nrows;
        n_cols = _ncols;
        nnz = 0;

        int nnz_dense = n_rows*n_cols;

        idx1.resize(n_rows + 1);
        if (nnz_dense)
        {
            idx2.reserve(nnz_dense);
        }

        idx1[0] = 0;
        for (int i = 0; i < n_rows; i++)
        {
            for (int j = 0; j < n_cols; j++)
            {
                if (_data[i*n_cols + j])
                {
                    idx2.push_back(j);
                }
            }
            idx1[i+1] = idx2.size();
        }
        nnz = idx2.size();
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
    explicit CSRBoolMatrix(const COOMatrix* A) 
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
    explicit CSRBoolMatrix(const CSCMatrix* A)
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
    explicit CSRBoolMatrix(const CSRMatrix* A) 
    {
        copy(A);
    }

    CSRBoolMatrix()
    {
    }

    ~CSRBoolMatrix()
    {

    }

    void print();

    void copy(const COOMatrix* A);
    void copy(const CSRMatrix* A);
    void copy(const CSCMatrix* A);

    void sort();
    void move_diag();
    void remove_duplicates();
    
    void add_value(int row, int col, double val)
    {
        
    }

    void mult(std::vector<double>& x, std::vector<double>& b)
    {
        for (int i = 0; i < n_rows; i++)
            b[i] = 0.0;
        mult_append(x, b);
    }
    void mult_T(std::vector<double>& x, std::vector<double>& b)

    {
        for (int i = 0; i < n_cols; i++)
            b[i] = 0.0;

        mult_append_T(x, b);    
    }
    void mult_append(std::vector<double>& x, std::vector<double>& b)
    { 
        int start, end;
        for (int i = 0; i < n_rows; i++)
        {
            start = idx1[i];
            end = idx1[i+1];
            for (int j = start; j < end; j++)
            {
                b[i] += x[idx2[j]];
            }
        }
    }
    void mult_append_T(std::vector<double>& x, std::vector<double>& b)
    {
        int start, end;
        for (int i = 0; i < n_rows; i++)
        {
            start = idx1[i];
            end = idx1[i+1];
            for (int j = start; j < end; j++)
            {
                b[idx2[j]] += x[i];
            }
        }
    }
    void mult_append_neg(std::vector<double>& x, std::vector<double>& b)
    {
        int start, end;
        for (int i = 0; i < n_rows; i++)
        {
            start = idx1[i];
            end = idx1[i+1];
            for (int j = start; j < end; j++)
            {
                b[i] -= x[idx2[j]];
            }
        }
    }
    void mult_append_neg_T(std::vector<double>& x, std::vector<double>& b)
    {
        int start, end;
        for (int i = 0; i < n_rows; i++)
        {
            start = idx1[i];
            end = idx1[i+1];
            for (int j = start; j < end; j++)
            {
                b[idx2[j]] -= x[i];
            }
        }
    }
    void residual(const std::vector<double>& x, const std::vector<double>& b, 
            std::vector<double>& r)
    {
        for (int i = 0; i < n_rows; i++)
            r[i] = b[i];
     
        int start, end;
        for (int i = 0; i < n_rows; i++)
        {
            start = idx1[i];
            end = idx1[i+1];
            for (int j = start; j < end; j++)
            {
                r[i] -= x[idx2[j]];
            }
        }
    }

    format_t format()
    {
        return CSRBool;
    }

    std::vector<int>& row_ptr()
    {
        return idx1;
    }

    std::vector<int>& cols()
    {
        return idx2;
    }
};


}

#endif

