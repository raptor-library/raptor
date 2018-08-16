// Copyright (c) 2015-2017, RAPtor Developer Team
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
 ***** idx1 : aligned_vector<int>
 *****    List of position indices, specific to type of matrix
 ***** idx2 : aligned_vector<int>
 *****    List of position indices, specific to type of matrix
 ***** vals : aligned_vector<double>
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

    template <typename T>
    void init_from_lists(aligned_vector<int>& _idx1, aligned_vector<int>& _idx2, 
            aligned_vector<T>& data)
    {
        nnz = data.size();
        resize_data(nnz);

        T* val_list = (T*) get_data();

        std::copy(_idx1.begin(), _idx1.end(), std::back_inserter(idx1));
        std::copy(_idx2.begin(), _idx2.end(), std::back_inserter(idx2));

        for (int i = 0; i < nnz; i++)
        {
            val_list[i] = copy_val(data[i]);
        }
    }

    // Virtual Methods
    virtual format_t format() = 0;
    virtual void sort() = 0;
    virtual void move_diag() = 0;
    virtual void remove_duplicates() = 0;
    virtual void print() = 0;
    virtual CSRMatrix* to_CSR() = 0;
    virtual CSCMatrix* to_CSC() = 0;
    virtual COOMatrix* to_COO() = 0;
    virtual Matrix* copy() = 0;

    virtual void spmv_append(const aligned_vector<double>& x, aligned_vector<double>& b) = 0;
    virtual void spmv_append_T(const aligned_vector<double>& x, aligned_vector<double>& b) = 0;
    virtual void spmv_append_neg(const aligned_vector<double>& x, aligned_vector<double>& b) = 0;
    virtual void spmv_append_neg_T(const aligned_vector<double>& x, aligned_vector<double>& b) = 0;

    virtual CSRMatrix* spgemm(const CSRMatrix* B) = 0;
    virtual CSRMatrix* spgemm_T(const CSCMatrix* A) = 0;
    virtual Matrix* transpose() = 0;

    aligned_vector<double>& get_values(Vector& x)
    {
        return x.values;
    }
    template<typename T> aligned_vector<T>& get_values(aligned_vector<T>& x)
    {
        return x;
    }
    
    // Method for printing the value at one position
    // (either single or block value)
    void val_print(int row, int col, double val)
    {
        printf("A[%d][%d] = %e\n", row, col, val);
    }
    void val_print(int row, int col, double* val)
    {
        for (int i = 0; i < b_rows; i++)
        {
            for (int j = 0; j < b_cols; j++)
            {
                printf("A[%d][%d], BlockPos[%d][%d] = %e\n", row, col, i, j, val[i*b_cols+j]);
            }
        }
    }

    double copy_val(double val)
    {
        return val;
    }
    double* copy_val(double* val)
    {
        double* new_val = new double[b_size];
        for (int i = 0; i < b_size; i++)
        {
            new_val[i] = val[i];
        }
        return new_val;
    }

    // Method for finding the absolute value of 
    // either a single or block value
    double abs_val(double val)
    {
        return fabs(val);
    }
    double abs_val(double* val)
    {
        double sum = 0;
        for (int i = 0; i < b_size; i++)
        {
            sum += fabs(val[i]);
        }
        return sum;
    }

    // Methods for appending two values
    // (either single or block values)
    void append_vals(double* val, double addl_val)
    {
        *val += addl_val;
    }
    void append_vals(double** val, double* addl_val)
    {
        for (int i = 0; i < b_size; i++)
        {
            *val[i] += addl_val[i];
        }
    }


    void append(double* b, const double* x, const double val)
    {
        *b += val*(*x);
    }
    void append_T(double* b, const double* x, const double val)
    {
        *b += val*(*x);
    }
    void append_neg(double* b, const double* x, const double val)
    {
        *b -= val*(*x);
    }
    void append_neg_T(double* b, const double* x, const double val)
    {
        *b -= val*(*x);
    }
    void append(double* b, const double* x, const double* val)
    {
        for (int row = 0; row < b_rows; row++)
        {
            for (int col = 0; col < b_cols; col++)
            {
                b[row] += (val[row * b_cols + col] * x[col]);
            }
        }
    }
    void append_T(double* b, const double* x, const double* val)
    {
        for (int row = 0; row < b_rows; row++)
        {
            for (int col = 0; col < b_cols; col++)
            {
                b[col] += (val[row * b_cols + col] * x[row]);
            }
        }
    }
    void append_neg(double* b, const double* x, const double* val)
    {
        for (int row = 0; row < b_rows; row++)
        {
            for (int col = 0; col < b_cols; col++)
            {
                b[row] -= (val[row * b_cols + col] * x[col]);
            }
        }
    }
    void append_neg_T(double* b, const double* x, const double* val)
    {
        for (int row = 0; row < b_rows; row++)
        {
            for (int col = 0; col < b_cols; col++)
            {
                b[col] -= (val[row * b_cols + col] * x[row]);
            }
        }
    }

    template <typename T, typename U> void mult(T& x, U& b)
    {
        for (int i = 0; i < n_rows; i++)
            b[i] = 0.0;
        spmv_append(get_values(x), get_values(b));
    }
    template <typename T, typename U> void mult_T(T& x, U& b)
    {
        for (int i = 0; i < n_cols; i++)
            b[i] = 0.0;
        spmv_append_T(get_values(x), get_values(b));
    }
    template <typename T, typename U> void mult_append(T& x, U& b)
    {
        spmv_append(get_values(x), get_values(b));
    }
    template <typename T, typename U> void mult_append_T(T& x, U& b)
    {
        spmv_append_T(get_values(x), get_values(b));
    }
    template <typename T, typename U> void mult_append_neg(T& x, U& b)
    {
        spmv_append_neg(get_values(x), get_values(b));
    }
    template <typename T, typename U> void mult_append_neg_T(T& x, U& b)
    {
        spmv_append_neg_T(get_values(x), get_values(b));
    }
    template <typename T, typename U, typename V> void residual(T& x, U& b, V& r)
    {
        for (int i = 0; i < n_rows; i++)
            r[i] = b[i];
        spmv_append_neg(get_values(x), get_values(r));
    }

    CSRMatrix* mult(const CSRMatrix* B)
    {
        return spgemm(B);
    }
    CSRMatrix* mult_T(const CSCMatrix* A)
    {
        return spgemm_T(A);
    }

    virtual void add_value(int row, int col, double value) = 0;

    Matrix* add(CSRMatrix* A);
    Matrix* subtract(CSRMatrix* A);

    void resize(int _n_rows, int _n_cols);

    virtual void resize_data(int size) = 0;
    virtual void* get_data() = 0;

    aligned_vector<int> idx1;
    aligned_vector<int> idx2;
    aligned_vector<double> vals;

    int b_rows;
    int b_cols;
    int b_size;

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
 *****     Returns aligned_vector<int>& containing the rows corresponding
 *****     to each nonzero
 ***** cols()
 *****     Returns aligned_vector<int>& containing the cols corresponding
 *****     to each nonzero
 ***** data()
 *****     Returns aligned_vector<double>& containing the nonzero values
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
        int _nnz = nnz_per_row * _nrows;
        if (_nnz)
        {
            idx1.reserve(_nnz);
            idx2.reserve(_nnz);
            vals.reserve(_nnz);
        }
    }

    COOMatrix(int _nrows, int _ncols, double* _data) : Matrix(_nrows, _ncols)
    {
        init_from_dense(_data);
    }

    COOMatrix(int _nrows, int _ncols, aligned_vector<int>& rows, aligned_vector<int>& cols, 
            aligned_vector<double>& data) : Matrix(_nrows, _ncols)
    {
        init_from_lists(rows, cols, data);
    }

    COOMatrix()
    {
    }

    ~COOMatrix()
    {

    }

    template <typename T>
    void init_from_dense(T* _data)
    {
        nnz = 0;
        int nnz_dense = n_rows*n_cols;

        if (nnz_dense)
        {
            idx1.resize(nnz_dense);
            idx2.resize(nnz_dense);
            resize_data(nnz_dense);
        }

        T* val_list = (T*) get_data();

        for (int i = 0; i < n_rows; i++)
        {
            for (int j = 0; j < n_cols; j++)
            {
                int pos = i * n_cols + j;
                if (abs_val(_data[pos]) > zero_tol)
                {
                    idx1[nnz] = i;
                    idx2[nnz] = j;
                    val_list[nnz] = copy_val(_data[pos]);
                    nnz++;
                }
            }
        }
    }

    COOMatrix* transpose();

    void print();
    void copy_helper(const COOMatrix* A);
    void copy_helper(const CSRMatrix* A);
    void copy_helper(const CSCMatrix* A);

    void sort();
    void move_diag();
    void remove_duplicates();

    void spmv_append(const aligned_vector<double>& x, aligned_vector<double>& b);
    void spmv_append_T(const aligned_vector<double>& x, aligned_vector<double>& b);
    void spmv_append_neg(const aligned_vector<double>& x, aligned_vector<double>& b);
    void spmv_append_neg_T(const aligned_vector<double>& x, aligned_vector<double>& b);

    CSRMatrix* spgemm(const CSRMatrix* B)
    {
        return NULL;
    }
    CSRMatrix* spgemm_T(const CSCMatrix* A)
    {
        return NULL;
    }

    COOMatrix* to_COO();
    CSRMatrix* to_CSR();
    CSCMatrix* to_CSC();

    COOMatrix* copy()
    {
        COOMatrix* A = new COOMatrix();
        A->copy_helper(this);
        return A;
    }
    
    void add_value(int row, int col, double value)
    {
        idx1.push_back(row);
        idx2.push_back(col);
        vals.push_back(value);
        nnz++;
    }

    format_t format()
    {
        return COO;
    }

    void* get_data()
    {
       return vals.data();
    } 

    void resize_data(int size)
    {
        vals.resize(size);
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
 *****     Returns aligned_vector<int>& row pointer.  The ith element points to
 *****     the index of indices() corresponding to the first column to lie on 
 *****     row i.
 ***** indices()
 *****     Returns aligned_vector<int>& containing the cols corresponding
 *****     to each nonzero
 ***** data()
 *****     Returns aligned_vector<double>& containing the nonzero values
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
        init_from_dense(_data);
    }

    CSRMatrix(int _nrows, int _ncols, aligned_vector<int>& rowptr, 
            aligned_vector<int>& cols, aligned_vector<double>& data) : Matrix(_nrows, _ncols)
    {
        init_from_lists(rowptr, cols, data);
    }

    CSRMatrix()
    {
    }

    ~CSRMatrix()
    {

    }

    template <typename T>
    void init_from_dense(T* _data)
    {
        int nnz_dense = n_rows*n_cols;
        idx1.resize(n_rows + 1);
        if (nnz_dense)
        {
            idx2.resize(nnz_dense);
            resize_data(nnz_dense);
        }

        T* val_list = (T*) get_data();

        idx1[0] = 0;
        for (int i = 0; i < n_rows; i++)
        {
            for (int j = 0; j < n_cols; j++)
            {
                int pos = i * n_cols + j;
                if (abs_val(_data[pos]))
                {
                    idx2[nnz] = j;
                    val_list[nnz] = copy_val(_data[pos]);
                    nnz++;
                }
            }
            idx1[i+1] = nnz;
        }
    }

    CSRMatrix* transpose();

    void print();

    void copy_helper(const COOMatrix* A);
    void copy_helper(const CSRMatrix* A);
    void copy_helper(const CSCMatrix* A);

    void sort();
    void move_diag();
    void remove_duplicates();

    void spmv_append(const aligned_vector<double>& x, aligned_vector<double>& b);
    void spmv_append_T(const aligned_vector<double>& x, aligned_vector<double>& b);
    void spmv_append_neg(const aligned_vector<double>& x, aligned_vector<double>& b);
    void spmv_append_neg_T(const aligned_vector<double>& x, aligned_vector<double>& b);

    CSRMatrix* spgemm(const CSRMatrix* B);
    CSRMatrix* spgemm_T(const CSCMatrix* A);

    CSRMatrix* add(CSRMatrix* A);
    CSRMatrix* subtract(CSRMatrix* A);

    CSRMatrix* strength(strength_t strength_type = Classical,
            double theta = 0.0, int num_variables = 1, int* variables = NULL);
    CSRMatrix* aggregate();
    CSRMatrix* fit_candidates(data_t* B, data_t* R, int num_candidates, 
            double tol = 1e-10);

    COOMatrix* to_COO();
    CSRMatrix* to_CSR();
    CSCMatrix* to_CSC();
    CSRMatrix* copy()
    {
        CSRMatrix* A = new CSRMatrix();
        A->copy_helper(this);
        return A;
    }

    format_t format()
    {
        return CSR;
    }

    void add_value(int row, int col, double value) 
    {
        idx2.push_back(col);
        vals.push_back(value);
        nnz++;
    }

    void* get_data()
    {
       return vals.data();
    } 
    void resize_data(int size)
    {
        vals.resize(size);
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
 *****     Returns aligned_vector<int>& column pointer.  The ith element points to
 *****     the index of indices() corresponding to the first row to lie on 
 *****     column i.
 ***** indices()
 *****     Returns aligned_vector<int>& containing the rows corresponding
 *****     to each nonzero
 ***** data()
 *****     Returns aligned_vector<double>& containing the nonzero values
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
        init_from_dense(_data);
    }

    CSCMatrix(int _nrows, int _ncols, aligned_vector<int>& colptr, 
            aligned_vector<int>& rows, aligned_vector<double>& data) : Matrix(_nrows, _ncols)
    {
        init_from_lists(colptr, rows, data);
    }

    CSCMatrix()
    {
    }

    ~CSCMatrix()
    {

    }

    template <typename T>
    void init_from_dense(T* _data)
    {
        int nnz_dense = n_rows*n_cols;

        idx1.resize(n_cols + 1);
        if (nnz_dense)
        {
            idx2.resize(nnz_dense);
            resize_data(nnz_dense);
        }

        T* val_list = (T*) get_data();

        idx1[0] = 0;
        for (int i = 0; i < n_cols; i++)
        {
            for (int j = 0; j < n_rows; j++)
            {
                int pos = i * n_cols + j;
                if (abs_val(_data[pos]) > zero_tol)
                {
                    idx2[nnz] = j;
                    val_list[nnz] = copy_val(_data[pos]);
                    nnz++;
                }
            }
            idx1[i+1] = nnz;
        }

    }

    CSCMatrix* transpose();
    void print();

    void copy_helper(const COOMatrix* A);
    void copy_helper(const CSRMatrix* A);
    void copy_helper(const CSCMatrix* A);

    void sort();
    void move_diag();
    void remove_duplicates();

    void spmv_append(const aligned_vector<double>& x, aligned_vector<double>& b);
    void spmv_append_T(const aligned_vector<double>& x, aligned_vector<double>& b);
    void spmv_append_neg(const aligned_vector<double>& x, aligned_vector<double>& b);
    void spmv_append_neg_T(const aligned_vector<double>& x, aligned_vector<double>& b);


    CSRMatrix* spgemm(const CSRMatrix* B)
    {
        return NULL;
    }
    CSRMatrix* spgemm_T(const CSCMatrix* A)
    {
        return NULL;
    }

    void jacobi(Vector& x, Vector& b, Vector& tmp, double omega = .667);    

    COOMatrix* to_COO();
    CSRMatrix* to_CSR();
    CSCMatrix* to_CSC();
    CSCMatrix* copy()
    {
        CSCMatrix* A = new CSCMatrix();
        A->copy_helper(this);
        return A;
    }

    format_t format()
    {
        return CSC;
    }

    void add_value(int row, int col, double value)
    {
        idx2.push_back(row);
        vals.push_back(value);
        nnz++;
    }

    void* get_data()
    {
       return vals.data();
    } 
    void resize_data(int size)
    {
        vals.resize(size);
    }

  };





// Forward Declaration of Blocked Classes 
class BCOOMatrix;
class BSRMatrix;
class BSCMatrix;

class BCOOMatrix : public COOMatrix
{
  public:
    BCOOMatrix(int num_block_rows, int num_block_cols, int block_row_size, 
            int block_col_size, int nnz_per_block_row = 1) 
        : COOMatrix(num_block_rows, num_block_cols, 0)
    {
        b_rows = block_row_size;
        b_cols = block_col_size;
        b_size = b_rows * b_cols;
    }

    BCOOMatrix(int num_block_rows, int num_block_cols,
            int block_row_size, int block_col_size, double** values) 
        : COOMatrix(num_block_rows, num_block_cols, 0)
    {
        b_rows = block_row_size;
        b_cols = block_col_size;
        b_size = b_rows * b_cols;
        
        init_from_dense(values); 
    }

    BCOOMatrix(int num_block_rows, int num_block_cols,
            int block_row_size, int block_col_size,
            aligned_vector<int>& rows, aligned_vector<int>& cols, 
            aligned_vector<double*>& data)
       : COOMatrix(num_block_rows, num_block_cols, 0) 
    {
        b_rows = block_row_size;
        b_cols = block_col_size;
        b_size = b_rows * b_cols;

        init_from_lists(rows, cols, data);
    }

    BCOOMatrix() : COOMatrix()
    {
        b_rows = 1;
        b_cols = 1;
        b_size = 1;
    }

    ~BCOOMatrix()
    {
        for (int i = 0; i < nnz; i++)
        {
            double* val_ptr = vals[i];
            delete[] val_ptr;
        }
    }

    BCOOMatrix* transpose();

    BCOOMatrix* copy()
    {
        BCOOMatrix* A = new BCOOMatrix();
        A->copy_helper(this);
        return A;
    }
    COOMatrix* to_COO();
    CSRMatrix* to_CSR();
    CSCMatrix* to_CSC();

    void copy_helper(const BCOOMatrix* A);
    void copy_helper(const BSRMatrix* A);
    void copy_helper(const BSCMatrix* A);

    void add_value(int row, int col, double* values)
    {
        idx1.push_back(row);
        idx2.push_back(col);
        vals.push_back(copy_val(values));
        nnz++;
    }

    format_t format()
    {
        return BCOO;
    }

    void* get_data()
    {
       return vals.data();
    } 
    void resize_data(int size)
    {
        vals.resize(size);
    }

    int b_rows;
    int b_cols;
    int b_size;

    aligned_vector<double*> vals;
};

class BSRMatrix : public CSRMatrix
{
  public:
    BSRMatrix(int num_block_rows, int num_block_cols, int block_row_size, 
            int block_col_size, int _nnz = 1) 
        : CSRMatrix(num_block_rows, num_block_cols, 0)
    {
        b_rows = block_row_size;
        b_cols = block_col_size;
        b_size = b_rows * b_cols;
    }

    BSRMatrix(int num_block_rows, int num_block_cols, 
            int block_row_size, int block_col_size, double** data)
        :  CSRMatrix(num_block_rows, num_block_cols, 0)
    {
        b_rows = block_row_size;
        b_cols = block_col_size;
        b_size = b_rows * b_cols;

        init_from_dense(data);
    }


    BSRMatrix(int num_block_rows, int num_block_cols, 
            int block_row_size, int block_col_size, aligned_vector<int>& rowptr, 
            aligned_vector<int>& cols, aligned_vector<double*>& data)
        :  CSRMatrix(num_block_rows, num_block_cols, 0)
    {
        b_rows = block_row_size;
        b_cols = block_col_size;
        b_size = b_rows * b_cols;

        init_from_lists(rowptr, cols, data);
    }

    
    BSRMatrix() : CSRMatrix()
    {
        b_rows = 1;
        b_cols = 1;
        b_size = 1;
    }

    ~BSRMatrix()
    {
        for (int i = 0; i < nnz; i++)
        {
            double* val_ptr = vals[i];
            delete[] val_ptr;
        }
    }

    BSRMatrix* transpose();

    COOMatrix* to_COO();
    CSRMatrix* to_CSR();
    CSCMatrix* to_CSC();

    BSRMatrix* copy()
    {
        BSRMatrix* A = new BSRMatrix();
        A->copy_helper(this);
        return A;
    }

    void copy_helper(const BCOOMatrix* A);
    void copy_helper(const BSRMatrix* A);
    void copy_helper(const BSCMatrix* A);

    format_t format()
    {
        return BSR;
    }

    void add_value(int row, int col, double* value) 
    {
        idx2.push_back(col);
        vals.push_back(copy_val(value));
        nnz++;
    }

    int b_rows;
    int b_cols;
    int b_size;

    void* get_data()
    {
       return vals.data();
    } 
    void resize_data(int size)
    {
        vals.resize(size);
    }

    aligned_vector<double*> vals;
};

// Blocks are still stored row-wise in BSC matrix...
class BSCMatrix : public CSCMatrix
{
  public:
    BSCMatrix(int num_block_rows, int num_block_cols, int block_row_size, 
            int block_col_size, int _nnz = 1) 
        : CSCMatrix(num_block_rows, num_block_cols, 0)
    {
        b_rows = block_row_size;
        b_cols = block_col_size;
        b_size = b_rows * b_cols;
    }

    BSCMatrix(int num_block_rows, int num_block_cols, 
            int block_row_size, int block_col_size, double** data)
        :  CSCMatrix(num_block_rows, num_block_cols, 0)
    {
        b_rows = block_row_size;
        b_cols = block_col_size;
        b_size = b_rows * b_cols;

        init_from_dense(data);
    }


    BSCMatrix(int num_block_rows, int num_block_cols, 
            int block_row_size, int block_col_size, aligned_vector<int>& colptr, 
            aligned_vector<int>& rows, aligned_vector<double*>& data)
        :  CSCMatrix(num_block_rows, num_block_cols, 0)
    {
        b_rows = block_row_size;
        b_cols = block_col_size;
        b_size = b_rows * b_cols;

        init_from_lists(colptr, rows, data);
    }

    
    BSCMatrix() : CSCMatrix()
    {
        b_rows = 1;
        b_cols = 1;
        b_size = 1;
    }

    ~BSCMatrix()
    {
        for (int i = 0; i < nnz; i++)
        {
            double* val_ptr = vals[i];
            delete[] val_ptr;
        }
    }

    BSCMatrix* transpose();

    COOMatrix* to_COO();
    CSRMatrix* to_CSR();
    CSCMatrix* to_CSC();

    BSCMatrix* copy()
    {
        BSCMatrix* A = new BSCMatrix();
        A->copy_helper(this);
        return A;
    }

    void copy_helper(const BCOOMatrix* A);
    void copy_helper(const BSRMatrix* A);
    void copy_helper(const BSCMatrix* A);

    format_t format()
    {
        return BSC;
    }

    int b_rows;
    int b_cols;
    int b_size;

    void add_value(int row, int col, double* value)
    {
        idx2.push_back(row);
        vals.push_back(copy_val(value));
        nnz++;
    }

    void* get_data()
    {
       return vals.data();
    }
    void resize_data(int size)
    {
        vals.resize(size);
    }

    aligned_vector<double*> vals;
};



}

#endif

