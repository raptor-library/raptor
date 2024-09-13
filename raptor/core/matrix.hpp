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
        b_rows = 1;
        b_cols = 1;
        b_size = 1;
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
        b_rows = 1;
        b_cols = 1;
        b_size = 1;
    }

    virtual ~Matrix(){}

    template <typename T>
    void init_from_lists(std::vector<int>& _idx1, std::vector<int>& _idx2, 
            std::vector<T>& data)
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
    virtual CSRMatrix* to_BSR() = 0;
    virtual CSCMatrix* to_BSC() = 0;
    virtual COOMatrix* to_BCOO() = 0;
    virtual void block_removal_col_check(bool* col_check) = 0;
    virtual Matrix* copy() = 0;

    virtual void spmv(const double* x, double* b) const = 0;
    virtual void spmv_append(const double* x, double* b) const = 0;
    virtual void spmv_append_T(const double* x, double* b) const = 0;
    virtual void spmv_append_neg(const double* x, double* b) const = 0;
    virtual void spmv_append_neg_T(const double* x, double* b) const = 0;
    virtual void spmv_residual(const double* x, const double* b, double* r) const = 0;

    virtual CSRMatrix* spgemm(CSRMatrix* B, int* B_to_C = NULL) = 0;
    virtual CSRMatrix* spgemm_T(CSCMatrix* A, int* C_map = NULL) = 0;
    virtual Matrix* transpose() = 0;

    double* get_values(Vector& x) const
    {
        return x.values.data();
    }
    template<typename T> T* get_values(std::vector<T>& x) const
    {
        return x.data();
    }
    template<typename T> T* get_values(T* x) const
    {
        return x;
    }
    
    // Method for printing the value at one position
    // (either single or block value)
    void val_print(int row, int col, double val) const
    {
        printf("A[%d][%d] = %e\n", row, col, val);
    }
    void val_print(int row, int col, double* val) const
    {
        for (int i = 0; i < b_rows; i++)
        {
            for (int j = 0; j < b_cols; j++)
            {
                printf("A[%d][%d], BlockPos[%d][%d] = %e\n", row, col, i, j, val[i*b_cols+j]);
            }
        }
    }

    double copy_val(double val) const
    {
        return val;
    }
    double* copy_val(double* val) const
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
    double abs_val(double val) const
    {
        return fabs(val);
    }
    double abs_val(double* val) const
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
    void append_vals(double* val, double* addl_val) const
    {
        *val += *addl_val;
    }
    void append_vals(double** val, double** addl_val) const
    {
        for (int i = 0; i < b_size; i++)
        {
            *val[i] += *addl_val[i];
        }
        delete[] *addl_val;
    }
    void mult_vals(double val, double addl_val, double* sum, 
            int nr, int nc0, int n_inner) const
    {
        *sum += (val * addl_val);
    }
    void mult_vals(double* val, double* addl_val, double** sum,
            int nr, int nc, int n_inner) const
    {
        for (int i = 0; i < nr; i++) // Go through b_rows of A
        { 
            for (int j = 0; j < nc; j++) // Go through b_cols of B
            {
                double s = 0;
                for (int k = 0; k < n_inner; k++) // Go through b_cols of A (== b_rows of B)
                {
                    s += val[i*n_inner + k] * addl_val[k*n_inner + j];
                }
                (*sum)[i*nc + j] += s;
            }
        }
    }
    void mult_T_vals(double val, double addl_val, double* sum,
            int nr, int nc, int n_inner) const
    {
        *sum += (val * addl_val);
    }
    void mult_T_vals(double* val, double* addl_val, double** sum,
            int nr, int nc, int n_inner) const
    {
        for (int i = 0; i < nr; i++) // Go through b_rows of A
        { 
            for (int j = 0; j < nc; j++) // Go through b_cols of B
            {
                double s = 0;
                for (int k = 0; k < n_inner; k++) // Go through b_cols of A (== b_rows of B)
                {
                    s += val[k*n_inner + i] * addl_val[k*n_inner + j];
                }
                (*sum)[i*nc + j] += s;
            }
        }
    }


    void append(int _idx1, int _idx2, double* b, const double* x, const double val) const
    {
        b[_idx1] += val*x[_idx2];
    }
    void append_T(int _idx1, int _idx2, double* b, const double* x, const double val) const
    {
        b[_idx2] += val*x[_idx1];
    }
    void append_neg(int _idx1, int _idx2, double* b, const double* x, const double val) const
    {
        b[_idx1] -= val*x[_idx2];
    }
    void append_neg_T(int _idx1, int _idx2, double* b, const double* x, const double val) const
    {
        b[_idx2] -= val*x[_idx1];
    }
    void append(int _idx1, int _idx2, double* b, const double* x, const double* val) const
    {
        int first_row = _idx1*b_rows;
        int first_col = _idx2*b_cols;
        for (int row = 0; row < b_rows; row++)
        {
            for (int col = 0; col < b_cols; col++)
            {
                b[first_row + row] += (val[row * b_cols + col] * x[first_col + col]);
            }
        }
    }
    void append_T(int _idx1, int _idx2, double* b, const double* x, const double* val) const
    {
        int first_row = _idx1*b_rows;
        int first_col = _idx2*b_cols;

        for (int row = 0; row < b_rows; row++)
        {
            double x_val = x[first_row + row];
            for (int col = 0; col < b_cols; col++)
            {
                b[first_col + col] += (val[row * b_cols + col] * x_val);
            }
        }
    }
    void append_neg(int _idx1, int _idx2, double* b, const double* x, const double* val) const
    {
        int first_row = _idx1*b_rows;
        int first_col = _idx2*b_cols;
        for (int row = 0; row < b_rows; row++)
        {
            for (int col = 0; col < b_cols; col++)
            {
                b[first_row + row] -= (val[row * b_cols + col] * x[first_col + col]);
            }
        }
    }
    void append_neg_T(int _idx1, int _idx2, double* b, const double* x, const double* val) const
    {
        int first_row = _idx1*b_rows;
        int first_col = _idx1*b_cols;
        for (int row = 0; row < b_rows; row++)
        {
            for (int col = 0; col < b_cols; col++)
            {
                b[first_col + col] -= (val[row * b_cols + col] * x[first_row + row]);
            }
        }
    }

    template <typename T, typename U> void mult(T& x, U& b) const
    {
        spmv(get_values(x), get_values(b));
    }
    template <typename T, typename U> void mult_T(T& x, U& b) const
    {
        int cols = n_cols * b_cols;
        for (int i = 0; i < cols; i++)
            b[i] = 0.0;
        spmv_append_T(get_values(x), get_values(b));
    }
    template <typename T, typename U> void mult_append(T& x, U& b) const
    {
        spmv_append(get_values(x), get_values(b));
    }
    template <typename T, typename U> void mult_append_T(T& x, U& b) const
    {
        spmv_append_T(get_values(x), get_values(b));
    }
    template <typename T, typename U> void mult_append_neg(T& x, U& b) const
    {
        spmv_append_neg(get_values(x), get_values(b));
    }
    template <typename T, typename U> void mult_append_neg_T(T& x, U& b) const
    {
        spmv_append_neg_T(get_values(x), get_values(b));
    }
    template <typename T, typename U, typename V> void residual(T& x, U& b, V& r) const
    {
        spmv_residual(get_values(x), get_values(b), get_values(r));
    }

    CSRMatrix* mult(CSRMatrix* B, int* B_to_C = NULL);
    CSRMatrix* mult(CSCMatrix* B, int* B_to_C = NULL);
    CSRMatrix* mult(COOMatrix* B, int* B_to_C = NULL);
    CSRMatrix* mult_T(CSCMatrix* A, int* C_map = NULL);
    CSRMatrix* mult_T(CSRMatrix* A, int* C_map = NULL);
    CSRMatrix* mult_T(COOMatrix* A, int* C_map = NULL);

    virtual void add_value(int row, int col, double value) = 0;
    virtual void add_value(int row, int col, double* value) = 0;

    Matrix* add(CSRMatrix* A, bool remove_dup = true);
    void add_append(CSRMatrix* A, CSRMatrix* C, bool remove_dup = true);
    Matrix* subtract(CSRMatrix* A);

    void resize(int _n_rows, int _n_cols);

    virtual void resize_data(int size) = 0;
    virtual void* get_data() = 0;
    virtual int data_size() const = 0;
    virtual void reserve_size(int size) = 0;
    virtual double get_val(const int j, const int k) = 0;

    std::vector<int> idx1;
    std::vector<int> idx2;
    std::vector<double> vals;

	std::tuple<decltype(idx1)&, decltype(idx2)&, decltype(vals)&> vecs() {
		return {idx1, idx2, vals};
	}

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

    COOMatrix(int _nrows, int _ncols, std::vector<int>& rows, std::vector<int>& cols, 
            std::vector<double>& data) : Matrix(_nrows, _ncols)
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

    void sort();
    void move_diag();
    void remove_duplicates();

    void spmv(const double* x, double* b) const;
    void spmv_append(const double* x, double* b) const;
    void spmv_append_T(const double* x, double* b) const;
    void spmv_append_neg(const double* x, double* b) const;
    void spmv_append_neg_T(const double* x, double* b) const;
    void spmv_residual(const double* x, const double* b, double* r) const; 

    CSRMatrix* spgemm(CSRMatrix* B, int* B_to_C = NULL);
    CSRMatrix* spgemm_T(CSCMatrix* A, int* C_map = NULL);

    COOMatrix* to_COO();
    CSRMatrix* to_CSR();
    CSCMatrix* to_CSC();
    CSRMatrix* to_BSR();
    CSCMatrix* to_BSC();
    COOMatrix* to_BCOO();

    void block_removal_col_check(bool* col_check);

    COOMatrix* copy();
    
    void add_value(int row, int col, double value)
    {
        if (fabs(value) > zero_tol)
        {
            idx1.emplace_back(row);
            idx2.emplace_back(col);
            vals.emplace_back(value);
            nnz++;
        }
    }

    void add_value(int row, int col, double* value)
    {
        idx1.emplace_back(row);
        idx2.emplace_back(col);
        vals.emplace_back(*value);
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
    int data_size() const
    {
        return vals.size();
    }

    void resize_data(int size)
    {
        vals.resize(size);
    }

    void reserve_size(int size)
    {
        idx1.reserve(size);
        idx2.reserve(size);
        vals.reserve(size);
    }
    
    double get_val(const int j, const int k)
    {
        return vals[j];
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
        init_from_dense(_data);
    }

    CSRMatrix(int _nrows, int _ncols, std::vector<int>& rowptr, 
            std::vector<int>& cols, std::vector<double>& data) : Matrix(_nrows, _ncols)
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

    void sort();
    void move_diag();
    void remove_duplicates();

    void spmv(const double* x, double* b) const;
    void spmv_append(const double* x, double* b) const;
    void spmv_append_T(const double* x, double* b) const;
    void spmv_append_neg(const double* x, double* b) const;
    void spmv_append_neg_T(const double* x, double* b) const;
    void spmv_residual(const double* x, const double* b, double* r) const; 

    CSRMatrix* spgemm(CSRMatrix* B, int* B_to_C = NULL);
    CSRMatrix* spgemm_T(CSCMatrix* A, int* C_map = NULL);

    CSRMatrix* add(CSRMatrix* A, bool remove_dup = true);
    void add_append(CSRMatrix* A, CSRMatrix* C, bool remove_dup = true);
    CSRMatrix* subtract(CSRMatrix* A);

    CSRMatrix* strength(strength_t strength_type = Classical,
            double theta = 0.0, int num_variables = 1, int* variables = NULL);
    CSRMatrix* aggregate();
    CSRMatrix* fit_candidates(data_t* B, data_t* R, int num_candidates, 
            double tol = 1e-10);

    COOMatrix* to_COO();
    CSRMatrix* to_CSR();
    CSCMatrix* to_CSC();
    CSRMatrix* to_BSR();
    CSCMatrix* to_BSC();
    COOMatrix* to_BCOO();

    void block_removal_col_check(bool* col_check);

    CSRMatrix* copy();

    format_t format()
    {
        return CSR;
    }

    void add_value(int row, int col, double value) 
    {
        if (fabs(value) > zero_tol)
        {
            idx2.emplace_back(col);
            vals.emplace_back(value);
            nnz++;
        }
    }
    void add_value(int row, int col, double* value)
    {
        idx2.emplace_back(col);
        vals.emplace_back(*value);
        nnz++;
    }

    void* get_data()
    {
       return vals.data();
    } 
    int data_size() const
    {
        return vals.size();
    }
    void resize_data(int size)
    {
        vals.resize(size);
    }
    void reserve_size(int size)
    {
        idx2.reserve(size);
        vals.reserve(size);
    }

    double get_val(const int j, const int k)
    {
        return vals[j];
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
        init_from_dense(_data);
    }

    CSCMatrix(int _nrows, int _ncols, std::vector<int>& colptr, 
            std::vector<int>& rows, std::vector<double>& data) : Matrix(_nrows, _ncols)
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

    void sort();
    void move_diag();
    void remove_duplicates();

    void spmv(const double* x, double* b) const;
    void spmv_append(const double* x, double* b) const;
    void spmv_append_T(const double* x, double* b) const;
    void spmv_append_neg(const double* x, double* b) const;
    void spmv_append_neg_T(const double* x, double* b) const;
    void spmv_residual(const double* x, const double* b, double* r) const; 


    CSRMatrix* spgemm(CSRMatrix* B, int* B_to_C = NULL);
    CSRMatrix* spgemm_T(CSCMatrix* A, int* C_map = NULL);

    void jacobi(Vector& x, Vector& b, Vector& tmp, double omega = .667);    

    COOMatrix* to_COO();
    CSRMatrix* to_CSR();
    CSCMatrix* to_CSC();
    CSRMatrix* to_BSR();
    CSCMatrix* to_BSC();
    COOMatrix* to_BCOO();

    void block_removal_col_check(bool* col_check);

    CSCMatrix* copy();

    format_t format()
    {
        return CSC;
    }

    void add_value(int row, int col, double value)
    {
        if (fabs(value) > zero_tol)
        {
            idx2.emplace_back(row);
            vals.emplace_back(value);
            nnz++;
        }
    }
    void add_value(int row, int col, double* value)
    {
        idx2.emplace_back(row);
        vals.emplace_back(*value);
        nnz++;
    }

    void* get_data()
    {
       return vals.data();
    } 
    int data_size() const
    {
        return vals.size();
    }
    void resize_data(int size)
    {
        vals.resize(size);
    }
    void reserve_size(int size)
    {
        idx2.reserve(size);
        vals.reserve(size);
    }

    double get_val(const int j, const int k)
    {
        return vals[j];
    }

  };





// Forward Declaration of Blocked Classes 
class BCOOMatrix;
class BSRMatrix;
class BSCMatrix;

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
            int block_row_size, int block_col_size, std::vector<int>& rowptr, 
            std::vector<int>& cols, std::vector<double*>& data)
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
        for (std::vector<double*>::iterator it = block_vals.begin();
                it != block_vals.end(); ++it)
            delete[] *it;
    }

    BSRMatrix* transpose();
    void sort();
    void remove_duplicates();
    void move_diag();

    COOMatrix* to_COO();
    CSRMatrix* to_CSR();
    CSCMatrix* to_CSC();
    CSRMatrix* to_BSR();
    CSCMatrix* to_BSC();
    COOMatrix* to_BCOO();

    void block_removal_col_check(bool* col_check);

    void print();
    BSRMatrix* copy();

    BSRMatrix* spgemm(CSRMatrix* B, int* B_to_C = NULL);
    BSRMatrix* spgemm_T(CSCMatrix* A, int* C_map = NULL);

    void spmv(const double* x, double* b) const;
    void spmv_append(const double* x, double* b) const;
    void spmv_append_T(const double* x, double* b) const;
    void spmv_append_neg(const double* x, double* b) const;
    void spmv_append_neg_T(const double* x, double* b) const;
    void spmv_residual(const double* x, const double* b, double* r) const;

    void add_append(BSRMatrix * A, BSRMatrix * C, bool remove_dup = true);

    format_t format()
    {
        return BSR;
    }

    void add_value(int row, int col, double* value) 
    {
        idx2.emplace_back(col);
        block_vals.emplace_back(copy_val(value));
        nnz++;
    }

    void* get_data()
    {
       return block_vals.data();
    } 
    int data_size() const
    {
        return block_vals.size();
    }
    void resize_data(int size)
    {
        block_vals.resize(size);
    }
    void reserve_size(int size)
    {
        idx2.reserve(size);
        block_vals.reserve(size);
    }

    double get_val(const int j, const int k)
    {
        return block_vals[j][k];
    }

    std::vector<double*> block_vals;
};

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
            std::vector<int>& rows, std::vector<int>& cols, 
            std::vector<double*>& data)
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
        for (std::vector<double*>::iterator it = block_vals.begin();
                it != block_vals.end(); ++it)
            delete[] *it;
    }

    BCOOMatrix* transpose();
    void sort();
    void remove_duplicates();
    void move_diag();

    void print();
    BCOOMatrix* copy();
    COOMatrix* to_COO();
    CSRMatrix* to_CSR();
    CSCMatrix* to_CSC();
    CSRMatrix* to_BSR();
    CSCMatrix* to_BSC();
    COOMatrix* to_BCOO();

    void block_removal_col_check(bool* col_check);

    BSRMatrix* spgemm(CSRMatrix* B, int* B_to_C = NULL);
    BSRMatrix* spgemm_T(CSCMatrix* A, int* C_map = NULL);

    void spmv(const double* x, double* b) const;
    void spmv_append(const double* x, double* b) const;
    void spmv_append_T(const double* x, double* b) const;
    void spmv_append_neg(const double* x, double* b) const;
    void spmv_append_neg_T(const double* x, double* b) const;
    void spmv_residual(const double* x, const double* b, double* r) const; 

    void add_value(int row, int col, double* values)
    {
        idx1.emplace_back(row);
        idx2.emplace_back(col);
        block_vals.emplace_back(copy_val(values));
        nnz++;
    }

    format_t format()
    {
        return BCOO;
    }

    void* get_data()
    {
       return block_vals.data();
    } 
    int data_size() const
    {
        return block_vals.size();
    }
    void resize_data(int size)
    {
        block_vals.resize(size);
    }
    void reserve_size(int size)
    {
        idx1.reserve(size);
        idx2.reserve(size);
        block_vals.reserve(size);
    }

    double get_val(const int j, const int k)
    {
        return block_vals[j][k];
    }

    std::vector<double*> block_vals;
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
            int block_row_size, int block_col_size, std::vector<int>& colptr, 
            std::vector<int>& rows, std::vector<double*>& data)
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
        for (std::vector<double*>::iterator it = block_vals.begin();
                it != block_vals.end(); ++it)
            delete[] *it;
    }

    BSCMatrix* transpose();
    void sort();
    void remove_duplicates();
    void move_diag();

    COOMatrix* to_COO();
    CSRMatrix* to_CSR();
    CSCMatrix* to_CSC();
    CSRMatrix* to_BSR();
    CSCMatrix* to_BSC();
    COOMatrix* to_BCOO();
    
    void block_removal_col_check(bool* col_check);

    void print();
    BSCMatrix* copy();

    BSRMatrix* spgemm(CSRMatrix* B, int* B_to_C = NULL);
    BSRMatrix* spgemm_T(CSCMatrix* A, int* C_map = NULL);

    void spmv(const double* x, double* b) const;
    void spmv_append(const double* x, double* b) const;
    void spmv_append_T(const double* x, double* b) const;
    void spmv_append_neg(const double* x, double* b) const;
    void spmv_append_neg_T(const double* x, double* b) const;
    void spmv_residual(const double* x, const double* b, double* r) const; 

    format_t format()
    {
        return BSC;
    }

    void add_value(int row, int col, double* value)
    {
        idx2.emplace_back(row);
        block_vals.emplace_back(copy_val(value));
        nnz++;
    }

    void* get_data()
    {
       return block_vals.data();
    }
    void resize_data(int size)
    {
        block_vals.resize(size);
    }
    int data_size() const
    {
        return block_vals.size();
    }
    void reserve_size(int size)
    {
        idx2.reserve(size);
        block_vals.reserve(size);
    }

    double get_val(const int j, const int k)
    {
        return block_vals[j][k];
    }

    std::vector<double*> block_vals;
};



}

#endif

