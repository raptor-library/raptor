// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef PARMATRIX_HPP
#define PARMATRIX_HPP

#include <mpi.h>
#include <math.h>
#include <set>

#include "matrix.hpp"
#include "par_vector.hpp"
#include "comm_pkg.hpp"
#include "types.hpp"

// Making Par Matrix an abstract Class
/**************************************************************
 *****   ParMatrix Class
 **************************************************************
 ***** This class constructs a parallel matrix object, holding
 ***** a local diagonal matrix, a local off-diagonal block matrix,
 ***** and communication information
 *****
 ***** Attributes
 ***** -------------
 ***** global_num_rows : index_t
 *****    Number of rows in the global parallel matrix
 ***** global_num_cols : index_t
 *****    Number of columns in the parallel matrix
 ***** local_nnz : int
 *****    Number of nonzeros stored locally
 ***** local_num_rows : int
 *****    Number of rows stored locally
 ***** local_num_cols : int 
 *****    Number of columns stored locally
 ***** first_local_row : index_t
 *****    Global index of first row local to process
 ***** first_local_col : index_t
 *****    Global index of first column to fall in local block
 ***** diag : Matrix*
 *****    Matrix storing local diagonal block
 ***** offd : Matrix*
 *****    Matrix storing local off-diagonal block
 ***** offd_num_cols : index_t
 *****    Number of columns in the off-diagonal matrix
 ***** offd_column_map : std::vector<index_t>
 *****    Maps local columns of offd Matrix to global
 ***** comm : ParComm*
 *****    Parallel communicator for matrix
 ***** 
 ***** Methods
 ***** -------
 ***** initalize_partition()
 *****    Determines which rows are local to process and which 
 *****    columns fall in local block 
 ***** add_value()
 *****    Adds a value to a given local row and global column.  
 *****    Determines if this value is in the diagonal or 
 *****    off-diagonal block.
 ***** add_global_value()
 *****    Adds a value to a given global row and global column.  
 *****    Determines if this value is in the diagonal or 
 *****    off-diagonal block.
 ***** finalize()
 *****    Finalizes a matrix after values have been added.  
 *****    Converts the matrices to the appropriate formats and 
 *****    creates the parallel communicator.
 **************************************************************/
namespace raptor
{
  class ParCOOMatrix;
  class ParCSRMatrix;
  class ParCSCMatrix;

  class ParMatrix
  {
  public:
    ParMatrix(index_t glob_rows, 
            index_t glob_cols)
    {
        // Initialize matrix dimensions
        global_num_rows = glob_rows;
        global_num_cols = glob_cols;

        // Initialize matrix partition information
        initialize_partition();
        comm = NULL;
        tap_comm = NULL;
    }

    ParMatrix(index_t glob_rows, 
            index_t glob_cols, 
            int local_rows, 
            int local_cols, 
            index_t first_row, 
            index_t first_col)
    {
        global_num_rows = glob_rows;
        global_num_cols = glob_cols;
        local_num_rows = local_rows;
        local_num_cols = local_cols;
        first_local_row = first_row;
        first_local_col = first_col;

        comm = NULL;
        tap_comm = NULL;
    }
       
    ParMatrix()
    {
        local_num_rows = 0;
        local_num_cols = 0;
        off_proc_num_cols = 0;

        comm = NULL;
        tap_comm = NULL;
    }

    virtual ~ParMatrix()
    {
        delete off_proc;
        delete on_proc;
        delete comm;
        delete tap_comm;
    }


    /**************************************************************
    *****   ParMatrix Add Value
    **************************************************************
    ***** Adds a value to the local portion of the parallel matrix,
    ***** determining whether it should be added to diagonal or 
    ***** off-diagonal block.  
    *****
    ***** Parameters
    ***** -------------
    ***** local_row : index_t
    *****    Local row of value 
    ***** global_col : index_t 
    *****    Global column of value
    ***** value : data_t
    *****    Value to be added to parallel matrix
    **************************************************************/
    void add_value(index_t row, index_t global_col, data_t value);

    /**************************************************************
    *****   ParMatrix Add Global Value
    **************************************************************
    ***** Adds a value to the local portion of the parallel matrix,
    ***** determining whether it should be added to diagonal or 
    ***** off-diagonal block.  
    *****
    ***** Parameters
    ***** -------------
    ***** global_row : index_t
    *****    Global row of value 
    ***** global_col : index_t 
    *****    Global column of value
    ***** value : data_t
    *****    Value to be added to parallel matrix
    **************************************************************/
    void add_global_value(int row, int global_col, double value);

    /**************************************************************
    *****   ParMatrix Finalize
    **************************************************************
    ***** Finalizes the diagonal and off-diagonal matrices.  Sorts
    ***** the local_to_global indices, and creates the parallel
    ***** communicator
    **************************************************************/
    void finalize(bool create_comm = true);

    void initialize_partition();

    void residual(ParVector& x, ParVector& b, ParVector& r);
    void tap_residual(ParVector& x, ParVector& b, ParVector& r);
    void mult(ParVector& x, ParVector& b);
    void tap_mult(ParVector& x, ParVector& b);
    void mult(ParCSRMatrix& B, ParCSRMatrix* C);
    void mult(ParCSCMatrix& B, ParCSCMatrix* C);
    void mult(ParCSCMatrix& B, ParCSRMatrix* C);

    virtual void copy(ParCSRMatrix* A) = 0;
    virtual void copy(ParCSCMatrix* A) = 0;
    virtual void copy(ParCOOMatrix* A) = 0;
    virtual Matrix* communicate(ParComm* comm) = 0;

    // Store dimensions of parallel matrix
    int local_nnz;
    int local_num_rows;
    int local_num_cols;
    index_t global_num_rows;
    index_t global_num_cols;
    index_t first_local_row;
    index_t first_local_col;

    // Store two matrices: on_proc containing columns 
    // corresponding to vector values stored on_process
    // and off_proc columns correspond to vector values
    // stored off process (on other processes)
    Matrix* on_proc; 
    Matrix* off_proc;

    // Store information about columns of off_proc
    // It will be condensed to only store columns with 
    // nonzeros, and these must be mapped to 
    // global column indices
    std::vector<index_t> off_proc_column_map;
    int off_proc_num_cols;

    // Parallel communication package indicating which 
    // processes hold vector values associated with off_proc,
    // and which processes need vector values from this proc
    ParComm* comm;
    TAPComm* tap_comm;

  };

  class ParCOOMatrix : public ParMatrix
  {
  public:
    ParCOOMatrix() : ParMatrix()
    {
        on_proc = new COOMatrix(0, 0, 0);
        off_proc = new COOMatrix(0, 0, 0);
    }

    ParCOOMatrix(index_t glob_rows, 
            index_t glob_cols, 
            int nnz_per_row = 5) : ParMatrix(glob_rows, glob_cols)
    {
        // Initialize diag and offd matrices as COO for adding entries
        // This should later be changed to CSR or CSC
        // A guess of 5 nnz per row is used for reserving matrix space
        on_proc = new COOMatrix(local_num_rows, local_num_cols, nnz_per_row);
        off_proc = new COOMatrix(local_num_rows, global_num_cols, nnz_per_row);
    }

    ParCOOMatrix(index_t glob_rows, index_t glob_cols, int local_rows, 
            int local_cols, index_t first_row, index_t first_col, 
            int nnz_per_row = 5) : ParMatrix(glob_rows, glob_cols,
                local_rows, local_cols, first_row, first_col)
    {
        on_proc = new COOMatrix(local_num_rows, local_num_cols, nnz_per_row);
        off_proc = new COOMatrix(local_num_rows, global_num_cols, nnz_per_row);
    }


    ParCOOMatrix(index_t glob_rows, 
            index_t glob_cols, 
            data_t* values) : ParMatrix(glob_rows, glob_cols)
    {
        // Initialize empty diag/offd matrices
        on_proc = new COOMatrix(local_num_rows, local_num_cols, local_num_rows*5);
        off_proc = new COOMatrix(local_num_rows, global_num_cols, local_num_rows*5);

        // Add values to on/off proc matrices
        int val_start = first_local_row * global_num_cols;
        int val_end = (first_local_row + local_num_rows) * global_num_cols;
        for (index_t i = val_start; i < val_end; i++)
        {
            if (fabs(values[i]) > zero_tol)
            {
                int global_col = i % global_num_cols;
                int global_row = i / global_num_cols;
                add_value(global_row - first_local_row, global_col, values[i]);
            }
        }

        // Convert diag/offd to compressed formats and
        // create parallel communicator
        finalize();
    }  

    ParCOOMatrix(ParCSRMatrix* A)
    {
        copy(A);
    }

    ParCOOMatrix(ParCSCMatrix* A)
    {
        copy(A);
    }

    ParCOOMatrix(ParCOOMatrix* A)
    {
        copy(A);
    }


    void copy(ParCSRMatrix* A);
    void copy(ParCSCMatrix* A);
    void copy(ParCOOMatrix* A);
    Matrix* communicate(ParComm* comm);
    void mult(ParVector& x, ParVector& b);
    void tap_mult(ParVector& x, ParVector& b);
  };

  class ParCSRMatrix : public ParMatrix
  {
  public:
    ParCSRMatrix() : ParMatrix()
    {
        on_proc = new CSRMatrix(0, 0, 0);
        off_proc = new CSRMatrix(0, 0, 0);
    }

    ParCSRMatrix(index_t glob_rows, 
            index_t glob_cols, 
            int nnz_per_row = 5) : ParMatrix(glob_rows, glob_cols)
    {
        // Initialize diag and offd matrices as COO for adding entries
        // This should later be changed to CSR or CSC
        // A guess of 5 nnz per row is used for reserving matrix space
        on_proc = new CSRMatrix(local_num_rows, local_num_cols, nnz_per_row);
        off_proc = new CSRMatrix(local_num_rows, global_num_cols, nnz_per_row);
    }

    ParCSRMatrix(index_t glob_rows, index_t glob_cols, int local_rows, 
            int local_cols, index_t first_row, index_t first_col, 
            int nnz_per_row = 5) : ParMatrix(glob_rows, glob_cols,
                local_rows, local_cols, first_row, first_col)
    {
        on_proc = new CSRMatrix(local_num_rows, local_num_cols, nnz_per_row);
        off_proc = new CSRMatrix(local_num_rows, global_num_cols, nnz_per_row);
    }


    ParCSRMatrix(index_t glob_rows, 
            index_t glob_cols, 
            data_t* values) : ParMatrix(glob_rows, glob_cols)
    {
        // Initialize empty diag/offd matrices
        on_proc = new COOMatrix(local_num_rows, local_num_cols, local_num_rows*5);
        off_proc = new COOMatrix(local_num_rows, global_num_cols, local_num_rows*5);

        // Add values to on/off proc matrices
        on_proc->idx1[0] = 0;
        off_proc->idx1[0] = 0;

        int val_start = first_local_row * global_num_cols;
        for (int i = 0; i < local_num_rows; i++)
        {
            for (int j = 0; j < global_num_cols; j++)
            {
                int idx = val_start + (i*global_num_cols) + j;

                if (fabs(values[idx]) > zero_tol)
                {
                    int global_col = idx % global_num_cols;
                    int global_row = idx / global_num_cols;
                    if (global_col >= first_local_col && 
                        global_col < first_local_col + local_num_cols)
                    {
                        on_proc->idx2.push_back(global_col - first_local_col);
                        on_proc->vals.push_back(values[idx]);
                    }
                    else
                    {
                        off_proc->idx2.push_back(global_col);
                        off_proc->vals.push_back(values[idx]);
                    }

                }
            }
            on_proc->idx1[i+1] = on_proc->idx2.size();
            off_proc->idx1[i+1] = off_proc->idx2.size();
        }
        on_proc->nnz = on_proc->idx2.size();
        off_proc->nnz = off_proc->idx2.size();

        // Convert on/off proc to compressed formats and
        // create parallel communicator
        finalize();
    }  

    ParCSRMatrix(ParCSRMatrix* A)
    {
        copy(A);
    }

    ParCSRMatrix(ParCSCMatrix* A)
    {
        copy(A);
    }

    ParCSRMatrix(ParCOOMatrix* A)
    {
        copy(A);
    }

    void copy(ParCSRMatrix* A);
    void copy(ParCSCMatrix* A);
    void copy(ParCOOMatrix* A);
    Matrix* communicate(ParComm* comm);

    void mult(ParVector& x, ParVector& b);
    void tap_mult(ParVector& x, ParVector& b);
    void mult(ParCSRMatrix& B, ParCSRMatrix* C);
    void mult(ParCSCMatrix& B, ParCSCMatrix* C);
    void mult(ParCSCMatrix& B, ParCSRMatrix* C);
  };

  class ParCSCMatrix : public ParMatrix
  {
  public:
    ParCSCMatrix() : ParMatrix()
    {
        on_proc = new CSCMatrix(0, 0, 0);
        off_proc = new CSCMatrix(0, 0, 0);
    }

    ParCSCMatrix(index_t glob_rows, 
            index_t glob_cols, 
            int nnz_per_row = 5) : ParMatrix(glob_rows, glob_cols)
    {
        // Initialize diag and offd matrices as COO for adding entries
        // This should later be changed to CSR or CSC
        // A guess of 5 nnz per row is used for reserving matrix space
        on_proc = new CSCMatrix(local_num_rows, local_num_cols, nnz_per_row);
        off_proc = new CSCMatrix(local_num_rows, global_num_cols, nnz_per_row);
    }

    ParCSCMatrix(index_t glob_rows, index_t glob_cols, int local_num_rows, 
            int local_num_cols, index_t first_local_row, index_t first_col, 
            int nnz_per_row = 5) : ParMatrix(glob_rows, glob_cols,
                local_num_rows, local_num_cols, first_local_row, first_col)
    {
        on_proc = new CSCMatrix(local_num_rows, local_num_cols, nnz_per_row);
        off_proc = new CSCMatrix(local_num_rows, global_num_cols, nnz_per_row);
    }


    ParCSCMatrix(index_t glob_rows, 
            index_t glob_cols, 
            data_t* values) : ParMatrix(glob_rows, glob_cols)
    {
        printf("Only currently supported for COO and CSR ParMatrices.\n");
    }  

    ParCSCMatrix(ParCSRMatrix* A)
    {
        copy(A);
    }

    ParCSCMatrix(ParCSCMatrix* A)
    {
        copy(A);
    }

    ParCSCMatrix(ParCOOMatrix* A)
    {
        copy(A);
    }


    void copy(ParCSRMatrix* A);
    void copy(ParCSCMatrix* A);
    void copy(ParCOOMatrix* A);
    Matrix* communicate(ParComm* comm);

    void mult(ParVector& x, ParVector& b);
    void tap_mult(ParVector& x, ParVector& b);
    void mult(ParCSCMatrix& B, ParCSCMatrix* C);
  };
}
#endif
