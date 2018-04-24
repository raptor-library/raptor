// Copyright (c) 2015-2017, RAPtor Developer Team
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
#include "partition.hpp"

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
  class ParComm;
  class TAPComm;
  class ParCOOMatrix;
  class ParCSRMatrix;
  class ParCSCMatrix;
  class ParBSRMatrix;

  class ParMatrix
  {
  public:
    ParMatrix(Partition* part)
    {
        partition = part;
        partition->num_shared++;

        global_num_rows = partition->global_num_rows;
        global_num_cols = partition->global_num_cols;
        on_proc_num_cols = partition->local_num_cols;
        local_num_rows = partition->local_num_rows;

        comm = NULL;
        tap_comm = NULL;

        spgemm_data = (OpData) {0.0, 0.0, 0.0, 0.0};
        spgemm_T_data = (OpData) {0.0, 0.0, 0.0, 0.0};
        spmv_data = (OpData) {0.0, 0.0, 0.0, 0.0};;
        spmv_T_data = (OpData) {0.0, 0.0, 0.0, 0.0};
    }

    ParMatrix(Partition* part, index_t glob_rows, index_t glob_cols, int local_rows, 
            int on_proc_cols)
    {
        partition = part;
        partition->num_shared++;

        global_num_rows = glob_rows;
        global_num_cols = glob_cols;
        on_proc_num_cols = on_proc_cols;
        local_num_rows = local_rows;

        comm = NULL;
        tap_comm = NULL;

        spgemm_data = (OpData) {0.0, 0.0, 0.0, 0.0};
        spgemm_T_data = (OpData) {0.0, 0.0, 0.0, 0.0};
        spmv_data = (OpData) {0.0, 0.0, 0.0, 0.0};;
        spmv_T_data = (OpData) {0.0, 0.0, 0.0, 0.0};
    }

    ParMatrix(index_t glob_rows,
            index_t glob_cols)
    {
        partition = new Partition(glob_rows, glob_cols);

        global_num_rows = partition->global_num_rows;
        global_num_cols = partition->global_num_cols;
        on_proc_num_cols = partition->local_num_cols;
        local_num_rows = partition->local_num_rows;

        comm = NULL;
        tap_comm = NULL;

        spgemm_data = (OpData) {0.0, 0.0, 0.0, 0.0};
        spgemm_T_data = (OpData) {0.0, 0.0, 0.0, 0.0};
        spmv_data = (OpData) {0.0, 0.0, 0.0, 0.0};;
        spmv_T_data = (OpData) {0.0, 0.0, 0.0, 0.0};
    }

    // For BSR Matrix class use only
    ParMatrix(index_t glob_rows, index_t glob_cols, int _brows, int _bcols)
    {
        partition = new Partition(glob_rows, glob_cols, _brows, _bcols);

        global_num_rows = partition->global_num_rows;
        global_num_cols = partition->global_num_cols;
        on_proc_num_cols = partition->local_num_cols;
        local_num_rows = partition->local_num_rows;

        comm = NULL;
        tap_comm = NULL;

        spgemm_data = (OpData) {0.0, 0.0, 0.0, 0.0};
        spgemm_T_data = (OpData) {0.0, 0.0, 0.0, 0.0};
        spmv_data = (OpData) {0.0, 0.0, 0.0, 0.0};;
        spmv_T_data = (OpData) {0.0, 0.0, 0.0, 0.0};
    }

    ParMatrix(index_t glob_rows, 
            index_t glob_cols, 
            int local_rows, 
            int local_cols, 
            index_t first_row, 
            index_t first_col, 
            Topology* topology = NULL)
    {
        partition = new Partition(glob_rows, glob_cols,
                local_rows, local_cols, first_row, first_col, topology);

        global_num_rows = partition->global_num_rows;
        global_num_cols = partition->global_num_cols;
        on_proc_num_cols = partition->local_num_cols;
        local_num_rows = partition->local_num_rows;

        comm = NULL;
        tap_comm = NULL;

        spgemm_data = (OpData) {0.0, 0.0, 0.0, 0.0};
        spgemm_T_data = (OpData) {0.0, 0.0, 0.0, 0.0};
        spmv_data = (OpData) {0.0, 0.0, 0.0, 0.0};;
        spmv_T_data = (OpData) {0.0, 0.0, 0.0, 0.0};
    }
       
    ParMatrix()
    {
        local_num_rows = 0;
        global_num_rows = 0;
        global_num_cols = 0;
        off_proc_num_cols = 0;
        on_proc_num_cols = 0;

        comm = NULL;
        tap_comm = NULL;

        on_proc = NULL;
        off_proc = NULL;

        partition = NULL;

        spgemm_data = (OpData) {0.0, 0.0, 0.0, 0.0};
        spgemm_T_data = (OpData) {0.0, 0.0, 0.0, 0.0};
        spmv_data = (OpData) {0.0, 0.0, 0.0, 0.0};;
        spmv_T_data = (OpData) {0.0, 0.0, 0.0, 0.0};
    }

    virtual ~ParMatrix()
    {
        delete off_proc;
        delete on_proc;
        delete comm;
        delete tap_comm;

        if (partition)
        {
            if (partition->num_shared)
            {
                partition->num_shared--;
            }
            else
            {
                delete partition;
            }
        }
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

    int* map_partition_to_local();
    void condense_off_proc();

    void residual(ParVector& x, ParVector& b, ParVector& r);
    void tap_residual(ParVector& x, ParVector& b, ParVector& r);
    void mult(ParVector& x, ParVector& b);
    void tap_mult(ParVector& x, ParVector& b);
    void mult_T(ParVector& x, ParVector& b);
    void tap_mult_T(ParVector& x, ParVector& b);
    ParMatrix* mult(ParCSRMatrix* B);
    ParMatrix* tap_mult(ParCSRMatrix* B);
    ParMatrix* mult_T(ParCSCMatrix* B);
    ParMatrix* mult_T(ParCSRMatrix* B);
    ParMatrix* tap_mult_T(ParCSCMatrix* B);
    ParMatrix* tap_mult_T(ParCSRMatrix* B);
    ParMatrix* add(ParCSRMatrix* A);
    ParMatrix* subtract(ParCSRMatrix* A);

    void sort()
    {
        on_proc->sort();
        off_proc->sort();
    }

    virtual ParMatrix* transpose() = 0;

    std::vector<index_t>& get_off_proc_column_map()
    {
        return off_proc_column_map;
    }

    std::vector<index_t>& get_on_proc_column_map()
    {
        return on_proc_column_map;
    }

    std::vector<index_t>& get_local_row_map()
    {
        return local_row_map;
    }

    virtual void copy(ParCSRMatrix* A) = 0;
    virtual void copy(ParCSCMatrix* A) = 0;
    virtual void copy(ParCOOMatrix* A) = 0;

    virtual void add_block(int global_row_coarse, int global_col_coarse, std::vector<double>& data) = 0;

    // Store dimensions of parallel matrix
    int local_nnz;
    int local_num_rows;
    int global_num_rows;
    int global_num_cols;
    int off_proc_num_cols;
    int on_proc_num_cols;

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
    std::vector<index_t> off_proc_column_map; // Maps off_proc local to global
    std::vector<index_t> on_proc_column_map; // Maps on_proc local to global
    std::vector<index_t> local_row_map; // Maps local rows to global

    // Parallel communication package indicating which 
    // processes hold vector values associated with off_proc,
    // and which processes need vector values from this proc
    Partition* partition;
    ParComm* comm;
    TAPComm* tap_comm;

    struct OpData
    {
        double comm_time;
        double tap_comm_time;
        double time;
        double tap_time;
    };

    OpData spgemm_data;
    OpData spgemm_T_data;
    OpData spmv_data;
    OpData spmv_T_data;

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
        on_proc = new COOMatrix(partition->local_num_rows, partition->local_num_cols, 
                nnz_per_row);
        off_proc = new COOMatrix(partition->local_num_rows, partition->global_num_cols, 
                nnz_per_row);
    }

    ParCOOMatrix(index_t glob_rows, index_t glob_cols, int local_rows, 
            int local_cols, index_t first_row, index_t first_col,
            int nnz_per_row = 5) : ParMatrix(glob_rows, glob_cols,
                local_rows, local_cols, first_row, first_col)
    {
        on_proc = new COOMatrix(partition->local_num_rows, partition->local_num_cols, 
                nnz_per_row);
        off_proc = new COOMatrix(partition->local_num_rows, partition->global_num_cols, 
                nnz_per_row);
    }
    
    ParCOOMatrix(Partition* part, 
            int nnz_per_row = 5) : ParMatrix(part)
    {
        on_proc = new COOMatrix(partition->local_num_rows, partition->local_num_cols, 
                nnz_per_row);
        off_proc = new COOMatrix(partition->local_num_rows, partition->global_num_cols, 
                nnz_per_row);
    }

    ParCOOMatrix(index_t glob_rows, 
            index_t glob_cols, 
            data_t* values) : ParMatrix(glob_rows, glob_cols)
    {
        // Initialize empty diag/offd matrices
        on_proc = new COOMatrix(partition->local_num_rows, partition->local_num_cols, 
                partition->local_num_rows*5);
        off_proc = new COOMatrix(partition->local_num_rows, partition->global_num_cols, 
                partition->local_num_rows*5);

        // Add values to on/off proc matrices
        int val_start = partition->first_local_row * partition->global_num_cols;
        int val_end = (partition->first_local_row + partition->local_num_rows) 
            * partition->global_num_cols;
        for (index_t i = val_start; i < val_end; i++)
        {
            if (fabs(values[i]) > zero_tol)
            {
                int global_col = i % partition->global_num_cols;
                int global_row = i / partition->global_num_cols;
                add_value(global_row - partition->first_local_row, global_col, values[i]);
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
    void mult(ParVector& x, ParVector& b);
    void tap_mult(ParVector& x, ParVector& b);
    void mult_T(ParVector& x, ParVector& b);
    void tap_mult_T(ParVector& x, ParVector& b);

    void add_block(int global_row_coarse, int global_col_coarse, std::vector<double>& data);

    ParMatrix* transpose();
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
        on_proc = new CSRMatrix(partition->local_num_rows, partition->local_num_cols, 
                nnz_per_row);
        off_proc = new CSRMatrix(partition->local_num_rows, partition->global_num_cols, 
                nnz_per_row);
    }

    ParCSRMatrix(index_t glob_rows, index_t glob_cols, int local_rows, 
            int local_cols, index_t first_row, index_t first_col, Topology* topology = NULL,  
            int nnz_per_row = 5) : ParMatrix(glob_rows, glob_cols,
                local_rows, local_cols, first_row, first_col, topology)
    {
        on_proc = new CSRMatrix(partition->local_num_rows, partition->local_num_cols, 
                nnz_per_row);
        off_proc = new CSRMatrix(partition->local_num_rows, partition->global_num_cols, 
                nnz_per_row);
    }

    ParCSRMatrix(Partition* part, 
            int nnz_per_row = 5) : ParMatrix(part)
    {
        on_proc = new CSRMatrix(partition->local_num_rows, partition->local_num_cols, 
                nnz_per_row);
        off_proc = new CSRMatrix(partition->local_num_rows, partition->global_num_cols, 
                nnz_per_row);
    }

    ParCSRMatrix(Partition* part, Matrix* _on_proc, Matrix* _off_proc) : ParMatrix(part)
    {
        on_proc = _on_proc;
        off_proc = _off_proc;
        on_proc_num_cols = on_proc->n_cols;
        off_proc_num_cols = off_proc->n_cols;
        local_num_rows = on_proc->n_rows;
        finalize();
    }


    ParCSRMatrix(Partition* part, index_t glob_rows, index_t glob_cols, int local_rows,
            int on_proc_cols, int off_proc_cols, int nnz_per_row = 5) : ParMatrix(part, 
                glob_rows, glob_cols, local_rows, on_proc_cols)
    {
        off_proc_num_cols = off_proc_cols;
        on_proc = new CSRMatrix(local_num_rows, on_proc_cols, nnz_per_row);
        off_proc = new CSRMatrix(local_num_rows, off_proc_num_cols, nnz_per_row);
    }

    ParCSRMatrix(index_t glob_rows, 
            index_t glob_cols, 
            data_t* values) : ParMatrix(glob_rows, glob_cols)
    {
        // Initialize empty diag/offd matrices
        on_proc = new COOMatrix(partition->local_num_rows, partition->local_num_cols, 
                partition->local_num_rows*5);
        off_proc = new COOMatrix(partition->local_num_rows, partition->global_num_cols, 
                partition->local_num_rows*5);

        // Add values to on/off proc matrices
        on_proc->idx1[0] = 0;
        off_proc->idx1[0] = 0;

        int val_start = partition->first_local_row * partition->global_num_cols;
        for (int i = 0; i < partition->local_num_rows; i++)
        {
            for (int j = 0; j < partition->global_num_cols; j++)
            {
                int idx = val_start + (i*partition->global_num_cols) + j;

                if (fabs(values[idx]) > zero_tol)
                {
                    int global_col = idx % partition->global_num_cols;
                    
                    if (global_col >= partition->first_local_col && 
                        global_col < partition->first_local_col + partition->local_num_cols)
                    {
                        on_proc->idx2.push_back(global_col - partition->first_local_col);
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

    ParCSRMatrix* strength(double theta = 0.0, int num_variables = 1, int* variables = NULL);
    ParCSRMatrix* aggregate();
    ParCSRMatrix* fit_candidates(double* B, double* R, int num_candidates, 
            double tol = 1e-10);
    int maximal_independent_set(std::vector<int>& local_states,
            std::vector<int>& off_proc_states, int max_iters = -1);

    void mult(ParVector& x, ParVector& b);
    void tap_mult(ParVector& x, ParVector& b);
    void mult_T(ParVector& x, ParVector& b);
    void tap_mult_T(ParVector& x, ParVector& b);
    ParCSRMatrix* mult(ParCSRMatrix* B);
    ParCSRMatrix* tap_mult(ParCSRMatrix* B);
    ParCSRMatrix* mult_T(ParCSCMatrix* A);
    ParCSRMatrix* mult_T(ParCSRMatrix* A);
    ParCSRMatrix* tap_mult_T(ParCSCMatrix* A);
    ParCSRMatrix* tap_mult_T(ParCSRMatrix* A);
    ParCSRMatrix* add(ParCSRMatrix* A);
    ParCSRMatrix* subtract(ParCSRMatrix* B);
    
    void mult_helper(ParCSRMatrix* B, ParCSRMatrix* C, CSRMatrix* recv);
    CSRMatrix* mult_T_partial(ParCSCMatrix* A);
    CSRMatrix* mult_T_partial(CSCMatrix* A_off);
    void mult_T_combine(ParCSCMatrix* A, ParCSRMatrix* C, CSRMatrix* recv_on,
            CSRMatrix* recv_off);
    
    void add_block(int global_row_coarse, int global_col_coarse, std::vector<double>& data);

    ParMatrix* transpose();
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
        on_proc = new CSCMatrix(partition->local_num_rows, partition->local_num_cols, 
                nnz_per_row);
        off_proc = new CSCMatrix(partition->local_num_rows, partition->global_num_cols, 
                nnz_per_row);
    }

    ParCSCMatrix(index_t glob_rows, index_t glob_cols, int local_num_rows, 
            int local_num_cols, index_t first_local_row, index_t first_col, 
            int nnz_per_row = 5) : ParMatrix(glob_rows, glob_cols,
                local_num_rows, local_num_cols, first_local_row, first_col)
    {
        on_proc = new CSCMatrix(partition->local_num_rows, partition->local_num_cols, 
                nnz_per_row);
        off_proc = new CSCMatrix(partition->local_num_rows, partition->global_num_cols, 
                nnz_per_row);
    }

    ParCSCMatrix(Partition* part, 
            int nnz_per_row = 5) : ParMatrix(part)
    {
        on_proc = new CSRMatrix(partition->local_num_rows, partition->local_num_cols, 
                nnz_per_row);
        off_proc = new CSRMatrix(partition->local_num_rows, partition->global_num_cols, 
                nnz_per_row);
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

    void mult(ParVector& x, ParVector& b);
    void tap_mult(ParVector& x, ParVector& b);
    void mult_T(ParVector& x, ParVector& b);
    void tap_mult_T(ParVector& x, ParVector& b);

    void add_block(int global_row_coarse, int global_col_coarse, std::vector<double>& data);

    ParMatrix* transpose();
  };

  class ParBSRMatrix : public ParMatrix
  {
  public:
    ParBSRMatrix() : ParMatrix()
    {
        on_proc = new BSRMatrix(0, 0, 0, 0);
        off_proc = new BSRMatrix(0, 0, 0, 0);

	b_rows = 0;
	b_cols = 0;
	b_size = 0;
    }

    ParBSRMatrix(index_t glob_rows, 
            index_t glob_cols, int _brows, int _bcols,
            int blocks_per_row = 5) : ParMatrix(glob_rows, glob_cols, _brows, _bcols)
    {

	if ( glob_rows % _brows != 0 || glob_cols % _bcols != 0)
	{
	    printf("Matrix dimensions must be divisible by block dimensions.\n");
	    exit(-1);
	}

        on_proc = new BSRMatrix(partition->local_num_rows, 
		partition->local_num_cols, _brows, _bcols, 
                blocks_per_row);
        off_proc = new BSRMatrix(partition->local_num_rows, 
		partition->global_num_cols, _brows, _bcols, 
                blocks_per_row);

	b_rows = _brows;
	b_cols = _bcols;
	b_size = b_rows * b_cols;
    }

    ParBSRMatrix(index_t glob_rows, index_t glob_cols, int local_rows, 
            int local_cols, int _brows, int _bcols, index_t first_row, index_t first_col, 
	    Topology* topology = NULL, int blocks_per_row = 5) : ParMatrix(glob_rows, glob_cols,
                local_rows, local_cols, first_row, first_col, topology)
    {
	if ( glob_rows % _brows != 0 || glob_cols % _bcols != 0)
	{
	    printf("Matrix dimensions must be divisible by block dimensions.\n");
	    exit(-1);
	}

        on_proc = new BSRMatrix(partition->local_num_rows, 
		partition->local_num_cols, _brows, _bcols, blocks_per_row);
        off_proc = new BSRMatrix(partition->local_num_rows, 
		partition->global_num_cols, _brows, _bcols, blocks_per_row);

	b_rows = _brows;
	b_cols = _bcols;
	b_size = b_rows * b_cols;
    }

    ParBSRMatrix(Partition* part, int _brows, int _bcols,
            int blocks_per_row = 5) : ParMatrix(part)
    {
        on_proc = new BSRMatrix(partition->local_num_rows, 
		partition->local_num_cols, _brows, _bcols, 
                blocks_per_row);
        off_proc = new BSRMatrix(partition->local_num_rows, 
		partition->global_num_cols, _brows, _bcols,
                blocks_per_row);

	b_rows = _brows;
	b_cols = _bcols;
	b_size = b_rows * b_cols;
    }

    ParBSRMatrix(Partition* part, BSRMatrix* _on_proc, BSRMatrix* _off_proc) : ParMatrix(part)
    {
        on_proc = _on_proc;
        off_proc = _off_proc;
        on_proc_num_cols = on_proc->n_cols;
        off_proc_num_cols = off_proc->n_cols;
        local_num_rows = on_proc->n_rows;
        finalize();
    }


    ParBSRMatrix(Partition* part, index_t glob_rows, index_t glob_cols, int local_rows,
	    int _brows, int _bcols, int on_proc_cols, int off_proc_cols, 
	    int blocks_per_row = 5) : ParMatrix(part, 
                glob_rows, glob_cols, local_rows, on_proc_cols)
    {
	if ( glob_rows % _brows != 0 || glob_cols % _bcols != 0)
	{
	    printf("Matrix dimensions must be divisible by block dimensions.\n");
	    exit(-1);
	}

        off_proc_num_cols = off_proc_cols;
        on_proc = new BSRMatrix(local_num_rows, on_proc_cols, _brows, _bcols, blocks_per_row);
        off_proc = new BSRMatrix(local_num_rows, off_proc_num_cols, _brows, _bcols, 
		blocks_per_row);

	b_rows = _brows;
	b_cols = _bcols;
	b_size = b_rows * b_cols;
    }

    // FIX THIS CONSTRUCTOR *****************************************
    ParBSRMatrix(index_t glob_rows, index_t glob_cols, int _brows, int _bcols,
            data_t* values) : ParMatrix(glob_rows, glob_cols, _brows, _bcols)
    {
        if (glob_rows % _brows != 0 || glob_cols % _bcols != 0)
	{
            printf("Matrix dimensions must be divisible by block dimensions.\n");
	    exit(-1);
	}

        // Initialize empty diag/offd matrices
        /*on_proc = new COOMatrix(partition->local_num_rows, partition->local_num_cols, 
                partition->local_num_rows*5);
        off_proc = new COOMatrix(partition->local_num_rows, partition->global_num_cols, 
                partition->local_num_rows*5);*/

	on_proc = new BSRMatrix(partition->local_num_rows, partition->local_num_cols,
			_brows, _bcols);
	off_proc = new BSRMatrix(partition->local_num_rows, partition->global_num_cols,
			_brows, _bcols);

        // ************************************************
        // Add values to on/off proc matrices
        on_proc->idx1[0] = 0;
        off_proc->idx1[0] = 0;

        int val_start = partition->first_local_row * partition->global_num_cols;
        for (int i = 0; i < partition->local_num_rows; i++)
        {
            for (int j = 0; j < partition->global_num_cols; j++)
            {
                int idx = val_start + (i*partition->global_num_cols) + j;

                if (fabs(values[idx]) > zero_tol)
                {
                    int global_col = idx % partition->global_num_cols;
                    
                    if (global_col >= partition->first_local_col && 
                        global_col < partition->first_local_col + partition->local_num_cols)
                    {
                        on_proc->idx2.push_back(global_col - partition->first_local_col);
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

	b_rows = _brows;
	b_cols = _bcols;
	b_size = b_rows * b_cols;
    }
    // *********************************************  

    /*ParBSRMatrix(ParCSRMatrix* A)
    {
        copy(A);
    }

    ParBSRMatrix(ParCSCMatrix* A)
    {
        copy(A);
    }

    ParBSRMatrix(ParCOOMatrix* A)
    {
        copy(A);
    }*/

    void copy(ParCSRMatrix* A);
    void copy(ParCSCMatrix* A);
    void copy(ParCOOMatrix* A);

    // Takes coarse global block indices
    void add_block(int global_row_coarse, int global_col_coarse, std::vector<double>& data);

    //ParCSRMatrix* strength(double theta = 0.0, int num_variables = 1, int* variables = NULL);
    //ParCSRMatrix* aggregate();
    //ParCSRMatrix* fit_candidates(double* B, double* R, int num_candidates, 
    //        double tol = 1e-10);
    //int maximal_independent_set(std::vector<int>& local_states,
    //        std::vector<int>& off_proc_states, int max_iters = -1);

    void mult(ParVector& x, ParVector& b);
    void tap_mult(ParVector& x, ParVector& b);
    void mult_T(ParVector& x, ParVector& b);
    void tap_mult_T(ParVector& x, ParVector& b);
    ParBSRMatrix* mult(ParBSRMatrix* B);
    ParBSRMatrix* tap_mult(ParBSRMatrix* B);
    //ParBSRMatrix* mult_T(ParCSCMatrix* A);
    ParBSRMatrix* mult_T(ParBSRMatrix* A);
    //ParBSRMatrix* tap_mult_T(ParCSCMatrix* A);
    ParBSRMatrix* tap_mult_T(ParBSRMatrix* A);
    //ParBSRMatrix* add(ParCSRMatrix* A);
    //ParBSRMatrix* subtract(ParCSRMatrix* B);
    
    //void mult_helper(ParCSRMatrix* B, ParCSRMatrix* C, CSRMatrix* recv);
    //CSRMatrix* mult_T_partial(ParCSCMatrix* A);
    //CSRMatrix* mult_T_partial(CSCMatrix* A_off);
    //void mult_T_combine(ParCSCMatrix* A, ParCSRMatrix* C, CSRMatrix* recv_on,
    //        CSRMatrix* recv_off);

    ParMatrix* transpose();

    int b_rows;
    int b_cols;
    int b_size;
  };

}
#endif
