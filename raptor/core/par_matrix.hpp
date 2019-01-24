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
 ***** offd_column_map : aligned_vector<int>
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
  class ParBCOOMatrix;
  class ParCSRMatrix;
  class ParBSRMatrix;
  class ParCSCMatrix;
  class ParBSCMatrix;

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
        tap_mat_comm = NULL;
        shared_comm = false;
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
        tap_mat_comm = NULL;
        shared_comm = false;
    }

    ParMatrix(index_t glob_rows, index_t glob_cols)
    {
        partition = new Partition(glob_rows, glob_cols);

        global_num_rows = partition->global_num_rows;
        global_num_cols = partition->global_num_cols;
        on_proc_num_cols = partition->local_num_cols;
        local_num_rows = partition->local_num_rows;

        comm = NULL;
        tap_comm = NULL;
        tap_mat_comm = NULL;
        shared_comm = false;
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
        tap_mat_comm = NULL;
        shared_comm = false;
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
        tap_mat_comm = NULL;
        shared_comm = false;

        on_proc = NULL;
        off_proc = NULL;

        partition = NULL;
    }

    virtual ~ParMatrix()
    {
        delete off_proc;
        delete on_proc;

        if (!shared_comm)
        {
            delete comm;
            delete tap_mat_comm;
            delete tap_comm;
        }

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
    void finalize(bool create_comm = true); //b_cols added for BSR

    int* map_partition_to_local();
    void condense_off_proc();

    void residual(ParVector& x, ParVector& b, ParVector& r, bool tap = false,
            data_t* comm_t = NULL); 
    void residual_timed(ParVector& x, ParVector& b, ParVector& r, aligned_vector<double>& times,
            bool tap = false, data_t* comm_t = NULL);
    void tap_residual(ParVector& x, ParVector& b, ParVector& r, data_t* comm_t = NULL);
    void mult(ParVector& x, ParVector& b, bool tap = false, data_t* comm_t = NULL);
    void mult_timed(ParVector& x, ParVector& b, aligned_vector<double>& times, bool tap = false, data_t* comm_t = NULL);
    void tap_mult(ParVector& x, ParVector& b, data_t* comm_t = NULL);
    void mult_append(ParVector& x, ParVector& b, bool tap = false, data_t* comm_t = NULL);
    void tap_mult_append(ParVector& x, ParVector& b, data_t* comm_t = NULL);
    void mult_T(ParVector& x, ParVector& b, bool tap = false, data_t* comm_t = NULL);
    void tap_mult_T(ParVector& x, ParVector& b, data_t* comm_t = NULL);
    ParMatrix* mult(ParCSRMatrix* B, bool tap = false, data_t* comm_t = NULL);
    ParMatrix* tap_mult(ParCSRMatrix* B, data_t* comm_t = NULL);
    ParMatrix* mult_T(ParCSCMatrix* B, bool tap = false, data_t* comm_t = NULL);
    ParMatrix* mult_T(ParCSRMatrix* B, bool tap = false, data_t* comm_t = NULL);
    ParMatrix* tap_mult_T(ParCSCMatrix* B, data_t* comm_t = NULL);
    ParMatrix* tap_mult_T(ParCSRMatrix* B, data_t* comm_t = NULL);
    ParMatrix* add(ParCSRMatrix* A);
    ParMatrix* subtract(ParCSRMatrix* A);

    void init_tap_communicators(MPI_Comm comm = MPI_COMM_WORLD, data_t* comm_t = NULL);
    void update_tap_comm(ParMatrix* old, const aligned_vector<int>& old_to_new,
            double* comm_t = NULL)
    {
        tap_comm = new TAPComm((TAPComm*) old->tap_comm, old_to_new, NULL, comm_t);
        tap_mat_comm = new TAPComm((TAPComm*) old->tap_mat_comm, old_to_new, 
                tap_comm->local_L_par_comm, comm_t);
    }
    void update_tap_comm(ParMatrix* old, const aligned_vector<int>& on_old_to_new,
            const aligned_vector<int>& off_old_to_new, double* comm_t = NULL)
    {
        tap_comm = new TAPComm((TAPComm*) old->tap_comm, on_old_to_new, off_old_to_new, 
                NULL, comm_t);
        tap_mat_comm = new TAPComm((TAPComm*) old->tap_mat_comm, on_old_to_new, 
                off_old_to_new, tap_comm->local_L_par_comm, comm_t);
    }



    void sort()
    {
        on_proc->sort();
        off_proc->sort();
    }

    virtual ParMatrix* transpose() = 0;

    aligned_vector<int>& get_off_proc_column_map()
    {
        return off_proc_column_map;
    }

    aligned_vector<int>& get_on_proc_column_map()
    {
        return on_proc_column_map;
    }

    aligned_vector<int>& get_local_row_map()
    {
        return local_row_map;
    }

    virtual ParCOOMatrix* to_ParCOO() = 0;
    virtual ParCSRMatrix* to_ParCSR() = 0;
    virtual ParCSCMatrix* to_ParCSC() = 0;
    virtual ParMatrix* copy() = 0;
    virtual void copy_helper(ParCSRMatrix* A);
    virtual void copy_helper(ParCSCMatrix* A);
    virtual void copy_helper(ParCOOMatrix* A);
    void default_copy_helper(ParMatrix* A);

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
    aligned_vector<int> off_proc_column_map; // Maps off_proc local to global
    aligned_vector<int> on_proc_column_map; // Maps on_proc local to global
    aligned_vector<int> local_row_map; // Maps local rows to global

    // Parallel communication package indicating which 
    // processes hold vector values associated with off_proc,
    // and which processes need vector values from this proc
    Partition* partition;
    ParComm* comm;
    TAPComm* tap_comm;
    TAPComm* tap_mat_comm;
    bool shared_comm;
  };

  class ParCOOMatrix : public ParMatrix
  {
  public:
    ParCOOMatrix(bool form_mat = true) : ParMatrix()
    {
        if (form_mat)
        {
            on_proc = new COOMatrix(0, 0, 0);
            off_proc = new COOMatrix(0, 0, 0);
        }
    }

    ParCOOMatrix(index_t glob_rows, 
            index_t glob_cols,
            int nnz_per_row = 5, bool form_mat = true) 
        : ParMatrix(glob_rows, glob_cols)
    {
        if (form_mat)
        {
            on_proc = new COOMatrix(partition->local_num_rows, partition->local_num_cols, 
                    nnz_per_row);
            off_proc = new COOMatrix(partition->local_num_rows, partition->global_num_cols, 
                    nnz_per_row);
        }
    }

    ParCOOMatrix(index_t glob_rows, index_t glob_cols, int local_rows, 
            int local_cols, index_t first_row, index_t first_col,
            int nnz_per_row = 5, bool form_mat = true) 
        : ParMatrix(glob_rows, glob_cols,
                local_rows, local_cols, first_row, first_col)
    {
        if (form_mat)
        {
            on_proc = new COOMatrix(partition->local_num_rows, partition->local_num_cols, 
                    nnz_per_row);
            off_proc = new COOMatrix(partition->local_num_rows, partition->global_num_cols, 
                    nnz_per_row);
        }
    }
    
    ParCOOMatrix(Partition* part, 
            int nnz_per_row = 5, bool form_mat = true) : ParMatrix(part)
    {
        if (form_mat)
        {
            on_proc = new COOMatrix(partition->local_num_rows, partition->local_num_cols, 
                    nnz_per_row);
            off_proc = new COOMatrix(partition->local_num_rows, partition->global_num_cols, 
                    nnz_per_row);
        }
    }

    ParCOOMatrix* to_ParCOO();
    ParCSRMatrix* to_ParCSR();
    ParCSCMatrix* to_ParCSC();
    ParCOOMatrix* copy()
    {
        ParCOOMatrix* A = new ParCOOMatrix();
        A->copy_helper(this);
        return A;
    }
    void copy_helper(ParCSRMatrix* A);
    void copy_helper(ParCSCMatrix* A);
    void copy_helper(ParCOOMatrix* A);

    void mult(ParVector& x, ParVector& b, bool tap = false, data_t* comm_t = NULL);
    void mult_timed(ParVector& x, ParVector& b, aligned_vector<double>& times, bool tap = false, data_t* comm_t = NULL);
    void tap_mult(ParVector& x, ParVector& b, data_t* comm_t = NULL);
    void mult_T(ParVector& x, ParVector& b, bool tap = false, data_t* comm_t = NULL);
    void tap_mult_T(ParVector& x, ParVector& b, data_t* comm_t = NULL);

    ParCOOMatrix* transpose();
  };


  class ParBCOOMatrix : public ParCOOMatrix
  {
  public:
    ParBCOOMatrix() : ParCOOMatrix(false)
    {
        on_proc = new BCOOMatrix(0, 0, 1, 1, 0);
        off_proc = new BCOOMatrix(0, 0, 1, 1, 0);
    }

    ParBCOOMatrix(int global_block_rows, int global_block_cols,
            int block_row_size, int block_col_size, int nnz_per_row)
        : ParCOOMatrix(global_block_rows, global_block_cols, nnz_per_row, false)
    {
        on_proc = new BCOOMatrix(partition->local_num_rows, partition->local_num_cols, 
                block_row_size, block_col_size, nnz_per_row);
        off_proc = new BCOOMatrix(partition->local_num_rows, partition->global_num_cols, 
                block_row_size, block_col_size, nnz_per_row);
    }

    ParBCOOMatrix(int global_block_rows, int global_block_cols,
            int local_block_rows, int local_block_cols, 
            int first_block_row, int first_block_col,
            int block_row_size, int block_col_size, int nnz_per_row = 5) 
        : ParCOOMatrix(global_block_rows, global_block_cols,
                local_block_rows, local_block_cols, first_block_row, 
                first_block_col, nnz_per_row, false)
    {
        on_proc = new BCOOMatrix(partition->local_num_rows, partition->local_num_cols, 
                block_row_size, block_col_size, nnz_per_row);
        off_proc = new BCOOMatrix(partition->local_num_rows, partition->global_num_cols, 
                block_row_size, block_col_size, nnz_per_row);
    }

    ParBCOOMatrix(Partition* part, int block_row_size, int block_col_size,
            int nnz_per_row = 5) : ParCOOMatrix(part, nnz_per_row, false)
    {
        on_proc = new BCOOMatrix(partition->local_num_rows, partition->local_num_cols, 
                block_row_size, block_col_size, nnz_per_row);
        off_proc = new BCOOMatrix(partition->local_num_rows, partition->global_num_cols, 
                block_row_size, block_col_size, nnz_per_row);
    }

    ParCOOMatrix* to_ParCOO();
    ParCSRMatrix* to_ParCSR();
    ParCSCMatrix* to_ParCSC();
    ParCOOMatrix* copy()
    {
        ParCOOMatrix* A = new ParCOOMatrix();
        A->copy_helper(this);
        return A;
    }
  };

  class ParCSRMatrix : public ParMatrix
  {
  public:
    ParCSRMatrix(bool form_mat = true) : ParMatrix()
    {
        if (form_mat)
        {
            on_proc = new CSRMatrix(0, 0, 0);
            off_proc = new CSRMatrix(0, 0, 0);
        }
    }

    ParCSRMatrix(index_t glob_rows, index_t glob_cols, int nnz = 0, 
            bool form_mat = true) : ParMatrix(glob_rows, glob_cols)
    {
        if (form_mat)
        {
            on_proc = new CSRMatrix(partition->local_num_rows, partition->local_num_cols, 
                    nnz);
            off_proc = new CSRMatrix(partition->local_num_rows, partition->global_num_cols, 
                    nnz);
        }
    }

    ParCSRMatrix(index_t glob_rows, index_t glob_cols, int local_rows, 
            int local_cols, index_t first_row, index_t first_col, Topology* topology = NULL,  
            int nnz = 0, bool form_mat = true) : ParMatrix(glob_rows, glob_cols,
                local_rows, local_cols, first_row, first_col, topology)
    {
        if (form_mat)
        {
            on_proc = new CSRMatrix(partition->local_num_rows, partition->local_num_cols, 
                    nnz);
            off_proc = new CSRMatrix(partition->local_num_rows, partition->global_num_cols, 
                    nnz);
        }
    }

    ParCSRMatrix(Partition* part, 
            int nnz = 0, bool form_mat = true) : ParMatrix(part)
    {
        if (form_mat)
        {
            on_proc = new CSRMatrix(partition->local_num_rows, partition->local_num_cols, 
                    nnz);
            off_proc = new CSRMatrix(partition->local_num_rows, partition->global_num_cols, 
                    nnz);
        }
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


    ParCSRMatrix(Partition* part, index_t glob_rows, index_t glob_cols, 
            int local_rows, int on_proc_cols, int off_proc_cols, int nnz = 0,
            bool form_mat = true) : ParMatrix(part, glob_rows, glob_cols, 
                local_rows, on_proc_cols)
    {
        off_proc_num_cols = off_proc_cols;
        if (form_mat)
        {
            on_proc = new CSRMatrix(local_num_rows, on_proc_cols, nnz);
            off_proc = new CSRMatrix(local_num_rows, off_proc_num_cols, nnz);
        }
    }

    ParCOOMatrix* to_ParCOO();
    ParCSRMatrix* to_ParCSR();
    ParCSCMatrix* to_ParCSC();
    ParCSRMatrix* copy()
    {
        ParCSRMatrix* A = new ParCSRMatrix();
        A->copy_helper(this);
        return A;
    }

    void copy_structure(ParBSRMatrix* A);

    ParBSRMatrix* to_ParBSR(const int block_row_size, const int block_col_size);

    void copy_helper(ParCSRMatrix* A);
    void copy_helper(ParCSCMatrix* A);
    void copy_helper(ParCOOMatrix* A);

    ParCSRMatrix* strength(strength_t strength_type, double theta = 0.0, 
            bool tap_amg = false, int num_variables = 1, int* variables = NULL,
            data_t* comm_t = NULL);
    ParCSRMatrix* aggregate();
    ParCSRMatrix* fit_candidates(double* B, double* R, int num_candidates, 
            double tol = 1e-10);
    int maximal_independent_set(aligned_vector<int>& local_states,
            aligned_vector<int>& off_proc_states, int max_iters = -1);

    void mult(ParVector& x, ParVector& b, bool tap = false, data_t* comm_t = NULL);
    void mult_timed(ParVector& x, ParVector& b, aligned_vector<double>& times, bool tap = false, data_t* comm_t = NULL);
    void tap_mult(ParVector& x, ParVector& b, data_t* comm_t = NULL);
    void mult_T(ParVector& x, ParVector& b, bool tap = false, data_t* comm_t = NULL);
    void tap_mult_T(ParVector& x, ParVector& b, data_t* comm_t = NULL);
    ParCSRMatrix* mult(ParCSRMatrix* B, bool tap = false, data_t* comm_t = NULL);
    ParCSRMatrix* tap_mult(ParCSRMatrix* B, data_t* comm_t = NULL);
    ParCSRMatrix* mult_T(ParCSCMatrix* A, bool tap = false, data_t* comm_t = NULL);
    ParCSRMatrix* mult_T(ParCSRMatrix* A, bool tap = false, data_t* comm_t = NULL);
    ParCSRMatrix* tap_mult_T(ParCSCMatrix* A, data_t* comm_t = NULL);
    ParCSRMatrix* tap_mult_T(ParCSRMatrix* A, data_t* comm_t = NULL);
    ParCSRMatrix* add(ParCSRMatrix* A);
    ParCSRMatrix* subtract(ParCSRMatrix* B);

    void print_mult(ParCSRMatrix* B, const aligned_vector<int>& proc_distances, 
                const aligned_vector<int>& worst_proc_distances);
    void print_mult_T(ParCSCMatrix* A, const aligned_vector<int>& proc_distances,
                const aligned_vector<int>& worst_proc_distances);
    void print_mult(const aligned_vector<int>& proc_distances,
                const aligned_vector<int>& worst_proc_distances);
    void print_mult_T(const aligned_vector<int>& proc_distances,
                const aligned_vector<int>& worst_proc_distances);
    
    void mult_helper(ParCSRMatrix* B, ParCSRMatrix* C, CSRMatrix* recv,
            CSRMatrix* C_on_on, CSRMatrix* C_on_off);
    CSRMatrix* mult_T_partial(ParCSCMatrix* A);
    CSRMatrix* mult_T_partial(CSCMatrix* A_off);
    void mult_T_combine(ParCSCMatrix* A, ParCSRMatrix* C, CSRMatrix* recv_mat,
            CSRMatrix* C_on_on, CSRMatrix* C_off_on);
    
    ParCSRMatrix* transpose();
  };

 class ParBSRMatrix : public ParCSRMatrix
  {
  public:
    ParBSRMatrix() : ParCSRMatrix(false)
    {
        on_proc = new BSRMatrix(0, 0, 1, 1, 0);
        off_proc = new BSRMatrix(0, 0, 1, 1, 0);
    }

    ParBSRMatrix(int global_block_rows, int global_block_cols,
            int block_row_size, int block_col_size,
            int nnz = 0) 
        :  ParCSRMatrix(global_block_rows, global_block_cols, nnz, false)
    {
        on_proc = new BSRMatrix(partition->local_num_rows, partition->local_num_cols, 
                block_row_size, block_col_size, nnz);
        off_proc = new BSRMatrix(partition->local_num_rows, partition->global_num_cols, 
                block_row_size, block_col_size, nnz);
    }

    ParBSRMatrix(int global_block_rows, int global_block_cols, 
            int local_block_rows, int local_block_cols, 
            int first_block_row, int first_block_col,
            int block_row_size, int block_col_size,
            Topology* topology = NULL, int nnz = 0)
        : ParCSRMatrix(global_block_rows, global_block_cols,
                local_block_rows, local_block_cols, 
                first_block_row, first_block_col, topology,
                nnz, false)
    {
        on_proc = new BSRMatrix(partition->local_num_rows, partition->local_num_cols, 
                block_row_size, block_col_size, nnz);
        off_proc = new BSRMatrix(partition->local_num_rows, partition->global_num_cols, 
                block_row_size, block_col_size, nnz);
    }

    ParBSRMatrix(Partition* part, int block_row_size, int block_col_size,
            int nnz = 0) : ParCSRMatrix(part, nnz, false)
    {
        on_proc = new BSRMatrix(partition->local_num_rows, partition->local_num_cols, 
                block_row_size, block_col_size, nnz);
        off_proc = new BSRMatrix(partition->local_num_rows, partition->global_num_cols, 
                block_row_size, block_col_size, nnz);
    }

    ParBSRMatrix(Partition* part, BSRMatrix* _on_proc, BSRMatrix* _off_proc) 
        : ParCSRMatrix(part)
    {
        on_proc = _on_proc;
        off_proc = _off_proc;
        on_proc_num_cols = on_proc->n_cols;
        off_proc_num_cols = off_proc->n_cols;
        local_num_rows = on_proc->n_rows;
        finalize();
    }

    ParBSRMatrix(Partition* part, int global_block_rows, int global_block_cols,
            int local_block_rows, int on_proc_block_cols, int off_proc_block_cols, 
            int block_row_size, int block_col_size, int nnz = 0)
          : ParCSRMatrix(part, global_block_rows, global_block_cols,
                  local_block_rows, on_proc_block_cols, off_proc_block_cols, 
                  nnz, false)
    {
        off_proc_num_cols = off_proc_block_cols;
        on_proc = new BSRMatrix(local_block_rows, on_proc_block_cols, 
                block_row_size, block_col_size, nnz);
        off_proc = new BSRMatrix(local_block_rows, off_proc_num_cols, 
                block_row_size, block_col_size, nnz);
    }

    ParCOOMatrix* to_ParCOO();
    ParCSRMatrix* to_ParCSR();
    ParCSCMatrix* to_ParCSC();
    ParBSRMatrix* copy()
    {
        ParBSRMatrix* A = new ParBSRMatrix();
        A->copy_helper(this);
        return A;
    }

  };



  class ParCSCMatrix : public ParMatrix
  {
  public:
    ParCSCMatrix(bool form_mat = true) : ParMatrix()
    {
        if (form_mat)
        {
            on_proc = new CSCMatrix(0, 0, 0);
            off_proc = new CSCMatrix(0, 0, 0);
        }
    }

    ParCSCMatrix(index_t glob_rows, index_t glob_cols, int nnz_per_row = 5, 
            bool form_mat = true) : ParMatrix(glob_rows, glob_cols)
    {
        if (form_mat)
        {
            on_proc = new CSCMatrix(partition->local_num_rows, partition->local_num_cols, 
                    nnz_per_row);
            off_proc = new CSCMatrix(partition->local_num_rows, partition->global_num_cols, 
                    nnz_per_row);
        }
    }

    ParCSCMatrix(index_t glob_rows, index_t glob_cols, int local_num_rows, 
            int local_num_cols, index_t first_local_row, index_t first_col, 
            int nnz_per_row = 5, bool form_mat = true) 
        : ParMatrix(glob_rows, glob_cols, local_num_rows, local_num_cols, 
                first_local_row, first_col)
    {
        if (form_mat)
        {
            on_proc = new CSCMatrix(partition->local_num_rows, partition->local_num_cols, 
                    nnz_per_row);
            off_proc = new CSCMatrix(partition->local_num_rows, partition->global_num_cols, 
                    nnz_per_row);
        }
    }

    ParCSCMatrix(Partition* part, index_t glob_rows, index_t glob_cols, int local_rows,
            int on_proc_cols, int off_proc_cols, int nnz_per_row = 5, bool form_mat = true) 
        : ParMatrix(part, glob_rows, glob_cols, local_rows, on_proc_cols)
    {
        off_proc_num_cols = off_proc_cols;
        if (form_mat)
        {
            on_proc = new CSCMatrix(local_num_rows, on_proc_cols, nnz_per_row);
            off_proc = new CSCMatrix(local_num_rows, off_proc_num_cols, nnz_per_row);
        }
    }


    ParCSCMatrix(Partition* part, int nnz_per_row = 5, bool form_mat = true) 
        : ParMatrix(part)
    {
        if (form_mat)
        {
            on_proc = new CSRMatrix(partition->local_num_rows, partition->local_num_cols, 
                    nnz_per_row);
            off_proc = new CSRMatrix(partition->local_num_rows, partition->global_num_cols, 
                    nnz_per_row);
        }
    }

    ParCOOMatrix* to_ParCOO();
    ParCSRMatrix* to_ParCSR();
    ParCSCMatrix* to_ParCSC();
    ParCSCMatrix* copy()
    {
        ParCSCMatrix* A = new ParCSCMatrix();
        A->copy_helper(this);
        return A;
    }

    void copy_helper(ParCSRMatrix* A);
    void copy_helper(ParCSCMatrix* A);
    void copy_helper(ParCOOMatrix* A);

    void mult(ParVector& x, ParVector& b, bool tap, data_t* comm_t);
    void mult_timed(ParVector& x, ParVector& b, aligned_vector<double>& times, bool tap, data_t* comm_t);
    void tap_mult(ParVector& x, ParVector& b, data_t* comm_t);
    void mult_T(ParVector& x, ParVector& b, bool tap, data_t* comm_t);
    void tap_mult_T(ParVector& x, ParVector& b, data_t* comm_t);

    ParCSCMatrix* transpose();
  };


class ParBSCMatrix : public ParCSCMatrix
  {
  public:
    ParBSCMatrix() : ParCSCMatrix(false)
    {
        on_proc = new BSCMatrix(0, 0, 1, 1, 0);
        off_proc = new BSCMatrix(0, 0, 1, 1, 0);
    }

    ParBSCMatrix(int global_block_rows, int global_block_cols,
            int block_row_size, int block_col_size,
            int nnz = 0) 
        : ParCSCMatrix(global_block_rows, global_block_cols, nnz, false)
    {
        on_proc = new BSCMatrix(partition->local_num_rows, partition->local_num_cols, 
                block_row_size, block_col_size, nnz);
        off_proc = new BSCMatrix(partition->local_num_rows, partition->global_num_cols, 
                block_row_size, block_col_size, nnz);
    }

    ParBSCMatrix(Partition* part, int block_row_size, int block_col_size,
            int nnz = 0) : ParCSCMatrix(part, nnz, false)
    {
        on_proc = new BSCMatrix(partition->local_num_rows, partition->local_num_cols, 
                block_row_size, block_col_size, nnz);
        off_proc = new BSCMatrix(partition->local_num_rows, partition->global_num_cols, 
                block_row_size, block_col_size, nnz);
    }

    ParBSCMatrix(Partition* part, int global_block_rows, int global_block_cols,
            int local_block_rows, int on_proc_block_cols, int off_proc_block_cols, 
            int block_row_size, int block_col_size, int nnz = 0)
          : ParCSCMatrix(part, global_block_rows, global_block_cols, local_block_rows, 
                  on_proc_block_cols, off_proc_block_cols, nnz, false)
    {
        off_proc_num_cols = off_proc_block_cols;
        on_proc = new BSCMatrix(local_num_rows, on_proc_block_cols, 
                block_row_size, block_col_size, nnz);
        off_proc = new BSCMatrix(local_num_rows, off_proc_num_cols, 
                block_row_size, block_col_size, nnz);
    }

    ParCOOMatrix* to_ParCOO();
    ParCSRMatrix* to_ParCSR();
    ParCSCMatrix* to_ParCSC();
    ParBSCMatrix* copy()
    {
        ParBSCMatrix* A = new ParBSCMatrix();
        A->copy_helper(this);
        return A;
    }

  };


}
#endif
