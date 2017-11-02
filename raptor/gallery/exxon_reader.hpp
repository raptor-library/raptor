// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef EXXON_READER_HPP
#define EXXON_READER_HPP

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <sstream>

#include "core/par_matrix.hpp"
#include "core/types.hpp"
#include "util/linalg/repartition.hpp"

using namespace raptor;

ParCSRMatrix* exxon_reader(char* folder, char* iname, char* fname, char* suffix, std::vector<int>& on_proc_column_map)

{
    // Get MPI Info
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Declare Matrix Variables
    ParCOOMatrix* A_coo = NULL;
    int on_proc_num_cols;
    int off_proc_num_cols;
    int block_on_proc_num_cols;
    int block_off_proc_num_cols;

    // Delcare File Reader Info
    int isize = 4;
    int dsize = 8;

    // Declare info to recv from Index file
    int n;
    int bs;

    // Declare header info for matrix file
    int header_size = 5;
    int* header;
    int first_block_row;
    int last_block_row;
    int first_block_col;
    int last_block_col;
    int block_size;

    // Declare info for reading row/column of matrix entry
    int pos[2];
    int pos_size = 2;
    int first_local_row, first_col;
    int local_row, local_col;
    int global_block_rows;
    int n_data;
    double value;
    double* data;
    Matrix* tmp_mat;

    int proc_start, proc_end;
    int ctr;
    int local, global;

    // Declare strings for names of index/matrix files
    std::ostringstream oss;
    std::string iname_string;
    std::string fname_string;
    char iname_r[1024];
    char fname_r[1024];
    FILE* infile;
    unsigned char bytes[4];

    // Find names of index and matrix files corresponding to my rank
    oss << folder << "/" << iname << rank;
    iname_string = oss.str();
    oss.str("");
    oss << folder << "/" << fname << rank << suffix;
    fname_string = oss.str();
    strncpy(iname_r, iname_string.c_str(), sizeof(iname_r));
    iname_r[sizeof(iname_r)-1] = 0;
    strncpy(fname_r, fname_string.c_str(), sizeof(fname_r));
    fname_r[sizeof(fname_r)-1] = 0;

    // Open index file, and read local number of rows (n),
    // block size (bs) and global row indices (index)
    infile = fopen(iname_r, "rb");
    fread(&n, isize, 1, infile);
    fread(&bs, isize, 1, infile);
    std::vector<int> global_indices;
    int idx;
    while (fread(&idx, isize, 1, infile) == 1) 
    {
        global_indices.push_back(idx);
    }

    // Close index file
    fclose(infile);

    A_coo = new ParCOOMatrix();

    // Open matrix file, and read the first/last block row/col 
    // as well as the block size.  
    infile = fopen(fname_r, "rb");
    header = new int[header_size];
    fread(header, isize, header_size, infile);
    first_block_row = header[0];
    last_block_row = header[1];
    first_block_col = header[2];
    last_block_col = header[3];
    block_size = header[4];
    delete[] header;

    // Determine number of columns (and block cols) in on and off
    // process blocks of matrix to be read in
    block_on_proc_num_cols = (last_block_row - first_block_row + 1);
    on_proc_num_cols = block_on_proc_num_cols * block_size;
    block_off_proc_num_cols = (last_block_col - first_block_col + 1) 
        - block_on_proc_num_cols;
    off_proc_num_cols = block_off_proc_num_cols * block_size;

    // Determine the number of values per block and initialize 
    // variable in which to read values
    n_data = block_size * block_size;
    data = new double[n_data];

    // Resize on_proc and off_proc matrices
    A_coo->on_proc->resize(on_proc_num_cols, on_proc_num_cols);
    if (block_off_proc_num_cols)
        A_coo->off_proc->resize(on_proc_num_cols, off_proc_num_cols);

    // Read in nonzeros of matrix, adding to appropriate block
    // (on_proc or off_proc)
    while (fread(pos, isize, 2, infile) == pos_size) 
    {
        local_row = pos[0];
        local_col = pos[1];
        fread(data, dsize, n_data, infile);
        first_local_row = local_row * block_size;

        if (local_col >= first_block_row && local_col <= last_block_row)
        {
            first_col = local_col * block_size;            
            for (int i = 0; i < block_size; i++)
            {
                for (int j = 0; j < block_size; j++)
                {
                    value = data[i*block_size + j];
                    if (fabs(value) > 1e-15) // Only add if large enough
                    {
                        A_coo->on_proc->add_value(first_local_row + i, 
                                first_col + j, value);
                    }
                }
            }
        }
        else
        {   
            first_col = (local_col - block_on_proc_num_cols) * block_size;
            for (int i = 0; i < block_size; i++)
            {
                for (int j = 0; j < block_size; j++)
                {
                    value = data[i*block_size + j];
                    if (fabs(value) > 1e-15) // Only add if large enough
                    {
                        A_coo->off_proc->add_value(first_local_row + i, 
                                first_col + j, value);
                    }
                }
            }
        }
    }

    // Delete variable into which data was read,
    // and close matrix file
    delete[] data;
    fclose(infile);

    // Set dimensions of matrix    
    // All reduce global number of rows in matrix
    MPI_Allreduce(&on_proc_num_cols, &(A_coo->global_num_rows), 1, MPI_INT,
            MPI_SUM, MPI_COMM_WORLD);
    A_coo->local_num_rows = on_proc_num_cols;
    A_coo->on_proc_num_cols = A_coo->local_num_rows;
    A_coo->global_num_cols = A_coo->global_num_rows;
    A_coo->local_nnz = A_coo->on_proc->nnz + A_coo->off_proc->nnz;


    int first_row;
    std::vector<int> proc_sizes(num_procs);
    MPI_Allgather(&A_coo->local_num_rows, 1, MPI_INT, proc_sizes.data(),
            1, MPI_INT, MPI_COMM_WORLD);
    first_row = 0;
    for (int i = 0; i < rank; i++)
    {
        first_row += proc_sizes[i];
    }

    A_coo->partition = new Partition(A_coo->global_num_rows, A_coo->global_num_rows,
            A_coo->local_num_rows, A_coo->local_num_rows, 
            first_row, first_row);

    if (A_coo->on_proc->n_cols)
    {
        A_coo->on_proc_column_map.resize(A_coo->on_proc->n_cols);
    }
    A_coo->off_proc_num_cols = A_coo->off_proc->n_cols;
    if (A_coo->off_proc_num_cols)
    {
        A_coo->off_proc_column_map.resize(A_coo->off_proc_num_cols);
    }

    // Create column maps from all local to global columns
    int first_local_col;
    int first_global_col;

    std::map<int, int> on_proc_global_to_local;
    std::map<int, int> off_proc_global_to_local;
    for (int i = first_block_row; i <= last_block_row; i++)
    {
        first_local_col = i*block_size;
        first_global_col = global_indices[i] * block_size;
        for (int j = 0; j < block_size; j++)
        {
            A_coo->on_proc_column_map[first_local_col + j] = first_global_col + j;
        }
    }
    for (int i = block_on_proc_num_cols; i <= last_block_col; i++)
    {
        first_local_col = (i - block_on_proc_num_cols) * block_size;
        first_global_col = global_indices[i] * block_size;
        for (int j = 0; j < block_size; j++)
        {
            A_coo->off_proc_column_map[first_local_col + j] = first_global_col + j;
        }
    }
    A_coo->local_row_map = A_coo->get_on_proc_column_map();

    ParCSRMatrix* A = new ParCSRMatrix(A_coo);
    make_contiguous(A);

    delete A_coo;
    return A;
}

ParCSRMatrix* exxon_pressure_reader(char* folder, char* iname, char* fname, char* suffix, std::vector<int>& on_proc_column_map)
{
    // Get MPI Info
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Declare Matrix Variables
    ParCOOMatrix* A_coo = NULL;
    int block_on_proc_num_cols;
    int block_off_proc_num_cols;

    // Delcare File Reader Info
    int isize = 4;
    int dsize = 8;

    // Declare info to recv from Index file
    int n;
    int bs;

    // Declare header info for matrix file
    int header_size = 5;
    int* header;
    int first_block_row;
    int last_block_row;
    int first_block_col;
    int last_block_col;
    int block_size;
    int pressure_pos;

    // Declare info for reading row/column of matrix entry
    int pos[2];
    int pos_size = 2;
    int first_local_row, first_col;
    int local_row, local_col;
    int global_block_rows;
    int n_data;
    double value;
    double* data;
    Matrix* tmp_mat;

    int proc_start, proc_end;
    int ctr;
    int local, global;

        // Declare strings for names of index/matrix files
    std::ostringstream oss;
    std::string iname_string;
    std::string fname_string;
    char iname_r[1024];
    char fname_r[1024];
    FILE* infile;
    unsigned char bytes[4];

    // Find names of index and matrix files corresponding to my rank
    oss << folder << "/" << iname << rank;
    iname_string = oss.str();
    oss.str("");
    oss << folder << "/" << fname << rank << suffix;
    fname_string = oss.str();
    strncpy(iname_r, iname_string.c_str(), sizeof(iname_r));
    iname_r[sizeof(iname_r)-1] = 0;
    strncpy(fname_r, fname_string.c_str(), sizeof(fname_r));
    fname_r[sizeof(fname_r)-1] = 0;

    // Open index file, and read local number of rows (n),
    // block size (bs) and global row indices (index)
    infile = fopen(iname_r, "rb");
    fread(&n, isize, 1, infile);
    fread(&bs, isize, 1, infile);
    std::vector<int> global_indices;
    int idx;
    while (fread(&idx, isize, 1, infile) == 1) 
    {
        global_indices.push_back(idx);
    }

    // Close index file
    fclose(infile);

    // Create a new, empty Parallel Matrix Object
    A_coo = new ParCOOMatrix();

    // Open matrix file, and read the first/last block row/col 
    // as well as the block size.  
    infile = fopen(fname_r, "rb");
    header = new int[header_size];
    fread(header, isize, header_size, infile);
    first_block_row = header[0];
    last_block_row = header[1];
    first_block_col = header[2];
    last_block_col = header[3];
    block_size = header[4];
    pressure_pos = (block_size * block_size) - 1;
    delete[] header;

        // Determine number of columns (and block cols) in on and off
    // process blocks of matrix to be read in
    block_on_proc_num_cols = (last_block_row - first_block_row + 1);
    block_off_proc_num_cols = (last_block_col - first_block_col + 1) 
        - block_on_proc_num_cols;

    // Determine the number of values per block and initialize 
    // variable in which to read values
    n_data = block_size * block_size;
    data = new double[n_data];

    // Resize on_proc and off_proc matrices
    A_coo->on_proc->resize(block_on_proc_num_cols, block_on_proc_num_cols);
    if (block_off_proc_num_cols) 
        A_coo->off_proc->resize(block_on_proc_num_cols, block_off_proc_num_cols);

    // Read in nonzeros of matrix, adding to appropriate block
    // (on_proc or off_proc)
    while (fread(pos, isize, 2, infile) == pos_size) 
    {
        local_row = pos[0];
        local_col = pos[1];
        fread(data, dsize, n_data, infile);
        if (local_col >= first_block_row && local_col <= last_block_row)
        {
            if (fabs(data[pressure_pos]) > 1e-15)
                A_coo->on_proc->add_value(local_row, local_col, data[pressure_pos]);
        }
        else
        {   
            if (fabs(data[pressure_pos]) > 1e-15)
                A_coo->off_proc->add_value(local_row, 
                        local_col - block_on_proc_num_cols, data[pressure_pos]);
        }
    }

    // Delete variable into which data was read,
    // and close matrix file
    delete[] data;
    fclose(infile);

    // Set dimensions of matrix    
    // All reduce global number of rows in matrix
    MPI_Allreduce(&block_on_proc_num_cols, &(A_coo->global_num_rows), 
            1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    A_coo->local_num_rows = block_on_proc_num_cols;
    A_coo->on_proc_num_cols = A_coo->local_num_rows;
    A_coo->global_num_cols = A_coo->global_num_rows;
    A_coo->local_nnz = A_coo->on_proc->nnz + A_coo->off_proc->nnz;

    // Create column maps from all local to global columns
    int first_local_col;
    int first_global_col;
    for (int i = first_block_row; i <= last_block_row; i++)
    {
        on_proc_column_map.push_back(global_indices[i]);
    }
    for (int i = block_on_proc_num_cols; i <= last_block_col; i++)
    {
        A_coo->off_proc_column_map.push_back(global_indices[i]);
        A_coo->off_proc_num_cols = A_coo->off_proc_column_map.size();
    }

    ParCSRMatrix* A = new ParCSRMatrix(A_coo);
    make_contiguous(A);

    delete A_coo;
    return A;
}


void exxon_vector_reader(char* folder, char* fname, char* suffix, 
        int first_local, ParVector& x)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int first_block_row, last_block_row, block_size;
    int block_row;
    int first_idx;
    std::vector<double> block_values;

    // Delcare File Reader Info
    int isize = 4;
    int dsize = 8;

    // Declare strings for names of index/matrix files
    std::ostringstream oss;
    std::string fname_string;
    char fname_r[1024];
    FILE* infile;
    unsigned char bytes[4];

    // Find names of index and matrix files corresponding to my rank
    oss << folder << "/" << fname << rank << suffix;
    fname_string = oss.str();
    strncpy(fname_r, fname_string.c_str(), sizeof(fname_r));
    fname_r[sizeof(fname_r)-1] = 0;

    // Open index file, and read local number of rows (n),
    // block size (bs) and global row indices (index)
    infile = fopen(fname_r, "rb");
    fread(&first_block_row, isize, 1, infile);
    fread(&last_block_row, isize, 1, infile);
    fread(&block_size, isize, 1, infile);

    int local_size = (last_block_row - first_block_row + 1) * block_size;
    x.local_n = local_size;
    x.local.num_values = local_size;
    x.first_local = first_local;
    if (local_size)
    {
        x.local.values.resize(local_size);
    }

    block_values.resize(block_size);

    while (fread(&block_row, isize, 1, infile) == 1)
    {
        first_idx = block_row * block_size;
        fread(block_values.data(), dsize, block_size, infile);
        for (int i = 0; i < block_size; i++)
        {
            x.local.values[first_idx + i] = block_values[i];
        }
    }

    // Close index file
    fclose(infile);
}

void exxon_pressure_vector_reader(char* folder, char* fname, char* suffix, 
        int first_local,  ParVector& x)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int first_block_row, last_block_row, block_size;
    int block_row;
    int first_idx;
    std::vector<double> block_values;

    // Delcare File Reader Info
    int isize = 4;
    int dsize = 8;

    // Declare strings for names of index/matrix files
    std::ostringstream oss;
    std::string fname_string;
    char fname_r[1024];
    FILE* infile;
    unsigned char bytes[4];

    // Find names of index and matrix files corresponding to my rank
    oss << folder << "/" << fname << rank << suffix;
    fname_string = oss.str();
    strncpy(fname_r, fname_string.c_str(), sizeof(fname_r));
    fname_r[sizeof(fname_r)-1] = 0;

    // Open index file, and read local number of rows (n),
    // block size (bs) and global row indices (index)
    infile = fopen(fname_r, "rb");
    fread(&first_block_row, isize, 1, infile);
    fread(&last_block_row, isize, 1, infile);
    fread(&block_size, isize, 1, infile);

    int local_size = last_block_row - first_block_row + 1;
    x.local_n = local_size;
    x.local.num_values = local_size;
    x.first_local = first_local;
    if (local_size)
    {
        x.local.values.resize(local_size);
    }

    block_values.resize(block_size);

    while (fread(&block_row, isize, 1, infile) == 1)
    {
        first_idx = block_row * block_size;
        fread(block_values.data(), dsize, block_size, infile);
        x.local.values[block_row] = block_values[block_size-1];
    }

    // Close index file
    fclose(infile);
}

#endif
