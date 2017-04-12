// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "exxon_reader.hpp"
#include <string>
#include <set>
#include <sstream>
#include <stdio.h>

ParCSRMatrix* exxon_reader(char* folder, char* iname, char* fname, char* suffix, int** global_num_rows)
{
    // Get MPI Info
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Declare Matrix Variables
    ParCOOMatrix* A_coo = NULL;
    int diag_num_cols;
    int offd_num_cols;
    int block_diag_num_cols;
    int block_offd_num_cols;

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
    int* sizes;
    int* displs;
    int* orig_block_rows;
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
    delete[] header;

    block_diag_num_cols = (last_block_row - first_block_row + 1);
    diag_num_cols = block_diag_num_cols * block_size;
    block_offd_num_cols = (last_block_col - first_block_col + 1) - block_diag_num_cols;
    offd_num_cols = block_offd_num_cols * block_size;
    n_data = block_size * block_size;
    data = new double[n_data];
    std::set<int> offd_col_set;

    A_coo->on_proc->resize(diag_num_cols, diag_num_cols);
    if (block_offd_num_cols)
        A_coo->off_proc->resize(diag_num_cols, block_offd_num_cols*block_size);


int count = 0;
int g_count;
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
                    if (fabs(value) > zero_tol)
                    {
                        A_coo->on_proc->add_value(first_local_row + i, first_col + j, value);
                    }
count++;
                }
            }
        }
        else
        {   
            first_col = global_indices[local_col]*block_size;
            for (int i = 0; i < block_size; i++)
            {
                for (int j = 0; j < block_size; j++)
                {
                    value = data[i*block_size + j];
                    if (fabs(value) > zero_tol)
                    {
                        A_coo->off_proc->add_value(first_local_row + i, first_col + j, value);
                        offd_col_set.insert(first_col + j);
                    }
count++;
                }
            }
        }
    }
    offd_num_cols = offd_col_set.size();

MPI_Reduce(&count, &g_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

int tmpl = A_coo->on_proc->nnz + A_coo->off_proc->nnz;
int tmpg;
MPI_Reduce(&tmpl, &tmpg, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    delete[] data;
    fclose(infile);

    A_coo->local_num_rows = diag_num_cols;
    A_coo->local_num_cols = diag_num_cols;
    A_coo->off_proc_num_cols = offd_num_cols;

    global_block_rows = 0;
    sizes = new int[num_procs];
    displs = new int[num_procs+1];
    MPI_Allgather(&block_diag_num_cols, 1, MPI_INT, sizes, 1, MPI_INT, MPI_COMM_WORLD);
    displs[0] = 0;
    for (int i = 0; i < num_procs; i++)
    {
        displs[i+1] = displs[i] + sizes[i];
    }
    global_block_rows = displs[num_procs];
    first_block_row = displs[rank];

    orig_block_rows = new int[global_block_rows];
    MPI_Allgatherv(global_indices.data(), block_diag_num_cols, MPI_INT, orig_block_rows, 
            sizes, displs, MPI_INT, MPI_COMM_WORLD);
    

    // offd-columns, local to original values
    int* local_block_to_global = new int[block_offd_num_cols];
    
    // Add diag orig values to vector and sort
    std::vector<int> orig_vector;
    orig_vector.reserve(block_offd_num_cols);
    for (int i = 0; i < block_offd_num_cols; i++)
    {
        orig_vector.push_back(global_indices[block_diag_num_cols + i]);
    }
    std::sort(orig_vector.begin(), orig_vector.end());

    // Go through global indices, one process at a time
    std::vector<std::pair<int, int>> proc_rows;
    std::vector<std::pair<int,int>>::iterator it;
    for (int proc = 0; proc < num_procs; proc++)
    {
        // Add new and global indices to list
        proc_start = displs[proc];
        proc_end = displs[proc+1];
        proc_rows.resize(proc_end - proc_start);
        for (int i = proc_start; i < proc_end; i++)
        {
            proc_rows[i-proc_start] = std::make_pair(orig_block_rows[i], i);
        }
        std::sort(proc_rows.begin(), proc_rows.end(),
                [](const std::pair<int, int>& lhs,
                    const std::pair<int, int>& rhs)
                { return lhs.first < rhs.first; } );

        ctr = 0;
        for (it = proc_rows.begin(); it != proc_rows.end(); ++it)
        {
            while (ctr + 1 < block_offd_num_cols && orig_vector[ctr+1] <= it->first) ctr++;
            if (orig_vector[ctr] == it->first)
            {
                local_block_to_global[ctr] = it->second;
            }
        }
    }

    A_coo->global_num_rows = global_block_rows * block_size;
    A_coo->global_num_cols = A_coo->global_num_rows;
    A_coo->first_local_row = first_block_row * block_size;
    A_coo->first_local_col = A_coo->first_local_row;

    A_coo->local_nnz = A_coo->on_proc->nnz + A_coo->off_proc->nnz;
    if (A_coo->off_proc_num_cols) A_coo->off_proc_column_map.resize(A_coo->off_proc_num_cols);
    std::map<int, int> global_to_local;

    ctr = 0;
    for (std::set<int>::iterator it = offd_col_set.begin(); it != offd_col_set.end(); ++it)
    {
        A_coo->off_proc_column_map[ctr] = *it;
        global_to_local[*it] = ctr++;
    }

    if (A_coo->offd_num_cols)
    {
        std::sort(A_coo->offd_column_map.begin(), A_coo->offd_column_map.end());
        for (index_t i = 0; i < A_coo->offd_column_map.size(); i++)
        {
            index_t global_col = A_coo->offd_column_map[i];
            std::map<index_t, index_t>::iterator map_it = 
                global_to_local.find(global_col);
            map_it->second = i;
        }
    }

    ParMatrix* A = new ParMatrix(A_coo);
    A->finalize();
    delete A_coo;

    int* global_tmp = new int[A->local_num_rows];
    for (int i = 0; i < block_diag_num_cols; i++)
    {
        for (int j = 0; j < block_size; j++)
        {
            global_tmp[i*block_size + j] = global_indices[i]*block_size + j;
        }
    }
    *global_num_rows = global_tmp;
    
    delete[] orig_block_rows;
    delete[] sizes;
    delete[] displs;
    delete[] local_block_to_global;

    return A;
}
