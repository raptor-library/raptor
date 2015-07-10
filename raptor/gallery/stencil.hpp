// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef STENCIL_HPP
#define STENCIL_HPP

#include <mpi.h>
#include <cmath>
#include <stdlib.h>
#include <Eigen/Dense>
using Eigen::VectorXd;

#include "core/par_matrix.hpp"
#include "core/types.hpp"

ParMatrix* stencil_grid(data_t* stencil, index_t* grid, index_t dim, format_t format = CSR)
{
    // Get MPI Information
    index_t rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Declare stencil grid variables
    data_t zero_tol;
    data_t value;
    data_t* nonzero_stencil;
    data_t* data;
    data_t* values;
    index_t stencil_len;
    index_t N_v; // global num rows
    index_t n_v; // local num rows
    index_t N_s; // number of stencil elements
    index_t extra;
    index_t first_local_row;
    index_t last_local_row;
    index_t ctr;
    index_t init_step;
    index_t idx;
    index_t len;
    index_t step;
    index_t current_step;
    index_t col;
    index_t nnz;
    index_t* global_row_starts;
    index_t* diags;
    index_t* strides;
    index_t* stack_indices;
    index_t* row_ptr;
    index_t* col_idx;

    // Initialize variables
    zero_tol = 1e-6; //data_t is never equal to 0
    stencil_len = (index_t)pow(3, dim); // stencil - 3 ^ dim

    //N_v is global number of rows
    N_v = 1;
    for (index_t i = 0; i < dim; i++)
    {
       N_v *= grid[i];
    }

    //n_v is local number of rows
    extra = N_v % num_procs;
    global_row_starts = (int*) calloc(num_procs+1, sizeof(int));
    for (index_t i = 0; i < num_procs; i++)
    {
        index_t size = (N_v / num_procs);
        if (i < extra)
        {
            size++;
        }
        global_row_starts[i+1] = global_row_starts[i] + size;
    }
    first_local_row = global_row_starts[rank];
    last_local_row = global_row_starts[rank+1] - 1;
    n_v = last_local_row - first_local_row + 1;
    
    //N_s is number of nonzero stencil entries
    N_s = 0;
    for (index_t i = 0; i < stencil_len; i++)
    {
        if (fabs(stencil[i]) > zero_tol)
        {
            N_s++;
        }
    }

    diags = (index_t*) calloc(N_s, sizeof(index_t));
    nonzero_stencil = (data_t*) calloc(N_s, sizeof(data_t));
    //Calculate strides for index offset for each dof in stencil
    strides = (index_t*) calloc (dim, sizeof(index_t));
    strides[0] = 1;
    for (index_t i = 0; i < dim-1; i++)
    {
        strides[i+1] = grid[dim-i-1] * strides[i];
    }

    //Calculate indices of nonzeros in  stencil
    index_t indices[N_s][dim];
    ctr = 0;
    for (index_t i = 0; i < stencil_len; i++)
    {
        if (fabs(stencil[i]) > zero_tol)
        {
            for (index_t j = 0; j < dim; j++)
            {
                index_t power = pow(3, j);
                index_t idiv = i / power;
                indices[ctr][dim-j-1] = (idiv % 3) - (3 / 2);
            }
            nonzero_stencil[ctr] = stencil[i];
            ctr++;
        }
    }

    //Add strides to diags
    for (index_t i = 0; i < dim; i++)
    {
        for (index_t j = 0; j < N_s; j++)
        {
            diags[j] += strides[i] * indices[j][dim-i-1];
        }
    } 

    //Initial data array
    data = (data_t*) calloc (N_s*n_v, sizeof(data_t));
    for (index_t i = 0; i < N_s; i++)
    {
        for (index_t j = 0; j < n_v; j++)
        {
            data[i*n_v + j] = nonzero_stencil[i];
        }
    }

    //Vertically stack indices (reorder)
    stack_indices = (index_t*) calloc (N_s*dim, sizeof(index_t));
    for (index_t i = 0; i < N_s; i++)
    {
        for (index_t j = 0; j < dim; j++)
        {
            stack_indices[i*dim+j] = indices[i][j];
        }
    }

    //Zero boundary conditions
    for (index_t i = 0; i < N_s; i++)
    {
        //get correct chunk of data 
        //(corresponding to single stencil entry)
        init_step = i*n_v;
        for (index_t j = 0; j < dim; j++)
        {
            //If main diagonal, no boundary conditions
            idx = stack_indices[i*dim + j];
            if (idx == 0)
            {
                continue;
            }

            //Calculate length of chunks that are to
            // be set to zero, and step size between
            // these blocks of data
            len = 1;
            step = 1;
            for (index_t k = 0; k < (dim-j-1); k++)
            {
                len *= grid[k];
            }
            step = len * grid[0];
            current_step = 0;

            //zeros at beginning
            if (idx > 0)
            {
                current_step = step * (first_local_row / step);

                //If previous boundary lies on processor
                for (index_t k = current_step; k < last_local_row+1; k+=step)
                {
                    for (index_t l = 0; l < len; l++)
                    {
                        if (k+l > last_local_row)
                        {
                            break;
                        }
                        if (k+l < first_local_row)
                        {
                            continue;
                        }
                        data[init_step + (k-first_local_row) + l] = 0;
                    }
                }
            }

            //zeros at end
            else if (idx < 0)
            {
                current_step = step*(((last_local_row-1)/step)+1);

                //If previous boundary lies on processor
                for (index_t k = current_step; k > first_local_row; k-=step)
                {
                    for (index_t l = 0; l < len; l++)
                    {
                        if (k - l - 1 < first_local_row)
                        {
                            break;
                        }
                        else if (k - l - 1 > last_local_row)
                        {
                            continue;
                        }
                        data[init_step + (k-l-first_local_row) -1] = 0;
                    }
                }
            }
        }
    }

    if (format == CSR)
    {
        //Add diagonals to a csr matrix
        row_ptr = (index_t*) calloc(n_v+1, sizeof(index_t));
        col_idx = (index_t*) calloc(n_v*N_s, sizeof(index_t));
        values = (data_t*) calloc(n_v*N_s, sizeof(data_t));
        nnz = 0;
        for (index_t i = 0; i < n_v; i++)
        {
            row_ptr[i] = nnz;
            for (index_t d = 0; d < N_s; d++)
            {
                //add data[i] if nonzero 
                col = diags[d] + i + first_local_row;
                value = data[(N_s-d-1)*n_v+i];
                if (col >= 0 && col < N_v && fabs(value) > zero_tol)
                {
                    col_idx[nnz] = col;
                    values[nnz] = value;
                    nnz++;
                }
            }
        }
        row_ptr[n_v] = nnz;
    }
    else if (format == COO)
    {
        // Add diagonals to a COO matrix
        row_ptr = (index_t*) calloc(n_v*N_s, sizeof(index_t));
        col_idx = (index_t*) calloc(n_v*N_s, sizeof(index_t));
        values = (data_t*) calloc(n_v*N_s, sizeof(data_t));
        nnz = 0;
        for (index_t i = 0; i < n_v; i++)
        {
            for (index_t d = 0; d < N_s; d++)
            {
                //add data[i] if nonzero 
                col = diags[d] + i + first_local_row;
                value = data[(N_s-d-1)*n_v+i];
                if (col >= 0 && col < N_v && fabs(value) > zero_tol)
                {
                    row_ptr[nnz] = i;
                    col_idx[nnz] = col;
                    values[nnz] = value;
                    nnz++;
                }
            }
        }
    }

    ParMatrix* A = new ParMatrix(N_v, N_v, nnz, row_ptr, col_idx, values, global_row_starts, format = format, 0);

    free(row_ptr);
    free(col_idx);
    free(values);

    free(nonzero_stencil);
    free(data);
    free(diags);
    free(strides);
    free(stack_indices);
    free(global_row_starts);

    return A;
} 

#endif
