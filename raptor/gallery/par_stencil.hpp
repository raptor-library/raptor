// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef PARSTENCIL_HPP
#define PARSTENCIL_HPP

#include <float.h>
#include <cmath>
#include <stdlib.h>

#include "core/types.hpp"
#include "core/par_matrix.hpp"

using namespace raptor;

ParCSRMatrix* par_stencil_grid(data_t* stencil, int* grid, int dim)
{
    // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::vector<int> diags;
    std::vector<double> nonzero_stencil;
    std::vector<int> strides(dim);
    std::vector<double> data;
    std::vector<int> stack_indices;

    int stencil_len, ctr;
    int N_v;  // Number of rows (and cols) in matrix
    int N_s;  // Number of nonzero stencil entries
    int n_v;  // Local number of rows (and cols)
    int extra, first_local, last_local;
    int init_step, idx;
    int len, step, current_step;
    int col;
    double value;

    // Initialize variables
    stencil_len = (index_t)pow(3, dim); // stencil - 3 ^ dim

    //N_v is global number of rows
    N_v = 1;
    for (index_t i = 0; i < dim; i++)
    {
       N_v *= grid[i];
    }

    //N_s is number of nonzero stencil entries
    N_s = 0;
    for (index_t i = 0; i < stencil_len; i++)
    {
        if (fabs(stencil[i]) > zero_tol)
        {
            N_s++;
        }
    }

    ParCSRMatrix* A = new ParCSRMatrix(N_v, N_v);

    n_v = A->partition->local_num_rows;
    int first_local_row = A->partition->first_local_row;
    int last_local_row = first_local_row + n_v - 1;

    A->on_proc->n_rows = n_v;
    A->on_proc->n_cols = n_v;
    A->on_proc->nnz = 0;
    A->on_proc->idx1.resize(n_v+1);
    A->on_proc->idx2.reserve(n_v*stencil_len);
    A->on_proc->vals.reserve(n_v*stencil_len);

    A->off_proc->n_rows = n_v;
    A->off_proc->n_cols = N_v;
    A->off_proc->nnz = 0;
    A->off_proc->idx1.resize(n_v+1);
    A->off_proc->idx2.reserve(0.3*n_v*stencil_len);
    A->off_proc->vals.reserve(0.3*n_v*stencil_len);


    diags.resize(N_s, 0);
    nonzero_stencil.resize(N_s);
    strides.resize(dim);
    //Calculate strides for index offset for each dof in stencil
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
    data.resize(N_s*n_v);
    for (index_t i = 0; i < N_s; i++)
    {
        for (index_t j = 0; j < n_v; j++)
        {
            data[i*n_v + j] = nonzero_stencil[i];
        }
    }

    //Vertically stack indices (reorder)
    stack_indices.resize(N_s*dim);
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

    //Add diagonals to ParMatrix A
    A->on_proc->idx1[0] = 0;
    A->off_proc->idx1[0] = 0;
    for (index_t i = 0; i < n_v; i++)
    {
        for (index_t d = 0; d < N_s; d++)
        {
            //add data[i] if nonzero 
            col = diags[d] + i + first_local_row;
            value = data[(N_s-d-1)*n_v+i];
            if (col >= 0 && col < N_v && fabs(value) > zero_tol)
            {
                A->add_value(i, col, value);
            }
        }
        A->on_proc->idx1[i+1] = A->on_proc->idx2.size();
        A->off_proc->idx1[i+1] = A->off_proc->idx2.size();
    }

    A->on_proc->nnz = A->on_proc->idx2.size();
    A->off_proc->nnz = A->off_proc->idx2.size();
    
    A->finalize();

    return A;
} 

#endif


