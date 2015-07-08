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

ParMatrix* stencil_grid(data_t* stencil, index_t* grid, index_t dim)
{
    // Get MPI Information
    index_t rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    //double is never 0.0 ... need zero tolerance
    data_t zero_tol = 1e-6;

    // stencil - 3 ^ dim
    index_t stencil_len = (index_t)pow(3, dim);

    //N_v is global number of rows
    index_t N_v = 1;
    for (index_t i = 0; i < dim; i++)
    {
       N_v *= grid[i];
    }

    //n_v is local number of rows
    index_t extra = N_v % num_procs;
    index_t* globalRowStarts = (int*) calloc(num_procs+1, sizeof(int));
    for (index_t i = 0; i < num_procs; i++)
    {
        index_t size = (N_v / num_procs);
        if (i < extra)
        {
            size++;
        }
        globalRowStarts[i+1] = globalRowStarts[i] + size;
    }
    index_t firstLocalRow = globalRowStarts[rank];
    index_t lastLocalRow = globalRowStarts[rank+1] - 1;
    index_t n_v = lastLocalRow - firstLocalRow + 1;
    
    //N_s is number of nonzero stencil entries
    index_t N_s = 0;
    for (index_t i = 0; i < stencil_len; i++)
    {
        if (fabs(stencil[i]) > zero_tol)
        {
            N_s++;
        }
    }

    index_t* diags = (index_t*) calloc(N_s, sizeof(index_t));
    data_t nonzero_stencil[N_s];
    //Calculate strides for index offset for each dof in stencil
    index_t strides[dim];
    strides[0] = 1;
    for (index_t i = 0; i < dim-1; i++)
    {
        strides[i+1] = grid[dim-i-1] * strides[i];
    }

    //Calculate indices of nonzeros in  stencil
    index_t indices[N_s][dim];
    index_t ctr = 0;
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
    data_t data[N_s * n_v];
    for (index_t i = 0; i < N_s; i++)
    {
        for (index_t j = 0; j < n_v; j++)
        {
            data[i*n_v + j] = nonzero_stencil[i];
        }
    }

    //Vertically stack indices (reorder)
    index_t stack_indices[N_s*dim];
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
        index_t init_step = i*n_v;
        for (index_t j = 0; j < dim; j++)
        {
            //If main diagonal, no boundary conditions
            index_t idx = stack_indices[i*dim + j];
            if (idx == 0)
            {
                continue;
            }

            //Calculate length of chunks that are to
            // be set to zero, and step size between
            // these blocks of data
            index_t len = 1;
            index_t step = 1;
            for (index_t k = 0; k < (dim-j-1); k++)
            {
                len *= grid[k];
            }
            step = len * grid[0];

            index_t pos = 0;
            index_t current_step = 0;
            index_t next_pos = 0;

            //zeros at beginning
            if (idx > 0)
            {
                current_step = step * (firstLocalRow / step);

                //If previous boundary lies on processor
                for (index_t k = current_step; k < lastLocalRow+1; k+=step)
                {
                    for (index_t l = 0; l < len; l++)
                    {
                        if (k+l > lastLocalRow)
                        {
                            break;
                        }
                        if (k+l < firstLocalRow)
                        {
                            continue;
                        }
                        data[init_step + (k-firstLocalRow) + l] = 0;
                    }
                }
            }

            //zeros at end
            else if (idx < 0)
            {
                current_step = step*(((lastLocalRow-1)/step)+1);

                //If previous boundary lies on processor
                for (index_t k = current_step; k > firstLocalRow; k-=step)
                {
                    for (index_t l = 0; l < len; l++)
                    {
                        if (k - l - 1 < firstLocalRow)
                        {
                            break;
                        }
                        else if (k - l - 1 > lastLocalRow)
                        {
                            continue;
                        }
                        data[init_step + (k-l-firstLocalRow) -1] = 0;
                    }
                }
            }
        }
    }

    //Add diagonals to a csr matrix
    index_t* row_ptr = (index_t*) calloc(n_v+1, sizeof(index_t));
    index_t* col_idx = (index_t*) calloc(n_v*N_s, sizeof(index_t));
    data_t* values = (data_t*) calloc(n_v*N_s, sizeof(data_t));
    index_t nnz = 0;
    for (index_t i = 0; i < n_v; i++)
    {
        row_ptr[i] = nnz;
        for (index_t d = 0; d < N_s; d++)
        {
            //add data[i] if nonzero 
            index_t col = diags[d] + i + firstLocalRow;
            data_t value = data[(N_s-d-1)*n_v+i];
            if (col >= 0 && col < N_v && fabs(value) > zero_tol)
            //if (col >= 0 && col < N_v)
            {
                col_idx[nnz] = col;
                values[nnz] = value;
                nnz++;
            }
        }
    }
    row_ptr[n_v] = nnz;
    return new ParMatrix(N_v, N_v, nnz, row_ptr, col_idx, values, globalRowStarts);
    
} 

#endif
