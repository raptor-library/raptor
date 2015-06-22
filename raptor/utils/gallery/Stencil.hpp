// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef STENCIL_HPP
#define STENCIL_HPP

#include <mpi.h>
#include <math.h>
#include <Eigen/Dense>
using Eigen::VectorXd;

#include "ParMatrix.hpp"

ParMatrix* stencil_grid(double* stencil, int* grid, int dim)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    //double is never 0.0 ... need zero tolerance
    double zero_tol = 1e-6;

    // stencil - 3 ^ dim
    int stencil_len = (int)pow(3, dim);

    //N_v is global number of rows
    int N_v = 1;
    for (int i = 0; i < dim; i++)
    {
       N_v *= grid[i];
    }

    //n_v is local number of rows
    int extra = N_v % num_procs;
    int* globalRowStarts = (int*) calloc(num_procs+1, sizeof(int));
    for (int i = 0; i < num_procs; i++)
    {
        int size = (N_v / num_procs) + (extra/(i+1));
        globalRowStarts[i+1] = globalRowStarts[i] + size;
    }
    int firstLocalRow = globalRowStarts[rank];
    int lastLocalRow = globalRowStarts[rank+1] - 1;
    int n_v = globalRowStarts[rank+1] - firstLocalRow;
    
    //N_s is number of nonzero stencil entries
    int N_s = 0;
    for (int i = 0; i < stencil_len; i++)
    {
        if (fabs(stencil[i]) > zero_tol)
        {
            N_s++;
        }
    }

    int* diags = (int*) calloc(N_s, sizeof(int));
    double nonzero_stencil[N_s];
    //Calculate strides for index offset for each dof in stencil
    int strides[dim];
    strides[0] = 1;
    for (int i = 0; i < dim-1; i++)
    {
        strides[i+1] = grid[dim-i-1] * strides[i];
    }

    //Calculate indices of nonzeros in  stencil
    int indices[N_s][dim];
    int ctr = 0;
    for (int i = 0; i < stencil_len; i++)
    {
        if (fabs(stencil[i]) > zero_tol)
        {
            for (int j = 0; j < dim; j++)
            {
                int power = pow(3, j);
                int idiv = i / power;
                indices[ctr][dim-j-1] = (idiv % 3) - (3 / 2);
            }
            nonzero_stencil[ctr] = stencil[i];
            ctr++;
        }
    }

    //Add strides to diags
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < N_s; j++)
        {
            diags[j] += strides[i] * indices[j][dim-i-1];
        }
    } 

    //Initial data array
    double data[N_s * n_v];
    for (int i = 0; i < N_s; i++)
    {
        for (int j = 0; j < n_v; j++)
        {
            data[i*n_v + j] = nonzero_stencil[i];
        }
    }

    //Vertically stack indices (reorder)
    int stack_indices[N_s*dim];
    for (int i = 0; i < N_s; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            stack_indices[i*dim+j] = indices[i][j];
        }
    }

    //Zero boundary conditions
    for (int i = 0; i < N_s; i++)
    {
        //get correct chunk of data 
        //(corresponding to single stencil entry)
        int init_step = i*n_v;
        for (int j = 0; j < dim; j++)
        {
            //If main diagonal, no boundary conditions
            int idx = stack_indices[i*dim + j];
            if (idx == 0)
            {
                continue;
            }

            //Calculate length of chunks that are to
            // be set to zero, and step size between
            // these blocks of data
            int len = 1;
            int step = 1;
            for (int k = 0; k < (dim-j-1); k++)
            {
                len *= grid[k];
            }
            step = len * grid[0];

            int pos = 0;
            int current_step = 0;
            int next_pos = 0;

            //zeros at beginning
            if (idx > 0)
            {
                current_step = step * (firstLocalRow / step);

                //If previous boundary lies on processor
                for (int k = current_step; k < lastLocalRow+1; k+=step)
                {
                    for (int l = 0; l < len; l++)
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
                for (int k = current_step; k > firstLocalRow; k-=step)
                {
                    for (int l = 0; l < len; l++)
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
    int* row_ptr = (int*) calloc(n_v+1, sizeof(int));
    int* col_idx = (int*) calloc(n_v*N_s, sizeof(int));
    double* values = (double*) calloc(n_v*N_s, sizeof(double));
    int nnz = 0;
    for (int i = 0; i < n_v; i++)
    {
        row_ptr[i] = nnz;
        for (int d = 0; d < N_s; d++)
        {
            //add data[i] if nonzero 
            int col = diags[d] + i + firstLocalRow;
            double value = data[(N_s-d-1)*n_v+i];
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


    return new ParMatrix(N_v, N_v, row_ptr, col_idx, values, globalRowStarts);
    
} 

#endif
