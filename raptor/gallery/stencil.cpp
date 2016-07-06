// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "stencil.hpp"


/**************************************************************
 *****   Stencil Grid
 **************************************************************
 ***** Generate a matrix from a specified stencil, and of a
 ***** specified size;
 *****
 ***** Parameters
 ***** -------------
 ***** stencil : data_t*
 *****    Stencil of the matrix
 ***** grid : int*
 *****    Dimensions of the matrix
 ***** dim : int
 *****    Dimension of grid variable
 *****
 ***** Returns
 ***** -------------
 ***** ParMatrix*
 *****    Matrix formed from stencil
 *****
 **************************************************************/
ParMatrix* stencil_grid(data_t* stencil, int* grid, int dim)
{
    // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Declare stencil grid variables
    data_t value;
    data_t* nonzero_stencil;
    data_t* data;
    int stencil_len;
    index_t N_v; // global num rows
    int n_v; // local num rows
    int N_s; // number of stencil elements
    int extra;
    index_t first_local_row;
    index_t last_local_row;
    int ctr;
    int init_step;
    int idx;
    int len;
    int step;
    int current_step;
    index_t col;
    int nnz;
    int* diags;
    int* strides;
    int* stack_indices;

    ParMatrix* A;

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

    A = new ParMatrix(N_v, N_v);

    n_v = A->local_rows;
    first_local_row = A->first_row;
    last_local_row = first_local_row + n_v - 1;

    index_t nnz_per;
    if (n_v < N_s)
    {
        nnz_per = n_v;
    }
    else
    {
        nnz_per = N_s;
    }

    diags = new int[N_s]();
    nonzero_stencil = new data_t[N_s];
    //Calculate strides for index offset for each dof in stencil
    strides = new int[dim];
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
    data = new data_t[N_s*n_v];
    for (index_t i = 0; i < N_s; i++)
    {
        for (index_t j = 0; j < n_v; j++)
        {
            data[i*n_v + j] = nonzero_stencil[i];
        }
    }

    //Vertically stack indices (reorder)
    stack_indices = new int[N_s*dim];
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

    //Add diagonals to ParMatrix A
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
    }
    
    A->finalize();

    delete[] nonzero_stencil;
    delete[] data;
    delete[] diags;
    delete[] strides;
    delete[] stack_indices;

    return A;
} 
