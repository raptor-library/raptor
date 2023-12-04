// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "stencil.hpp"

namespace raptor {
// Stencils are symmetric, so A could be CSR or CSC
CSRMatrix* stencil_grid(data_t* stencil, int* grid, int dim)
{
    std::vector<int> diags;
    std::vector<double> nonzero_stencil;
    std::vector<int> strides(dim);
    std::vector<double> data;
    std::vector<int> stack_indices;

    int stencil_len, ctr;
    int N_v;  // Number of rows (and cols) in matrix
    int N_s;  // Number of nonzero stencil entries
    int init_step, idx;
    int len, step;
    int col;
    double value;

    stencil_len = (int)pow(3, dim);

    N_v = 1;
    for (int i = 0; i < dim; i++)
    {
        N_v *= grid[i];
    }

    N_s = 0;
    for (int i = 0; i < stencil_len; i++)
    {
        if (fabs(stencil[i]) > zero_tol)
        {
            N_s++;
        }
    }

    // Set dimensions of A
    CSRMatrix* A = new CSRMatrix(N_v, N_v);

    diags.resize(N_s, 0);
    nonzero_stencil.resize(N_s);
    strides[0] = 1;
    for (int i = 0; i < dim - 1; i++)
    {
        strides[i+1] = grid[dim-i-1] * strides[i];
    }

    // Calculate indices of nonzeros in  stencil
    int indices[N_s][dim];
    ctr = 0;
    for (int i = 0; i < stencil_len; i++)
    {
        if (fabs(stencil[i]) > zero_tol)
        {
            for (int j = 0; j < dim; j++)
            {
                //int power = pow(3, j);
                int idiv = i / pow(3, j);
                indices[ctr][dim-j-1] = (idiv % 3) - (3 / 2);
            }
            nonzero_stencil[ctr] = stencil[i];
            ctr++;
        }
    }

    // Add strides to diags
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < N_s; j++)
        {
            diags[j] += strides[i] * indices[j][dim-i-1];
        }
    }

    // Initial data array
    data.resize(N_s*N_v);
    for (int i = 0; i < N_s; i++)
    {
        for (int j = 0; j < N_v; j++)
        {
            data[i*N_v + j] = nonzero_stencil[i];
        }
    }

    // Vertically stack indices (reorder)
    stack_indices.resize(N_s*dim);
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
        init_step = i*N_v;
        for (int j = 0; j < dim; j++)
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
            for (int k = 0; k < (dim-j-1); k++)
            {
                len *= grid[k];
            }
            step = len * grid[0];

            //zeros at beginning
            if (idx > 0)
            {
                //If previous boundary lies on processor
                for (int k = 0; k < N_v; k+=step)
                {
                    for (int l = 0; l < len; l++)
                    {
                        if (k+l > N_v)
                        {
                            break;
                        }
                        if (k+l < 0)
                        {
                            continue;
                        }
                        data[init_step + (k-0) + l] = 0;
                    }
                }
            }

            //zeros at end
            else if (idx < 0)
            {
                //If previous boundary lies on processor
                for (int k = N_v; k > 0; k-=step)
                {
                    for (int l = 0; l < len; l++)
                    {
                        if (k - l - 1 < 0)
                        {
                            break;
                        }
                        else if (k - l - 1 > N_v)
                        {
                            continue;
                        }
                        data[init_step + (k-l-0) -1] = 0;
                    }
                }
            }
        }
    }

    //Add diagonals to ParMatrix A
    A->idx2.reserve(N_s*N_v);
    A->vals.reserve(N_s*N_v);

    A->idx1[0] = 0;
    for (int i = 0; i < N_v; i++)
    {
        for (int d = 0; d < N_s; d++)
        {
            //add data[i] if nonzero 
            col = diags[d] + i;
            value = data[(N_s-d-1)*N_v+i];
            if (col >= 0 && col < N_v && fabs(value) > zero_tol)
            //if (fabs(value) > zero_tol)
            {
                A->idx2.emplace_back(col);
                A->vals.emplace_back(value);
            }
        }
        A->idx1[i+1] = A->idx2.size();
    }
    A->nnz = A->idx2.size();

    return A;
}

}
