// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "linear_elasticity.hpp"

Eigen::MatrixXd* q12d_local(data_t* vertices, data_t lame, data_t mu)
{
    data_t M = lame + 2 * mu;

    Eigen::Matrix4f R_11;
    Eigen::Matrix4f R_12;
    Eigen::Matrix4f R_22;

    R_11 << 2, -2, -1, 1,
            -2, 2, 1, -1,
            -1, 1, 2, -2,
            1, -1, -2, 2;

    R_11 = R_11 / 6.0;

    R_12 << 1, 1, -1, -1,
            -1, -1, 1, 1,
            -1, -1, 1, 1,
            1, 1, -1, -1;

    R_12 = R_12 / 4.0;

    R_22 << 2, 1, -1, -2,
            1, 2, -2, -1,
            -1, -2, 2, 1,
            -2, -1, 1, 2;

    R_22 = R_22 / 6.0;

    Eigen::Matrix2f F;
    F << vertices[2] - vertices[0], vertices[3] - vertices[1],
        vertices[6] - vertices[0], vertices[7] - vertices[1];

    Eigen::MatrixXd* K = new Eigen::MatrixXd(8, 8);
    Eigen::Matrix4f K_1;
    Eigen::Matrix4f K_2;
    Eigen::Matrix4f K_3;
    Eigen::Matrix4f K_4;

    Eigen::Matrix2f E;

    Eigen::Matrix2f tmp;

    tmp << M, 0, 0, mu;
    E = F.transpose() * tmp * F;
    K_1 = E(0, 0) * R_11 + E(0, 1) * R_12 +
        E(1, 0) * R_12.transpose() + E(1, 1) * R_22;

    tmp << mu, 0, 0, M;
    E = F.transpose() * tmp * F;
    K_2 = E(0, 0) * R_11 + E(0, 1) * R_12 +
        E(1, 0) * R_12.transpose() + E(1, 1) * R_22;
    
    tmp << 0, mu, lame, 0;
    E = F.transpose() * tmp * F;
    K_3 = E(0, 0) * R_11 + E(0, 1) * R_12 +
        E(1, 0) * R_12.transpose() + E(1, 1) * R_22;
    
    K_4 = K_3.transpose();

    for (index_t i = 0; i < 8; i++)
    {
        for (index_t j = 0; j < 8; j++)
        {
            if (i % 2 == 0)
            {
                if (j % 2 == 0)
                {
                    ((*K)(i, j)) = K_1(i / 2, j / 2);
                }
                else
                {
                    ((*K)(i, j)) = K_4(i / 2, (j - 1) / 2);
                }
            }
            else
            {
                if (j % 2 == 0)
                {
                    ((*K)(i, j)) = K_3((i - 1) / 2, j / 2);
                }
                else
                {
                    ((*K)(i, j)) = K_2((i - 1) / 2, (j - 1) / 2);
                }
            }
        }
    }

    (*K) = (*K) / F.determinant();

    return K;
}

ParMatrix* linear_elasticity(index_t* grid, ParMatrix** B, data_t E, data_t nu, index_t dirichlet, data_t* spacing)
{
    index_t X = grid[0];
    index_t Y = grid[1];

    data_t DX = 1.;
    data_t DY = 1.;

    if (dirichlet)
    {
        X++;
        Y++;
    }

    index_t nx = X + 1;
    index_t ny = Y + 1;
    index_t N = 2 * nx * ny;

    ParMatrix* A = new ParMatrix(N, N);

    index_t n = A->local_rows;
    index_t first_row = A->first_row;

    data_t* pts = new data_t[n];

    for (int i = 0; i < n; i++)
    {
        index_t global_row = i + first_row;
        if (global_row % 2 == 0)
        {
            pts[i] = -1.0 * (X / 2.0) + ((global_row/2) % nx);
        }
        else
        {
            pts[i] = -1.0 * (Y / 2.0) + floor((global_row/2)/ nx);
        }
    }

    if (spacing)
    {
        DX = spacing[0];
        DY = spacing[1];
    }

    // Lame's first parameter
    data_t lame = E * nu / ((1 + nu) * (1 - 2*nu));
    // Shear modulus
    data_t mu = E / (2 + 2*nu);

    data_t vertices[8] = {0, 0, DX, 0, DX, DY, 0, DY};
    Eigen::MatrixXd* K = q12d_local(vertices, lame, mu);

    index_t n_pos = 8;
    index_t pos[8] = {0, 1, 2, 3, 2*X + 4, 2*X + 5, 2*X + 2, 2*X + 3};

    index_t max_pos = (2*X*(Y+1));

    index_t global_row;
    index_t row_start;

    for (index_t row = 0; row < n; row++)
    {
        global_row = row + first_row;
        for (index_t i = 0; i < n_pos; i++)
        {
            row_start = global_row - pos[i];
            if (row_start >= 0 &&
                row_start < max_pos &&
                row_start % 2 == 0 && 
                (row_start / 2 + 1) % (Y + 1))
            {
                for (index_t j = 0; j < n_pos; j++)
                {
                    A->add_value(row,
                                row_start + pos[j],
                                ((*K)(i, j)));
                }
            }
        }
    }

    *B = new ParMatrix(2*(X+1)*(Y+1), 3);
    for (index_t i = 0; i < (*B)->local_rows; i++)
    {
        index_t global_row = (*B)->first_row + i;
        if (global_row % 2 == 0)
        {
            (*B)->add_value(i, 0, 1.0);
            (*B)->add_value(i, 2, -pts[i+1]);
        }
        else
        {
            (*B)->add_value(i, 1, 1.0);
            (*B)->add_value(i, 2, pts[i-1]);
        }
    }

    A->finalize(1);
    delete[] pts;
    delete K;


    if (dirichlet)
    {
        ParMatrix* P = new ParMatrix(2*(X+1)*(Y+1), 2*(X-1)*(Y-1));
        index_t init = 2*(Y+2);
        index_t x_len = 2*(Y+1);
        for (index_t i = 0; i < P->local_rows; i++)
        {
            index_t global_row = i + P->first_row;
            if (global_row >= init && global_row + init < P->global_rows)
            {
                index_t relative = global_row - init;
                index_t extra = relative % x_len;

                // Add to P in correct position
                if (extra < 2*(Y-1))
                {
                    index_t col = (2*(Y-1)*(relative / x_len)) + extra;
                    P->add_value(i, col, 1.0);
                }
            }
        }
        P->finalize(0);

        ParMatrix* A_tmp;
        ParMatrix* A_dirichlet;
        parallel_matmult(A, P, &A_tmp);
        delete A;
        parallel_matmult_T(A_tmp, P, &A_dirichlet);
        delete A_tmp;
        delete P;
        return A_dirichlet;
    }
    else
    {
        return A;
    }
}
