#ifndef RAPTOR_UTILS_LINALG_SPMV_H
#define RAPTOR_UTILS_LINALG_SPMV_H

#include <mpi.h>
#include <float.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Core>
//using Eigen::VectorXd;

#include "core/vector.hpp"
#include "core/par_matrix.hpp"
#include "core/matrix.hpp"
#include "core/par_vector.hpp"
using namespace raptor;

//void sequentialSPMV(CSRMatrix* A, Vector* x, Vector* y, double alpha, double beta);
//void parallelSPMV(ParMatrix* A, ParVector* x, ParVector* y, double alpha, double beta);

template <int MatType>
void sequentialSPMV_eigen(Matrix<MatType>* A, Vector* x, Vector* y, double alpha, double beta, index_t first_col = -1, index_t size = 0)
{ 
    //TODO -- should be std::numeric_limits<data_t>::epsilon ...
    data_t zero_tol = DBL_EPSILON;

    index_t alpha_zero = (fabs(alpha) < zero_tol);
    index_t alpha_one = (fabs(alpha - 1.0) < zero_tol);
    index_t alpha_neg_one = (fabs(alpha + 1.0) < zero_tol);

    index_t beta_zero = (fabs(beta) < zero_tol);
    index_t beta_one = (fabs(beta - 1.0) < zero_tol);
    index_t beta_neg_one = (fabs(beta + 1.0) < zero_tol);

    auto m = *(A->m);
    
    if (first_col > -1 && size)
    {
        m = m.block(0, first_col, A->m->rows(), size);
    }

    if (alpha_one)
    {
        if (beta_one)
        {
            *y = (m*(*x)) + (*y);
        }
        else if (beta_neg_one)
        {
            *y = (m*(*x)) - (*y);
        }
        else if (beta_zero)
        {
            *y = (m*(*x));
        }
        else
        {
            *y = (m*(*x)) + beta*(*y);
        }
    }
    else if (alpha_neg_one)
    {
        if (beta_one)
        {
            *y = -(m*(*x)) + (*y);
        }
        else if (beta_neg_one)
        {
            *y = -(m*(*x)) - (*y);
        }
        else if (beta_zero)
        {
            *y = -(m*(*x));
        }
        else
        {
            *y = -(m*(*x)) + beta*(*y);
        }
    }
    else if (alpha_zero)
    {
        if (beta_one)
        {
            //*y = (*y);
        }
        else if (beta_neg_one)
        {
            *y = -(*y);
        }
        else if (beta_zero)
        {
            *y *= 0.0;
        }
        else
        {
            *y = beta*(*y);
        }
    }
    else
    {
        if (beta_one)
        {
            *y = alpha*(m*(*x)) + (*y);
        }
        else if (beta_neg_one)
        {
            *y = alpha*(m*(*x)) - (*y);
        }
        else if (beta_zero)
        {
            *y = alpha*(m*(*x));
        }
        else
        {
            *y = alpha*(m*(*x)) + beta*(*y);
        }
    }

}

//CSC SpMV
void sequentialSPMV_raptor(Matrix<0>* A, Vector* x, Vector* y, double alpha, double beta, std::vector<index_t> map, index_t _first_col = -1, index_t _size = 0)
{
    //TODO -- should be std::numeric_limits<data_t>::epsilon ...
    data_t zero_tol = DBL_EPSILON;

    index_t alpha_zero = (fabs(alpha) < zero_tol);
    index_t alpha_one = (fabs(alpha - 1.0) < zero_tol);
    index_t alpha_neg_one = (fabs(alpha + 1.0) < zero_tol);

    index_t beta_zero = (fabs(beta) < zero_tol);
    index_t beta_one = (fabs(beta - 1.0) < zero_tol);
    index_t beta_neg_one = (fabs(beta + 1.0) < zero_tol);

    index_t* ptr = (A->m)->outerIndexPtr();
    index_t* idx = (A->m)->innerIndexPtr();
    data_t* values = (A->m)->valuePtr();
    index_t num_cols = (A->m)->outerSize();
    index_t num_rows = A->n_rows;

    index_t first_col = 0;
    index_t last_col = num_cols;

    if (_first_col > -1)
    {
        first_col = _first_col;
    }
    if (_size > 0)
    {
        last_col = first_col + _size;
    }

    index_t col_start;
    index_t col_end;
    data_t x_val;
    data_t* y_data = y->data();
    data_t* x_data = x->data();

    if (alpha_one)
    {
        if (beta_one)
        {
            for (index_t col = first_col; col < last_col; col++)
            {
                col_start = ptr[col];
                col_end = ptr[col + 1];
                data_t x_val = x_data[col];

                for (index_t j = col_start; j < col_end; j++)
                {
                    index_t row = idx[j];
                    y_data[row] += values[j] * x_val;
                }
            }

            for (index_t row = 0; row < num_rows; row++)
            {
                printf("Y[%d (%d)] = %2.3e\n", map[row], row, y_data[row]);
            }
        }
        else
        {
            printf("Section 2\n");
            for (index_t row = 0; row < num_rows; row++)
            {
                y_data[row] = beta * y_data[row];
            }

            for (index_t col = first_col; col < last_col; col++)
            {
                col_start = ptr[col];
                col_end = ptr[col + 1];
                data_t x_val = x_data[col];
                for (index_t j = col_start; j < col_end; j++)
                {
                    index_t row = idx[j];
                    y_data[row] += values[j] * x_val;
                }
            }
        }
    }
    else if (alpha_neg_one)
    {
        if (beta_one)
        {
            printf("Section 3\n");
            for (index_t col = first_col; col < last_col; col++)
            {
                col_start = ptr[col];
                col_end = ptr[col + 1];
                data_t x_val = x_data[col];
                for (index_t j = col_start; j < col_end; j++)
                {
                    index_t row = idx[j];
                    y_data[row] -= values[j] * x_val;
                }
            }
        }
        else
        {
            printf("Section 4\n");
            for (index_t row = 0; row < num_rows; row++)
            {
                y_data[row] = beta * y_data[row];
            }

            for (index_t col = first_col; col < last_col; col++)
            {
                col_start = ptr[col];
                col_end = ptr[col + 1];
                data_t x_val = x_data[col];
                for (index_t j = col_start; j < col_end; j++)
                {
                    index_t row = idx[j];
                    y_data[row] -= values[j] * x_val;
                }
            }
        }
    }
    else if (alpha_zero)
    {
        if (beta_zero)
        {
            printf("Section 5\n");
            for (index_t row = 0; row < num_rows; row++)
            {
                y_data[row] = 0.0;
            }
        }
        if (!beta_one)
        {
            printf("Section 6\n");
            for (index_t row = 0; row < num_rows; row++)
            {
                y_data[row] = beta * y_data[row];
            }
        }
    }
    else
    {
        if (beta_one)
        {
            printf("Section 7\n");
            for (index_t col = first_col; col < last_col; col++)
            {
                col_start = ptr[col];
                col_end = ptr[col + 1];
                data_t x_val = x_data[col];
                for (index_t j = col_start; j < col_end; j++)
                {
                    index_t row = idx[j];
                    y_data[row] += alpha * values[j] * x_val;
                }
            }
        }
        else
        {
            printf("Section 8\n");
            for (index_t row = 0; row < num_rows; row++)
            {
                y_data[row] = beta * y_data[row];
            }

            for (index_t col = first_col; col < last_col; col++)
            {
                col_start = ptr[col];
                col_end = ptr[col + 1];
                data_t x_val = x_data[col];
                for (index_t j = col_start; j < col_end; j++)
                {
                    index_t row = idx[j];
                    y_data[row] += alpha * values[j] * x_val;
                }
            }
        }
    }
}

//CSR SpMV
void sequentialSPMV_raptor(Matrix<1>* A, Vector* x, Vector* y, double alpha, double beta, index_t _first_col = -1, index_t _size = 0)
{
    //TODO -- should be std::numeric_limits<data_t>::epsilon ...
    data_t zero_tol = DBL_EPSILON;

    index_t alpha_zero = (fabs(alpha) < zero_tol);
    index_t alpha_one = (fabs(alpha - 1.0) < zero_tol);
    index_t alpha_neg_one = (fabs(alpha + 1.0) < zero_tol);

    index_t beta_zero = (fabs(beta) < zero_tol);
    index_t beta_one = (fabs(beta - 1.0) < zero_tol);
    index_t beta_neg_one = (fabs(beta + 1.0) < zero_tol);

    index_t* ptr = (A->m)->outerIndexPtr();
    index_t* idx = (A->m)->innerIndexPtr();
    data_t* values = (A->m)->valuePtr();

    index_t num_rows = (A->m)->outerSize();

    data_t* y_data = y->data();
    data_t* x_data = x->data();

    index_t row_start;
    index_t row_end;

    if (alpha_one)
    {
        if (beta_one)
        {
            //Ax + y
            for (index_t i = 0; i < num_rows; i++)
            {
                row_start = ptr[i];
                row_end = ptr[i+1];

                for (index_t j = row_start; j < row_end; j++)
                {
                    index_t col = idx[j];
                    y_data[i] += values[j] * x_data[col];
                }
            }
        }
        else if (beta_zero)
        {
            //Ax
            for (index_t i = 0; i < num_rows; i++)
            {
                row_start = ptr[i];
                row_end = ptr[i+1];

                y_data[i] = values[row_start] * x_data[idx[row_start]];

                for (index_t j = row_start + 1; j < row_end; j++)
                {
                    index_t col = idx[j];
                    y_data[i] += values[j] * x_data[col];
                }
            }
        }
        else
        {
            //Ax + beta * y
            for (index_t i = 0; i < num_rows; i++)
            {
                row_start = ptr[i];
                row_end = ptr[i+1];

                y_data[i] *= beta;

                for (index_t j = row_start + 1; j < row_end; j++)
                {
                    index_t col = idx[j];
                    y_data[i] += values[j] * x_data[col];
                }
            }
        }
    }
    else if (alpha_neg_one)
    {
        if (beta_one)
        {
            //-Ax + y
            for (index_t i = 0; i < num_rows; i++)
            {
                row_start = ptr[i];
                row_end = ptr[i+1];

                for (index_t j = row_start; j < row_end; j++)
                {
                    index_t col = idx[j];
                    y_data[i] -= values[j] * x_data[col];
                }
            }
        }
        else if (beta_zero)
        {
            //Ax
            for (index_t i = 0; i < num_rows; i++)
            {
                row_start = ptr[i];
                row_end = ptr[i+1];

                y_data[i] = - (values[row_start] * x_data[idx[row_start]]);

                for (index_t j = row_start + 1; j < row_end; j++)
                {
                    index_t col = idx[j];
                    y_data[i] -= values[j] * x_data[col];
                }
            }
        }
        else
        {
            //Ax + beta * y
            for (index_t i = 0; i < num_rows; i++)
            {
                row_start = ptr[i];
                row_end = ptr[i+1];

                y_data[i] *= beta;

                for (index_t j = row_start + 1; j < row_end; j++)
                {
                    index_t col = idx[j];
                    y_data[i] -= values[j] * x_data[col];
                }
            }
        }
    }
    else if (alpha_zero)
    {
        if (beta_zero)
        {
            //return 0
            for (index_t i = 0; i < num_rows; i++)
            {
                y_data[i] = 0.0;
            }
        }
        else if (!beta_one)
        {
            //beta * y
            for (index_t i = 0; i < num_rows; i++)
            {
                y_data[i] *= beta;
            }
        }
    }
    else
    {
        if (beta_one)
        {
            //alpha*Ax + y
            for (index_t i = 0; i < num_rows; i++)
            {
                row_start = ptr[i];
                row_end = ptr[i+1];

                for (index_t j = row_start; j < row_end; j++)
                {
                    index_t col = idx[j];
                    y_data[i] += alpha * values[j] * x_data[col];
                }
            }
        }
        else if (beta_zero)
        {
            //alpha*Ax
            for (index_t i = 0; i < num_rows; i++)
            {
                row_start = ptr[i];
                row_end = ptr[i+1];

                y_data[i] = alpha * values[row_start] * x_data[idx[row_start]];

                for (index_t j = row_start + 1; j < row_end; j++)
                {
                    index_t col = idx[j];
                    y_data[i] += alpha * values[j] * x_data[col];
                }
            }
        }
        else
        {
            //alpha * Ax + beta * y
            for (index_t i = 0; i < num_rows; i++)
            {
                row_start = ptr[i];
                row_end = ptr[i+1];

                y_data[i] *= beta;

                for (index_t j = row_start + 1; j < row_end; j++)
                {
                    index_t col = idx[j];
                    y_data[i] += alpha * values[j] * x_data[col];
                }
            }
        }
    }
}

void parallel_spmv(ParMatrix* A, ParVector* x, ParVector* y, data_t alpha, data_t beta, index_t async = 0)
{
    // Get MPI Information
    index_t rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	// TODO must enforce that y and x are not aliased, or this will NOT work
	//    or we could use blocking sends as long as we post the iRecvs first

    // Declare communication variables
	MPI_Request*                            send_requests;
    MPI_Request*                            recv_requests;
    data_t*                                 send_buffer;
    data_t*                                 recv_buffer;
    std::map<index_t, index_t>              recv_proc_starts;
    data_t*                                 local_data;
    Vector                                  offd_tmp;
    index_t                                 tmp_size;
    index_t                                 begin;
    index_t                                 ctr;
    index_t                                 request_ctr;
    ParComm*                                comm;
    std::vector<index_t>                    send_procs;
    std::vector<index_t>                    recv_procs;
    std::map<index_t, std::vector<index_t>> send_indices;
    std::map<index_t, std::vector<index_t>> recv_indices;

    // Initialize communication variables
    comm          = A->comm;
    send_procs    = comm->send_procs;
    recv_procs    = comm->recv_procs;
    send_indices  = comm->send_indices;
    recv_indices  = comm->recv_indices;
    tmp_size      = comm->size_recvs;
    local_data    = x->local->data();

    if (recv_procs.size())
    {

	    recv_requests = new MPI_Request [recv_procs.size()];
        recv_buffer = new data_t [comm->size_recvs];

        // Send and receive vector data
	    // Begin sending and gathering off-diagonal entries
        begin = 0;
        ctr = 0;
        request_ctr = 0;
        for (auto proc : recv_procs)
        {
            index_t num_recv = recv_indices[proc].size();
            recv_proc_starts[proc] = begin;
            MPI_Irecv(&recv_buffer[begin], num_recv, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, &(recv_requests[request_ctr++]));
            begin += num_recv;
        }
    
    }

    if (send_procs.size())
    {
	    // TODO we do not want to malloc these every time
	    send_requests = new MPI_Request [send_procs.size()];
        send_buffer = new data_t [comm->size_sends];

        begin = 0;
        request_ctr = 0;
	    for (auto proc : send_procs)
        {
            ctr = 0;
            for (auto send_idx : send_indices[proc])
            {
                send_buffer[begin + ctr] = local_data[send_idx];
                ctr++;
            }
            MPI_Isend(&send_buffer[begin], ctr, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, &(send_requests[request_ctr++]));
            begin += ctr;
        }
    }


	// Compute partial SpMV with local information
	sequentialSPMV_raptor(A->diag, x->local, y->local, alpha, beta);

    // Once data is available, add contribution of off-diagonals
	// TODO Deal with new entries as they become available
	// TODO Add an error check on the status
    // TODO Using MPI_STATUS_IGNORE (delete[] recvStatus was causing a segfault)
    if (recv_procs.size())
    {
        if (async)
        {
            for (index_t i = 0; i < recv_procs.size(); i++)
            {
                index_t recv_idx = 0;
                MPI_Waitany(recv_procs.size(), recv_requests, &recv_idx, MPI_STATUS_IGNORE);
                index_t proc = recv_procs[recv_idx];
                auto minmax = std::minmax_element(recv_indices[proc].begin(), recv_indices[proc].end());
                index_t first_idx = minmax.first - recv_indices[proc].begin();
                index_t last_idx = minmax.second - recv_indices[proc].begin();

                index_t first_col = recv_indices[proc][first_idx];
                index_t last_col = recv_indices[proc][last_idx];
                
                Vector offd_tmp = Eigen::Map<Vector>(&recv_buffer[recv_proc_starts[proc]], recv_indices[proc].size());
                sequentialSPMV_raptor(A->offd, &offd_tmp, y->local, alpha, 1.0, A->local_to_global, first_col, last_col - first_col+1);
            }
        }
        else
        {
            // Wait for all receives to finish
	        MPI_Waitall(recv_procs.size(), recv_requests, MPI_STATUS_IGNORE);

            // Add received data to Vector
            Vector offd_tmp = Eigen::Map<Vector>(recv_buffer, tmp_size);
            sequentialSPMV_raptor(A->offd, &offd_tmp, y->local, alpha, 1.0, A->local_to_global); 
        }
    	delete[] recv_requests; 
        delete[] recv_buffer;
    }

    if (send_procs.size())
    {
	    // Wait for all sends to finish
	    // TODO Add an error check on the status
	    MPI_Waitall(send_procs.size(), send_requests, MPI_STATUS_IGNORE);

        // Delete MPI_Requests
	    delete[] send_requests; 
        delete[] send_buffer;
    }
}

// void sequentialSPMV(Matrix* A, Vector* x, Vector* y, double alpha, double beta)
// {
//     *y = alpha*((*(A->m))*(*x)) + beta * (*y);
// }

#endif
