#include "spmv.hpp"
/**************************************************************
 *****   Sequential Matrix-Vector Multiplication
 **************************************************************
 ***** Performs matrix-vector multiplication on inner indices
 ***** y[inner] = alpha * A[inner, outer] * x[outer] + beta*y[inner]
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled
 ***** x : Vector*
 *****    Vector to be multiplied
 ***** y : Vector*
 *****    Vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy A*x by
 ***** beta : data_t
 *****    Scalar to multiply original y by
 **************************************************************/
void seq_inner_spmv(Matrix* A, const data_t* x, data_t* y, const data_t alpha, const data_t beta, data_t* result = NULL, int outer_start = 0, int n_outer = -1)
{
    index_t* ptr = A->indptr.data();
    index_t* idx = A->indices.data();
    data_t* values = A->data.data();

    index_t n_inner = A->n_inner;
    if (n_outer < 0)
    {
        n_outer = A->n_outer;
    }
    index_t outer_end = outer_start + n_outer;

    index_t ptr_start;
    index_t ptr_end;

    if (result == NULL)
    {
        result = y;
    }

    if ((fabs(alpha - 1.0) < zero_tol))
    {
        if ((fabs(beta - 1.0) >= zero_tol))
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                result[inner] = beta * y[inner];
            }
        }
        for (index_t outer = outer_start; outer < outer_end; outer++)
        {
            ptr_start = ptr[outer];
            ptr_end = ptr[outer + 1];
            data_t x_val = x[outer];
            for (index_t j = ptr_start; j < ptr_end; j++)
            {
                index_t inner = idx[j];
                result[inner] += values[j] * x_val;
            }
        }
    }
    else if ((fabs(alpha + 1.0) < zero_tol))
    {
        if ((fabs(beta - 1.0) >= zero_tol))
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                result[inner] = beta * y[inner];
            }
        }
        for (index_t outer = outer_start; outer < outer_end; outer++)
        {
            ptr_start = ptr[outer];
            ptr_end = ptr[outer + 1];
            data_t x_val = x[outer];
            for (index_t j = ptr_start; j < ptr_end; j++)
            {
                index_t inner = idx[j];
                result[inner] -= values[j] * x_val;
            }
        }
    }
    else if ((fabs(alpha) < zero_tol))
    {
        if ((fabs(beta) < zero_tol))
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                result[inner] = 0.0;
            }
        }
        else if ((fabs(beta - 1.0) >= zero_tol))
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                result[inner] = beta * y[inner];
            }
        }
    }
    else
    {
        if ((fabs(beta - 1.0) >= zero_tol))
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                result[inner] = beta * y[inner];
            }
        }
        for (index_t outer = outer_start; outer < outer_end; outer++)
        {
            ptr_start = ptr[outer];
            ptr_end = ptr[outer + 1];
            data_t x_val = x[outer];
            for (index_t j = ptr_start; j < ptr_end; j++)
            {
                index_t inner = idx[j];
                result[inner] += alpha * values[j] * x_val;
            }
        }
    }
}

/**************************************************************
 *****   Sequential Matrix-Vector Multiplication
 **************************************************************
 ***** Performs matrix-vector multiplication on outer indices
 ***** y[outer] = alpha * A[outer, inner] * x[inner] + beta*y[outer]
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled
 ***** x : Vector*
 *****    Vector to be multiplied
 ***** y : Vector*
 *****    Vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy A*x by
 ***** beta : data_t
 *****    Scalar to multiply original y by
 **************************************************************/
void seq_outer_spmv(Matrix* A, const data_t* x, data_t* y, const data_t alpha, const data_t beta, data_t* result = NULL, int outer_start = 0, int n_outer = -1)
{
    index_t alpha_zero = (fabs(alpha) < zero_tol);
    index_t alpha_one = (fabs(alpha - 1.0) < zero_tol);
    index_t alpha_neg_one = (fabs(alpha + 1.0) < zero_tol);

    index_t beta_zero = (fabs(beta) < zero_tol);
    index_t beta_one = (fabs(beta - 1.0) < zero_tol);

    index_t* ptr = A->indptr.data();
    index_t* idx = A->indices.data();
    data_t* values = A->data.data();

    if (n_outer <= 0)
    {
        n_outer = A->n_outer;
    }
    index_t outer_end = outer_start + n_outer;

    index_t ptr_start;
    index_t ptr_end;

    if (result == NULL)
    {
        result = y;
    }
    else
    {
        for (int i = 0; i < n_outer; i++)
        {
            result[i] = 0.0;
        }
    }

    if (alpha_one)
    {
        if (beta_one)
        {
            //Ax + y
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                for (index_t j = ptr_start; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    result[outer] += values[j] * x[inner];
                }
            }
        }
        else if (beta_zero)
        {
            //Ax
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                if (ptr_start < ptr_end)
                {
                    result[outer] = values[ptr_start] * x[idx[ptr_start]];
                }
                else
                {
                    result[outer] = 0.0;
                }

                for (index_t j = ptr_start + 1; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    result[outer] += values[j] * x[inner];

                }
            }
        }
        else
        {
            //Ax + beta * y
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                result[outer] *= beta;

                for (index_t j = ptr_start + 1; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    result[outer] += values[j] * x[inner];
                }
            }
        }
    }
    else if (alpha_neg_one)
    {
        if (beta_one)
        {
            //-Ax + y
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                for (index_t j = ptr_start; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    result[outer] -= values[j] * x[inner];
                }
            }
        }
        else if (beta_zero)
        {
            //Ax
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                if (ptr_start < ptr_end)
                {
                    result[outer] = - (values[ptr_start] * x[idx[ptr_start]]);
                }
                else
                {
                    result[outer] = 0.0;
                }

                for (index_t j = ptr_start + 1; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    result[outer] -= values[j] * x[inner];
                }
            }
        }
        else
        {
            //Ax + beta * y
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                result[outer] *= beta;

                for (index_t j = ptr_start + 1; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    result[outer] -= values[j] * x[inner];
                }
            }
        }
    }
    else if (alpha_zero)
    {
        if (beta_zero)
        {
            //return 0
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                result[outer] = 0.0;
            }
        }
        else if (!beta_one)
        {
            //beta * y
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                result[outer] *= beta;
            }
        }
    }
    else
    {
        if (beta_one)
        {
            //alpha*Ax + y
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                for (index_t j = ptr_start; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    result[outer] += alpha * values[j] * x[inner];
                }
            }
        }
        else if (beta_zero)
        {
            //alpha*Ax
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                if (ptr_start < ptr_end)
                {
                    result[outer] = alpha * values[ptr_start] * x[idx[ptr_start]];
                }
                else
                {
                    result[outer] = 0.0;
                }

                for (index_t j = ptr_start + 1; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    result[outer] += alpha * values[j] * x[inner];
                }
            }
        }
        else
        {
            //alpha * Ax + beta * y
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                result[outer] *= beta;

                for (index_t j = ptr_start + 1; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    result[outer] += alpha * values[j] * x[inner];
                }
            }
        }
    }
}

/**************************************************************
 *****   Sequential Matrix-Vector Multiplication
 **************************************************************
 ***** Performs partial matrix-vector multiplication, calling
 ***** method appropriate for matrix format
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled
 ***** x : Vector*
 *****    Vector to be multiplied
 ***** y : Vector*
 *****    Vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy A*x by
 ***** beta : data_t
 *****    Scalar to multiply original y by
 **************************************************************/
void sequential_spmv(Matrix* A, const data_t* x, data_t* y, const data_t alpha,
    const data_t beta, data_t* result, int outer_start, int outer_end)
{
    if (A->format == CSR)
    {
        seq_outer_spmv(A, x, y, alpha, beta, result, outer_start, outer_end);
    }
    else
    {
        seq_inner_spmv(A, x, y, alpha, beta, result, outer_start, outer_end);
    }   
}

/**************************************************************
 *****   Sequential Transpose Matrix-Vector Multiplication
 **************************************************************
 ***** Performs partial transpose matrix-vector multiplication, 
 ***** calling method appropriate for matrix format
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled
 ***** x : Vector*
 *****    Vector to be multiplied
 ***** y : Vector*
 *****    Vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy A*x by
 ***** beta : data_t
 *****    Scalar to multiply original y by
 **************************************************************/
void sequential_spmv_T(Matrix* A, const data_t* x, data_t* y, const data_t alpha, const data_t beta, data_t* result, int outer_start, int outer_end)
{
    if (A->format == CSR)
    {
        seq_inner_spmv(A, x, y, alpha, beta, result, outer_start, outer_end);
    }
    else
    {
        seq_outer_spmv(A, x, y, alpha, beta, result, outer_start, outer_end);
    }   
}

/**************************************************************
 *****   Parallel Matrix-Vector Multiplication
 **************************************************************
 ***** Performs parallel matrix-vector multiplication
 ***** y = alpha*A*x + beta*y
 *****
 ***** Parameters
 ***** -------------
 ***** A : ParMatrix*
 *****    Parallel matrix to be multipled
 ***** x : ParVector*
 *****    Parallel vector to be multiplied
 ***** y : ParVector*
 *****    Parallel vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy alpha*A*x
 ***** beta : data_t
 *****    Scalar to multiply original y by
 ***** async : index_t
 *****    Boolean flag for updating SpMV asynchronously
 **************************************************************/
void parallel_spmv(const ParMatrix* A, const ParVector* x, ParVector* y, const data_t alpha, const data_t beta, const int async, ParVector* result)
{
    data_t* result_data = NULL;
    if (result != NULL)
    {
        result_data = result->local->data();
    }

    // Get MPI Information
    MPI_Comm comm_mat = A->comm_mat;

    // TODO must enforce that y and x are not aliased, or this will NOT work
    //    or we could use blocking sends as long as we post the iRecvs first

    // Declare communication variables
    MPI_Request*                             send_requests;
    MPI_Request*                             recv_requests;
    data_t*                                  send_buffer;
    data_t*                                  recv_buffer;
    data_t*                                  x_data;
    data_t*                                  y_data;
    index_t                                  begin;
    index_t                                  ctr;
    index_t                                  request_ctr;
    index_t                                  tag;
    index_t                                  num_sends;
    index_t                                  num_recvs;
    index_t                                  size_sends;
    index_t                                  size_recvs;
    index_t*                                 send_procs;
    index_t*                                 send_row_starts;
    index_t*                                 send_row_indices;
    index_t*                                 recv_procs;
    index_t*                                 recv_col_starts;
    ParComm*                                 comm;

    // Initialize communication variables
    comm          = A->comm;
    num_sends = comm->num_sends;
    num_recvs = comm->num_recvs;
    size_sends = comm->size_sends;
    size_recvs = comm->size_recvs;

    if (num_sends)
    {
        send_procs    = comm->send_procs.data();
        send_row_starts = comm->send_row_starts.data();
        send_row_indices = comm->send_row_indices.data();
    }

    if (num_recvs)
    {
        recv_procs    = comm->recv_procs.data();
        recv_col_starts = comm->recv_col_starts.data();
    }

    if (x->local_n)
    {
        x_data = x->local->data();
    }

    if (y->local_n)
    {
        y_data = y->local->data();
    }


    // If receive values, post appropriate MPI Receives
    if (num_recvs)
    {
        // Initialize recv requests and buffer
        recv_requests = comm->recv_requests;
        recv_buffer = comm->recv_buffer;

        // Post receives for x-values that are needed
        index_t proc, num_recv, recv_start, recv_end;
        for (index_t i = 0; i < num_recvs; i++)
        {
            proc = recv_procs[i];
            recv_start = recv_col_starts[i];
            recv_end = recv_col_starts[i+1];
            num_recv = recv_end - recv_start;
            MPI_Irecv(&recv_buffer[recv_start], num_recv, MPI_DATA_T, proc, 0, comm_mat, &(recv_requests[i]));
        }
    }

    // Send values of x to appropriate processors
    if (num_sends)
    {
        int send_start, send_end, send_size;

        // TODO we do not want to malloc these every time
        send_requests = comm->send_requests;
        send_buffer = comm->send_buffer;

        for (index_t i = 0; i < num_sends; i++)
        {
            index_t proc = send_procs[i];
            send_start = send_row_starts[i];
            send_end = send_row_starts[i+1];
            send_size = send_end - send_start;

            for (int j = send_start; j < send_end; j++)
            {
                send_buffer[j] = x_data[send_row_indices[j]];

            }
            MPI_Isend(&send_buffer[send_start], send_size, MPI_DATA_T, proc, 0, comm_mat, &(send_requests[i]));
        }
    }

    if (A->local_rows)
    {
        // Compute partial SpMV with local information
        sequential_spmv(A->diag, x_data, y_data, alpha, beta, result_data);
    }

    // Once data is available, add contribution of off-diagonals
    // TODO Deal with new entries as they become available
    // TODO Add an error check on the status
    if (num_recvs)
    {
        if (async)
        {
            int recv_start, recv_end, num_cols;
            int ctr = 0;
            int finished = 0;
            while (ctr < num_recvs)
            {
                for (int i = 0; i < num_recvs; i++)
                {
                    if (recv_requests[i] == MPI_REQUEST_NULL) continue;
                    MPI_Test(&recv_requests[i], &finished, MPI_STATUS_IGNORE); 
                    if (finished)
                    {
                        recv_start = recv_col_starts[i];
                        recv_end = recv_col_starts[i+1];
                        num_cols = recv_end - recv_start;
                        sequential_spmv(A->offd, recv_buffer, y_data, alpha, 1.0, result_data, recv_start, num_cols);
                        ctr++;
                    }
                }
            }
        }
        // TODO Add an error check on the status
        else
        {
            // Wait for all receives to finish
            MPI_Waitall(num_recvs, recv_requests, MPI_STATUS_IGNORE);

            // Add received data to Vector
            sequential_spmv(A->offd, recv_buffer, y_data, alpha, 1.0, result_data); 
        }
    }

    // TODO Add an error check on the status
    if (num_sends)
    {
        // Wait for all sends to finish
        MPI_Waitall(num_sends, send_requests, MPI_STATUS_IGNORE);
    }
}

/**************************************************************
 *****   Parallel Transpose Matrix-Vector Multiplication
 **************************************************************
 ***** Performs parallel transpose matrix-vector multiplication
 ***** y = alpha*A^T*x + beta*y
 *****
 ***** Parameters
 ***** -------------
 ***** A : ParMatrix*
 *****    Parallel matrix to be transposed and multipled
 ***** x : ParVector*
 *****    Parallel vector to be multiplied
 ***** y : ParVector*
 *****    Parallel vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy alpha*A*x
 ***** beta : data_t
 *****    Scalar to multiply original y by
 ***** async : index_t
 *****    Boolean flag for updating SpMV asynchronously
 **************************************************************/
void parallel_spmv_T(const ParMatrix* A, const ParVector* x, ParVector* y, const data_t alpha, const data_t beta, const int async, ParVector* result)
{
    data_t* result_data = NULL;
    if (result != NULL)
    {
        result_data = result->local->data();
    }

    // Get MPI Information
    MPI_Comm comm_mat = A->comm_mat;

    // TODO must enforce that y and x are not aliased, or this will NOT work
    //    or we could use blocking sends as long as we post the iRecvs first

    // Declare communication variables
    MPI_Request*                             send_requests;
    MPI_Request*                             recv_requests;
    data_t*                                  send_buffer;
    data_t*                                  recv_buffer;
    data_t*                                  x_data;
    data_t*                                  y_data;
    index_t                                  begin;
    index_t                                  ctr;
    index_t                                  request_ctr;
    index_t                                  tag;
    index_t                                  num_sends;
    index_t                                  num_recvs;
    index_t                                  size_sends;
    index_t                                  size_recvs;
    index_t*                                 send_procs;
    index_t*                                 send_row_starts;
    index_t*                                 send_row_indices;
    index_t*                                 recv_procs;
    index_t*                                 recv_col_starts;
    index_t*                                 recv_col_indices;
    ParComm*                                 comm;

    // Initialize communication variables
    comm          = A->comm;
    num_sends = comm->num_recvs;
    num_recvs = comm->num_sends;
    size_sends = comm->size_recvs;
    size_recvs = comm->size_sends;

    if (num_sends)
    {
        send_procs    = comm->recv_procs.data();
        send_row_starts = comm->recv_col_starts.data();
    }

    if (num_recvs)
    {
        recv_procs    = comm->send_procs.data();
        recv_col_starts = comm->send_row_starts.data();
        recv_col_indices = comm->send_row_indices.data();
    }

    if (x->local_n)
    {
        x_data = x->local->data();
    }

    if (y->local_n)
    {
        y_data = y->local->data();
    }

    if (num_recvs)
    {
        recv_requests = comm->send_requests;
        recv_buffer = comm->send_buffer;

        // Send and receive vector data
        // Begin sending and gathering off-diagonal entries
        index_t proc, num_recv, recv_start, recv_end;
        for (index_t i = 0; i < num_recvs; i++)
        {
            proc = recv_procs[i];
            recv_start = recv_col_starts[i];
            recv_end = recv_col_starts[i+1];
            num_recv = recv_end - recv_start;
            MPI_Irecv(&recv_buffer[recv_start], num_recv, MPI_DATA_T, proc, 0, comm_mat, &(recv_requests[i]));
        }
    }

    // Perform local spmv -- into send_buffer
    // Send this result
    if (num_sends)
    {
        send_requests = comm->recv_requests;
        send_buffer = comm->recv_buffer;

        index_t proc, num_send, send_start, send_end;
        if (async)
        {
            for (int i = 0; i < num_sends; i++)
            {
                proc = send_procs[i];
                send_start = send_row_starts[i];
                send_end = send_row_starts[i+1];
                num_send = send_end - send_start;
                sequential_spmv_T(A->offd, x_data, send_buffer, alpha, 0.0, result_data, send_start, num_send);
                MPI_Isend(&send_buffer[send_start], num_send, MPI_DATA_T, proc, 0, comm_mat, &(send_requests[i]));
            }
        }
        else
        { 
            sequential_spmv_T(A->offd, x_data, send_buffer, alpha, 0.0, result_data);
            for (int i = 0; i < num_sends; i++)
            {
                proc = send_procs[i];
                send_start = send_row_starts[i];
                send_end = send_row_starts[i+1];
                num_send = send_end - send_start;
                MPI_Isend(&send_buffer[send_start], num_send, MPI_DATA_T, proc, 0, comm_mat, &(send_requests[i]));            
            }
        }
    }

    sequential_spmv_T(A->diag, x_data, y_data, alpha, beta, result_data);

    if (num_recvs)
    {
        int recv_idx, recv_start, recv_end, num_cols;
        if (async)
        {
            int ctr = 0;
            int finished = 0;
            while (ctr < num_recvs)
            {
                for (int i = 0; i < num_recvs; i++)
                {
                    if (recv_requests[i] == MPI_REQUEST_NULL) continue;
                    MPI_Test(&recv_requests[i], &finished, MPI_STATUS_IGNORE); 
                    if (finished)
                    {
                        recv_start = recv_col_starts[i];
                        recv_end = recv_col_starts[i+1];
                        for (int j = recv_start; j < recv_end; j++)
                        {
                            index_t row = recv_col_indices[j];
                            y_data[row] += recv_buffer[j];
                        }
                        ctr++;
                    }
                }
            }
        }
        else
        {
            // Wait for all recieves to finish
            MPI_Waitall(num_recvs, recv_requests, MPI_STATUS_IGNORE);

            for (int i = 0; i < num_recvs; i++)
            {
                recv_start = recv_col_starts[i];
                recv_end = recv_col_starts[i+1];
                for (int j = recv_start; j < recv_end; j++)
                {
                    index_t row = recv_col_indices[j];
                    y_data[row] += recv_buffer[j];
                }
            }
        }
    }

    if (num_sends)
    {
        // Wait for all sends to finish
        MPI_Waitall(num_sends, send_requests, MPI_STATUS_IGNORE);
    }

}
