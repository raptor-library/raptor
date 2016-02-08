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
    if (A->local_rows == 0) return;

    data_t* result_data = NULL;
    if (result != NULL)
    {
        result_data = result->local->data();
    }

    // Get MPI Information
    MPI_Comm comm_mat = A->comm_mat;
    int rank, num_procs;
    MPI_Comm_rank(comm_mat, &rank);
    MPI_Comm_size(comm_mat, &num_procs);

    // TODO must enforce that y and x are not aliased, or this will NOT work
    //    or we could use blocking sends as long as we post the iRecvs first

    // Declare communication variables
    MPI_Request*                             send_requests;
    MPI_Request*                             recv_requests;
    data_t*                                  send_buffer;
    data_t*                                  recv_buffer;
    data_t*                                  x_data;
    data_t*                                  y_data;
    Vector                                   offd_tmp;
    index_t                                  begin;
    index_t                                  ctr;
    index_t                                  request_ctr;
    index_t                                  tag;
    index_t                                  send_size;
    index_t                                  recv_size;
    index_t                                  num_sends;
    index_t                                  num_recvs;
    index_t*                                 send_procs;
    index_t*                                 send_row_starts;
    index_t*                                 send_row_indices;
    index_t*                                 recv_procs;
    index_t*                                 recv_col_starts;
//    index_t*                                 recv_col_indices;
    ParComm*                                 comm;

    // Initialize communication variables
    comm          = A->comm;
    num_sends = comm->send_procs.size();
    num_recvs = comm->recv_procs.size();

    if (num_sends)
    {
        send_procs    = comm->send_procs.data();
        send_row_starts = comm->send_row_starts.data();
        send_row_indices = comm->send_row_indices.data();
        send_size = send_row_starts[num_sends];
    }

    if (num_recvs)
    {
        recv_procs    = comm->recv_procs.data();
        recv_col_starts = comm->recv_col_starts.data();
//        recv_col_indices = comm->recv_col_indices.data();
        recv_size = recv_col_starts[num_recvs];
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
        recv_requests = new MPI_Request [num_recvs];
        for (index_t i = 0; i < num_recvs; i++)
        {
            recv_requests[i] = MPI_REQUEST_NULL;
        }
        recv_buffer = new data_t [recv_size];

        // Post receives for x-values that are needed
        begin = 0;
        ctr = 0;
        request_ctr = 0;

        index_t proc, num_recv, recv_start, recv_end;
        for (index_t i = 0; i < num_recvs; i++)
        {
            proc = recv_procs[i];
            recv_start = recv_col_starts[i];
            recv_end = recv_col_starts[i+1];
            num_recv = recv_end - recv_start;
            MPI_Irecv(&recv_buffer[begin], num_recv, MPI_DATA_T, proc, 0, comm_mat, &(recv_requests[request_ctr++]));
            begin += num_recv;
        }
    
    }

    // Send values of x to appropriate processors
    if (num_sends)
    {
        int send_start, send_end, send_size;

        // TODO we do not want to malloc these every time
        send_requests = new MPI_Request [num_sends];
        for (index_t i = 0; i < num_sends; i++)
        {
            send_requests[i] = MPI_REQUEST_NULL;
        }
        send_buffer = new data_t [send_size];

        begin = 0;
        request_ctr = 0;
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
            MPI_Isend(&send_buffer[begin], send_size, MPI_DATA_T, proc, 0, comm_mat, &(send_requests[request_ctr++]));
            begin += send_size;
        }
    }

    // Compute partial SpMV with local information
    sequential_spmv(A->diag, x_data, y_data, alpha, beta, result_data);

    // Once data is available, add contribution of off-diagonals
    // TODO Deal with new entries as they become available
    // TODO Add an error check on the status
    if (num_recvs)
    {
        if (async)
        {
            int recv_idx[num_recvs];
            index_t proc, recv_start, recv_end, first_col, num_cols;
            for (index_t i = 0; i < num_recvs;)
            {
                int n_recvd;
                MPI_Waitsome(num_recvs, recv_requests, &n_recvd, recv_idx, MPI_STATUS_IGNORE);
                for (index_t j = 0; j < n_recvd; j++)
                {
                    //index_t proc = recv_procs[recv_idx[j]];
                    recv_start = recv_col_starts[recv_idx[j]];
                    recv_end = recv_col_starts[recv_idx[j]+1];
                    //first_col = recv_col_indices[recv_start]; -- first col == recv_start
                    num_cols = recv_end - recv_start;
                    sequential_spmv(A->offd, recv_buffer, y_data, alpha, 1.0, result_data, recv_start, num_cols);
                }
                i += n_recvd;
            }
        }
        else
        {
            // Wait for all receives to finish
            MPI_Waitall(num_recvs, recv_requests, MPI_STATUS_IGNORE);

            // Add received data to Vector
            sequential_spmv(A->offd, recv_buffer, y_data, alpha, 1.0, result_data); 

        }

    	delete[] recv_requests; 
        delete[] recv_buffer;
    }

    if (num_sends)
    {
        // Wait for all sends to finish
        // TODO Add an error check on the status
        MPI_Waitall(num_sends, send_requests, MPI_STATUS_IGNORE);

        // Delete MPI_Requests
        delete[] send_requests; 
        delete[] send_buffer;
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
/*    if (A->local_rows == 0) return;

    data_t* result_data = NULL;
    if (result != NULL)
    {
        result_data = result->local->data();
    }

    // Declare communication variables
	MPI_Request*                            send_requests;
    MPI_Request*                            recv_requests;
    data_t*                                 send_buffer;
    data_t*                                 recv_buffer;
    data_t*                                  x_data;
    data_t*                                  y_data;
    std::map<index_t, index_t>              recv_proc_starts;
    data_t*                                 offd_tmp;
    index_t                                 tmp_size;
    index_t                                 begin;
    index_t                                 ctr;
    index_t                                 request_ctr;
    index_t                                 msg_tag;
    index_t                                 size_sends;
    index_t                                 size_recvs;
    index_t                                 num_sends;
    index_t                                 num_recvs;
    ParComm*                                comm;
    std::vector<index_t>                    send_procs;
    std::vector<index_t>                    recv_procs;
    std::map<index_t, std::vector<index_t>> send_indices;
    std::map<index_t, std::vector<index_t>> recv_indices;
    MPI_Comm                               comm_mat;
    
    msg_tag       = 1111;

    // Get MPI Information
    comm_mat = A->comm_mat;
    int rank, num_procs;
    MPI_Comm_rank(comm_mat, &rank);
    MPI_Comm_size(comm_mat, &num_procs);

    // Initialize communication variables
    comm          = A->comm;
    send_procs    = comm->recv_procs;
    recv_procs    = comm->send_procs;
    num_sends     = send_procs.size();
    num_recvs     = recv_procs.size();

    if (num_sends)
    {
        send_indices  = comm->recv_indices;
        size_sends    = comm->size_recvs;
        tmp_size      = size_sends;
    }

    if (num_recvs)
    {
        recv_indices  = comm->send_indices;
        size_recvs    = comm->size_sends;
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
        recv_requests = new MPI_Request [num_recvs];
        recv_buffer = new data_t [size_recvs];

        // Send and receive vector data
	    // Begin sending and gathering off-diagonal entries
        begin = 0;
        request_ctr = 0;
        for (auto proc : recv_procs)
        {
            index_t num_recv = recv_indices[proc].size();
            recv_proc_starts[proc] = begin;
            MPI_Irecv(&recv_buffer[begin], num_recv, MPI_DATA_T, proc, msg_tag, comm_mat, &(recv_requests[request_ctr++]));
            begin += num_recv;
        }   
    }

    if (num_sends)
    {
        // TODO we do not want to malloc these every time
        send_requests = new MPI_Request [num_sends];
        send_buffer = new data_t [size_sends];
        offd_tmp = new data_t[tmp_size];

        begin = 0;
        request_ctr = 0;
        ctr = 0;

        if (async)
        {
 //           for (auto proc : send_procs)
 //           {
 //               std::vector<index_t> tmp_indices = send_indices[proc];
 //               index_t tmp_idx_size = tmp_indices.size();
 //               offd_tmp->resize(tmp_idx_size);
 //               sequential_spmv_T(A->offd, x_data, offd_tmp, alpha, 0.0, send_indices[proc]);

 //               ctr = 0;
 //               for (index_t i = 0; i < tmp_idx_size; i++)
 //               {
 //                   send_buffer[begin + ctr] = offd_tmp[i];
 //                   ctr++;
 //               }
 //               MPI_Isend(&send_buffer[begin], ctr, MPI_DATA_T, proc, msg_tag, comm_mat, &(send_requests[request_ctr++]));
 //               begin += ctr;
 //           }
        }
        else
        {
            sequential_spmv_T(A->offd, x_data, offd_tmp, alpha, 0.0);

	        for (auto proc : send_procs)
            {
                ctr = 0;
                std::vector<index_t>& s_indices = send_indices[proc];
                for (auto send_idx : s_indices)
                {
                    send_buffer[begin + ctr] = offd_tmp[send_idx];
                    ctr++;
                }
                MPI_Isend(&send_buffer[begin], ctr, MPI_DATA_T, proc, msg_tag, comm_mat, &(send_requests[request_ctr++]));
                begin += ctr;
            }
        }
        delete offd_tmp;
    }

    sequential_spmv_T(A->diag, x_data, y_data, alpha, beta);

    if (num_recvs)
    {
        if (async)
        {
 //           for (index_t i = 0; i < num_recvs; i++)
 //           {
 //               index_t recv_idx = 0;
 //               MPI_Waitany(num_recvs, recv_requests, &recv_idx, MPI_STATUS_IGNORE);
 //               index_t proc = recv_procs[recv_idx];
 //               std::vector<index_t> tmp_indices = recv_indices[proc];
 //               index_t tmp_idx_size = tmp_indices.size();
 //               for (index_t j = 0; j < tmp_idx_size; j++)
 //               {
 //                   index_t row = tmp_indices[j];
 //                   y_data[row] += recv_buffer[recv_proc_starts[proc] + j];
 //               }
 //           }
        }
        else
        {
            // Wait for all receives to finish
            MPI_Waitall(num_recvs, recv_requests, MPI_STATUS_IGNORE);

            for (auto proc : recv_procs)
            {
                std::vector<index_t> tmp_indices = recv_indices[proc];
                index_t tmp_idx_size = tmp_indices.size();
                for (index_t j = 0; j < tmp_idx_size; j++)
                {
                    index_t row = tmp_indices[j];
                    y_data[row] += recv_buffer[recv_proc_starts[proc] + j];
                }
            }
        }

    	delete[] recv_requests; 
        delete[] recv_buffer;
    } 

    if (num_sends)
    {
	    // Wait for all sends to finish
	    // TODO Add an error check on the status
	    MPI_Waitall(num_sends, send_requests, MPI_STATUS_IGNORE);

        // Delete MPI_Requests
        delete[] send_requests; 
        delete[] send_buffer;
    } 
*/
}
