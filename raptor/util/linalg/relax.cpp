#include "relax.hpp"
#include "spmv.hpp"

void gs_diag(Matrix* A, const data_t* x, const data_t* y, data_t* result, data_t* diag_data)
{
    index_t* indptr = A->indptr.data();
    index_t* indices = A->indices.data();
    data_t* data = A->data.data();

    index_t num_rows = A->n_rows;
    index_t row_start, row_end;

    index_t col;
    data_t value;

    // Forward Sweep
    for (index_t row = 0; row < num_rows; row++)
    {
        row_start = indptr[row];
        row_end = indptr[row+1];

        for (index_t j = row_start; j < row_end; j++)
        {
            col = indices[j];
            value = data[j];
            if (col < row)
            {
                result[row] += value*result[col];
            }
            else if (col > row)
            {
                result[row] += value*x[col];
            }
            else if (col == row)
            {
                diag_data[row] = value;
            }
        }
        if (fabs(diag_data[row]) > zero_tol)
        {
            result[row] = (y[row] - result[row]) / diag_data[row];
        }
    }

    // Backward Sweep
    for (index_t row = num_rows-1; row >= 0; row--)
    {
        row_start = indptr[row];
        row_end = indptr[row+1];

        for (index_t j = row_start; j < row_end; j++)
        {
            col = indices[j];
            value = data[j];
            if (col > row)
            {
                result[row] += value*result[col];
            }
            else if (col < row)
            {
                result[row] += value*x[col];
            }
        }
        if (fabs(diag_data[row]) > zero_tol)
        {
            result[row] = (y[row] - result[row]) / diag_data[row];
        }
    }
}

void jacobi_diag(Matrix* A, const data_t* x, data_t* result, data_t* diag_data)
{
    index_t* indptr = A->indptr.data();
    index_t* indices = A->indices.data();
    data_t* data = A->data.data();

    index_t num_rows = A->n_rows;
    index_t row_start, row_end;

    index_t col;
    data_t value;

    for (index_t row = 0; row < num_rows; row++)
    {
        row_start = indptr[row];
        row_end = indptr[row+1];

        for (index_t j = row_start; j < row_end; j++)
        {
            col = indices[j];
            value = data[j];
            if (col == row)
            {
                diag_data[row] = value;
            }
            else
            {
                result[row] += value * x[col]; 
            }
        }
    }
}

void relax(const ParMatrix* A, ParVector* x, const ParVector* y, int num_sweeps, const int async)
{
    if (A->local_rows == 0) return;

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
    data_t*                                  local_data;
    Vector                                   offd_tmp;
    index_t                                  begin;
    index_t                                  ctr;
    index_t                                  request_ctr;
    index_t                                  size_sends;
    index_t                                  size_recvs;
    index_t                                  num_sends;
    index_t                                  num_recvs;
    index_t*                                 send_procs;
    index_t*                                 send_row_starts;
    index_t*                                 send_row_indices;
    index_t*                                 recv_procs;
    index_t*                                 recv_col_starts;
    ParComm*                                 comm;
    std::map<index_t, index_t>               recv_proc_starts;

    // Initialize communication variables
    comm          = A->comm;
    num_sends = comm->num_sends;
    num_recvs = comm->num_recvs;
    size_sends = comm->size_sends;
    size_recvs = comm->size_recvs;

    if (num_sends)
    {
        send_procs = comm->send_procs.data();
        send_row_starts = comm->send_row_starts.data();
        send_row_indices = comm->send_row_indices.data();
    }

    if (num_recvs)
    {
        recv_procs = comm->recv_procs.data();
        recv_col_starts = comm->recv_col_starts.data();
    }     

    local_data    = x->local->data();

    data_t* x_data = x->local->data();
    data_t* y_data = y->local->data();
    data_t* sum_data = new data_t[y->local_n]();
    data_t* x_new_data = new data_t[x->local_n]();
    data_t* diag_data = new data_t[A->diag->n_rows]();

    if (num_recvs)
    {
        // Initialize recv requests and buffer
        recv_requests = new MPI_Request [num_recvs];
        for (index_t i = 0; i < num_recvs; i++)
        {
            recv_requests[i] = MPI_REQUEST_NULL;
        }
        recv_buffer = new data_t [size_recvs];
    }

    if (num_sends)
    {
        // TODO we do not want to malloc these every time
        send_requests = new MPI_Request [num_sends];
        for (index_t i = 0; i < num_sends; i++)
        {
            send_requests[i] = MPI_REQUEST_NULL;
        }
        send_buffer = new data_t [size_sends];
    }

    for (int iter = 0; iter < num_sweeps; iter++)
    {
        // If receive values, post appropriate MPI Receives
        if (num_recvs)
        {
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
            int send_start, send_end, size_sends;
            begin = 0;
            request_ctr = 0;
            for (index_t i = 0; i < num_sends; i++)
            {
                index_t proc = send_procs[i];
                send_start = send_row_starts[i];
                send_end = send_row_starts[i+1];
                size_sends = send_end - send_start;

                for (int j = send_start; j < send_end; j++)
                {
                    send_buffer[j] = x_data[send_row_indices[j]];
                }
                MPI_Isend(&send_buffer[begin], size_sends, MPI_DATA_T, proc, 0, comm_mat, &(send_requests[request_ctr++]));
                begin += size_sends;
            }

        }

        // Compute partial relaxation with local information
        //jacobi_diag(A->diag, x_data, sum_data, diag_data);
        gs_diag(A->diag, x_data, y_data, x_new_data, diag_data);

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
                        //first_col = recv_col_indices[recv_start];
                        num_cols = recv_end - recv_start;
                        sequential_spmv(A->offd, x_data, sum_data, 1.0, 1.0, NULL, recv_start, num_cols);
                    }
                    i += n_recvd;
                }
            }
            else
            {
                // Wait for all receives to finish
                MPI_Waitall(num_recvs, recv_requests, MPI_STATUS_IGNORE);

                // Add received data to Vector
                sequential_spmv(A->offd, recv_buffer, sum_data, 1.0, 1.0); 

            }
        }

        index_t num_rows = A->local_rows;
        for (int i = 0; i < num_rows; i++)
        {
            if (fabs(diag_data[i]) > zero_tol)
            {
                x_data[i] = x_new_data[i] + (y_data[i] - sum_data[i]) / diag_data[i];
                sum_data[i] = 0;
                x_new_data[i] = 0;
            }
        }

        if (num_sends)
        {
            // Wait for all sends to finish
            // TODO Add an error check on the status
            MPI_Waitall(num_sends, send_requests, MPI_STATUS_IGNORE);
        }
    }

    if (num_recvs)
    {
        delete[] recv_requests; 
        delete[] recv_buffer;
    }
    if (num_sends)
    {
        // Delete MPI_Requests
        delete[] send_requests; 
        delete[] send_buffer;
    }
    delete[] diag_data;
    delete[] sum_data;
    delete[] x_new_data;
}


