// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "repartition.hpp"

	void make_contiguous(ParCSRMatrix* A, bool form_comm)
	{
	    int rank;
	    int num_procs;
	    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
	    RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);

	    int assumed_num_cols, assumed_first_col, assumed_last_col;
	    int local_assumed_num_cols;
	    int assumed_proc, orig_col, new_col;
	    int proc, proc_idx, idx, ctr;
	    int num_sends, n_sent, send_size;
	    int size_recvs;
	    int send_key, recv_key;
	    
	    int start, end;
	    int assumed_col, local_col;
	    int msg_avail, finished, count;
	    RAPtor_MPI_Request barrier_request;

	    aligned_vector<int> assumed_col_to_new;
	    aligned_vector<int> proc_num_cols(num_procs);
	    aligned_vector<int> send_procs;
	    aligned_vector<int> send_ptr;
	    aligned_vector<int> send_ctr;
	    aligned_vector<int> send_buffer;
	    aligned_vector<int> recv_buffer;
	    aligned_vector<MPI_Request> send_requests;

	    aligned_vector<MPI_Request> recv_requests;

	    std::map<int, int> global_to_local;
	    ctr = 0;
	    for (aligned_vector<int>::const_iterator it = A->off_proc_column_map.begin();  
		    it != A->off_proc_column_map.end(); ++it)
	    {
		global_to_local[*it] = ctr++;
	    }

	    // Find how many columns are local to each process
	    RAPtor_MPI_Allgather(&(A->on_proc_num_cols), 1, RAPtor_MPI_INT, proc_num_cols.data(), 1, RAPtor_MPI_INT,
		    RAPtor_MPI_COMM_WORLD);

	    // Determine the new first local row / first local col of rank
	    A->partition->first_local_col = 0;
	    for (int i = 0; i < rank; i++)
	    {
		A->partition->first_local_col += proc_num_cols[i];
	    }
	    A->partition->first_local_row = A->partition->first_local_col;

	    // Determine the global number of columns and rows
	    A->global_num_cols = A->partition->first_local_col;
	    for (int i = rank; i < num_procs; i++)
	    {
		A->global_num_cols += proc_num_cols[i];
	    }
	    A->global_num_rows = A->global_num_cols;

	    // Determine which columns are assumed local to rank
	    assumed_num_cols = ((A->global_num_cols - 1) / num_procs) + 1;
	    assumed_first_col = assumed_num_cols * rank;
	    assumed_last_col = assumed_first_col + assumed_num_cols;
	    if (assumed_first_col > A->global_num_cols)
		assumed_first_col = A->global_num_cols;
	    if (assumed_last_col > A->global_num_cols)
		assumed_last_col = A->global_num_cols;
	    local_assumed_num_cols = assumed_last_col - assumed_first_col;

	    if (local_assumed_num_cols)
	    {
		assumed_col_to_new.resize(local_assumed_num_cols);
	    }

	    // Determine number of local cols assumed to be on each distant proc
	    for (int i = 0; i < num_procs; i++)
	    {
		proc_num_cols[i] = 0;
	    }
	    for (int i = 0; i < A->on_proc_num_cols; i++)
	    {
		orig_col = A->on_proc_column_map[i];
		assumed_proc = orig_col / assumed_num_cols;
		proc_num_cols[assumed_proc]++;
	    }

	    // Re-arrange on_proc cols, ordered by assumed proc
	    send_size = 0;
	    send_ptr.emplace_back(send_size);
	    for (int i = 0; i < num_procs; i++)
	    {
		if (proc_num_cols[i])
		{
		    send_size += proc_num_cols[i];
		    proc_num_cols[i] = send_procs.size();
		    send_procs.emplace_back(i);
		    send_ptr.emplace_back(send_size);
		}
	    }
	    num_sends = send_procs.size();
	    if (send_size)
	    {
		send_ctr.resize(num_sends, 0);
		send_buffer.resize(2*send_size);
		send_requests.resize(num_sends);
		for (int i = 0; i < A->on_proc_num_cols; i++)
		{
		    orig_col = A->on_proc_column_map[i];
		    new_col = A->partition->first_local_col + i;
		    assumed_proc = orig_col / assumed_num_cols;
		    proc_idx = proc_num_cols[assumed_proc];
		    idx = send_ptr[proc_idx] + send_ctr[proc_idx]++;
		    send_buffer[2*idx] = orig_col;
		    send_buffer[2*idx+1] = new_col;
		}
	    }

	    n_sent = 0;
	    size_recvs = 0;
	    send_key = 7568;
	    for (int i = 0; i < num_sends; i++)
	    {
		proc = send_procs[i];
		start = 2*send_ptr[i];
		end = 2*send_ptr[i+1];
		if (proc != rank)
		{
		    RAPtor_MPI_Issend(&(send_buffer[start]), (end - start), RAPtor_MPI_INT, proc, 
			    send_key, RAPtor_MPI_COMM_WORLD, &(send_requests[n_sent++]));
		}
		else
		{
		    for (int j = start; j < end; j+=2)
		    {
			orig_col = send_buffer[j];
			new_col = send_buffer[j+1];
			local_col = orig_col - assumed_first_col;
			assumed_col_to_new[local_col] = new_col;
		    }
		    size_recvs += ((end - start) / 2);
		}
	    }

	    RAPtor_MPI_Status recv_status;
	    while (size_recvs < local_assumed_num_cols)
	    {
		RAPtor_MPI_Probe(RAPtor_MPI_ANY_SOURCE, send_key, RAPtor_MPI_COMM_WORLD, &recv_status);
		RAPtor_MPI_Get_count(&recv_status, RAPtor_MPI_INT, &count);
		proc = recv_status.MPI_SOURCE;
		int recvbuf[count];
		RAPtor_MPI_Recv(recvbuf, count, RAPtor_MPI_INT, proc, send_key, RAPtor_MPI_COMM_WORLD, &recv_status);
		for (int i = 0; i < count; i+= 2)
		{
		    orig_col = recvbuf[i];
		    new_col = recvbuf[i+1];
		    local_col = orig_col - assumed_first_col;
		    assumed_col_to_new[local_col] = new_col;
		}
		size_recvs += (count / 2);
	    }

	    if (n_sent)
	    {
		RAPtor_MPI_Waitall(n_sent, send_requests.data(), RAPtor_MPI_STATUS_IGNORE);
	    }


	    // Reset proc_num_cols values to 0
	    for (int i = 0; i < num_sends; i++)
	    {
		proc = send_procs[i];
		proc_num_cols[proc] = 0;
	    }

	    // Clear send info from previous communication
	    send_procs.clear();
	    send_ptr.clear();
	    send_ctr.clear();

	    // Go through off_proc columns, and find which proc with which each is
	    // assumed to be associated
	    for (int i = 0; i < A->off_proc_num_cols; i++)
	    {
		orig_col = A->off_proc_column_map[i];
		assumed_proc = orig_col / assumed_num_cols;
		proc_num_cols[assumed_proc]++;
	    }

	    // Create send_procs, send_ptr
	    send_size = 0;
	    send_ptr.emplace_back(send_size);
	    for (int i = 0; i < num_procs; i++)
	    {
		if (proc_num_cols[i])
		{
		    send_size += proc_num_cols[i];
		    proc_num_cols[i] = send_procs.size();
		    send_procs.emplace_back(i);
		    send_ptr.emplace_back(send_size);
		}
	    }
	    num_sends = send_procs.size();
	    if (num_sends)
	    {
		send_ctr.resize(num_sends, 0);
		send_buffer.resize(send_size);
		recv_buffer.resize(send_size);
		send_requests.resize(num_sends);
		recv_requests.resize(num_sends);
	    }
	    
	    // Add columns to send buffer, ordered by assumed process
	    for (int i = 0; i < A->off_proc_num_cols; i++)
	    {
		orig_col = A->off_proc_column_map[i];
		assumed_proc = orig_col / assumed_num_cols;
		proc_idx = proc_num_cols[assumed_proc];
		idx = send_ptr[proc_idx] + send_ctr[proc_idx]++;
		send_buffer[idx] = orig_col;
	    }

	    // Send off_proc_columns to proc assumed to hold col and recv new global idx
	    // of column.  If assumed proc is rank, find new col and add to
	    // off_proc_col_to_new
	    n_sent = 0;
	    send_key = 7980;
	    recv_key = 8976;
	    for (int i = 0; i < num_sends; i++)
	    {
		proc = send_procs[i];
		start = send_ptr[i];
		end = send_ptr[i+1];

		if (proc != rank)
		{
		    RAPtor_MPI_Issend(&(send_buffer[start]), end - start, RAPtor_MPI_INT, proc, send_key, 
			    RAPtor_MPI_COMM_WORLD, &(send_requests[n_sent]));
		    RAPtor_MPI_Irecv(&(recv_buffer[start]), end - start, RAPtor_MPI_INT, proc, recv_key, 
			    RAPtor_MPI_COMM_WORLD, &(recv_requests[n_sent++]));
		}
		else
		{
		    for (int j = start; j < end; j++)
		    {
			orig_col = send_buffer[j];
			assumed_col = orig_col - assumed_first_col;
			new_col = assumed_col_to_new[assumed_col];
			local_col = global_to_local[orig_col];
			A->off_proc_column_map[local_col] = new_col;
		    }
		}
	    }

	    // Recv columns corresponding to my assumed cols, and return their new cols
	    if (n_sent)
	    {
		RAPtor_MPI_Testall(n_sent, send_requests.data(), &finished, RAPtor_MPI_STATUSES_IGNORE);
		while (!finished)
		{
		    RAPtor_MPI_Iprobe(RAPtor_MPI_ANY_SOURCE, send_key, RAPtor_MPI_COMM_WORLD, &msg_avail, &recv_status);
		    if (msg_avail)
		    {
			RAPtor_MPI_Get_count(&recv_status, RAPtor_MPI_INT, &count);
			proc = recv_status.MPI_SOURCE;
			int recvbuf[count];
			RAPtor_MPI_Recv(recvbuf, count, RAPtor_MPI_INT, proc, send_key, RAPtor_MPI_COMM_WORLD, 
				&recv_status);
			for (int i = 0; i < count; i++)
			{
			    orig_col = recvbuf[i];
			    assumed_col = orig_col - assumed_first_col;
			    new_col = assumed_col_to_new[assumed_col];
			    recvbuf[i] = new_col;
			}
			RAPtor_MPI_Send(recvbuf, count, RAPtor_MPI_INT, proc, recv_key, RAPtor_MPI_COMM_WORLD);
		    }
		    RAPtor_MPI_Testall(n_sent, send_requests.data(), &finished, RAPtor_MPI_STATUSES_IGNORE);
		}
	    }
	    RAPtor_MPI_Ibarrier(RAPtor_MPI_COMM_WORLD, &barrier_request);
	    RAPtor_MPI_Test(&barrier_request, &finished, RAPtor_MPI_STATUS_IGNORE);
	    while (!finished)
	    {
		RAPtor_MPI_Iprobe(RAPtor_MPI_ANY_SOURCE, send_key, RAPtor_MPI_COMM_WORLD, &msg_avail, &recv_status);
		if (msg_avail)
		{
		    RAPtor_MPI_Get_count(&recv_status, RAPtor_MPI_INT, &count);
		    proc = recv_status.MPI_SOURCE;
		    int recvbuf[count];
		    RAPtor_MPI_Recv(recvbuf, count, RAPtor_MPI_INT, proc, send_key, RAPtor_MPI_COMM_WORLD, 
			    &recv_status);
		    for (int i = 0; i < count; i++)
		    {
			orig_col = recvbuf[i];
			assumed_col = orig_col - assumed_first_col;
			new_col = assumed_col_to_new[assumed_col];
			recvbuf[i] = new_col;
		    }
		    RAPtor_MPI_Send(recvbuf, count, RAPtor_MPI_INT, proc, recv_key, RAPtor_MPI_COMM_WORLD);
		}
		RAPtor_MPI_Test(&barrier_request, &finished, RAPtor_MPI_STATUS_IGNORE);
	    }

	    // Wait for recvs to complete, and map original local cols to new global
	    // columns
	    if (n_sent)
	    {
		RAPtor_MPI_Waitall(n_sent, recv_requests.data(), RAPtor_MPI_STATUSES_IGNORE);
	    }
	    for (int i = 0; i < num_sends; i++)
	    {
		proc = send_procs[i];
		start = send_ptr[i];
		end = send_ptr[i+1];
		if (proc != rank)
		{
		    for (int j = start; j < end; j++)
		    {
			orig_col = send_buffer[j];
			new_col = recv_buffer[j];
			local_col = global_to_local[orig_col];
			A->off_proc_column_map[local_col] = new_col;
		    }
		}
	    }

	    // re-index columns of off_proc (ordered by new global columns)
	    if (A->off_proc_num_cols)
	    {
		// Find permutation of off_proc columns, sorted by global 
		// column indices in ascending order
		aligned_vector<int> p(A->off_proc_num_cols);
		std::iota(p.begin(), p.end(), 0);
		std::sort(p.begin(), p.end(), 
			[&](int i, int j)
			{
			    return A->off_proc_column_map[i] < A->off_proc_column_map[j];
			});
	    
		// Form off_proc_orig_to_new, mapping original off_proc local
		// column indices to new local column indices
		aligned_vector<int> off_proc_orig_to_new(A->off_proc_num_cols);
		for (int i = 0; i < A->off_proc_num_cols; i++)
		{
		    off_proc_orig_to_new[p[i]] = i;
		}

		// Re-index columns of off_proc
		for (aligned_vector<int>::iterator it = A->off_proc->idx2.begin();
			it != A->off_proc->idx2.end(); ++it)
		{
		    *it = off_proc_orig_to_new[*it];
		}

		// Sort off_proc_column_map based on permutation vector p
		aligned_vector<bool> done(A->off_proc_num_cols);
		for (int i = 0; i < A->off_proc_num_cols; i++)
		{
		    if (done[i]) continue;

		    done[i] = true;
		    int prev_j = i;
		    int j = p[i];
		    while (i != j)
		    {
			std::swap(A->off_proc_column_map[prev_j], A->off_proc_column_map[j]);
			done[j] = true;
			prev_j = j;
			j = p[j];
		    }
		}
	    }

	    for (int i = 0; i < A->local_num_rows; i++)
	    {
		A->on_proc_column_map[i] = A->partition->first_local_col + i;
	    }
	    A->local_row_map = A->get_on_proc_column_map();

	    if (form_comm)
        A->comm = new ParComm(A->partition, A->off_proc_column_map);

    // Sort rows, removing duplicate entries and moving diagonal 
    // value to first
    A->on_proc->sort();
    A->on_proc->move_diag();
    A->off_proc->sort();
}


CSRMatrix* send_matrix(CSRMatrix* A_on, CSRMatrix* A_off, int* partition, 
        int* local_row_map, int* on_proc_column_map, int* off_proc_column_map, 
	aligned_vector<int>& proc_row_sizes, aligned_vector<int>& new_local_rows)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int start, end;
    int proc, row, col, idx, pos;
    int start_on, end_on;
    int start_off, end_off;
    int on_size, off_size;
    int size, tag, n_rows;
    int count, ctr, prev_ctr;
    int num_sends, send_row_size;
    int local_row, local_nnz;
    int global_row, global_col;
    int mat_size, prev_mat_size;
    MPI_Status recv_status;

    aligned_vector<int> proc_to_idx(num_procs);
    aligned_vector<int> proc_row_idx;

    aligned_vector<int> send_procs;
    aligned_vector<int> send_ptr;
    aligned_vector<MPI_Request> send_requests;

    // Find how many rows go to each proc
    for (int i = 0; i < A_on->n_rows; i++)
    {
        proc = partition[i];
        if (proc_row_sizes[proc] == 0)
        {
            proc_to_idx[proc] = send_procs.size();
            send_procs.push_back(proc);
        }
        proc_row_sizes[proc]++;
    }

    // Create send_procs: procs to which I must send rows
    // and send_ptr: number of rows to send to each2
    num_sends = send_procs.size();
    send_ptr.resize(num_sends+1);
    send_ptr[0] = 0;
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        send_ptr[i+1] = send_ptr[i] + proc_row_sizes[proc];
        proc_row_sizes[proc] = 0;
    }
    send_row_size = send_ptr[num_sends];
    
    // Add to proc_idx (rows, sorted by proc, according to proc_ptr)
    if (send_row_size) proc_row_idx.resize(send_row_size);
    for (int i = 0; i < A_on->n_rows; i++)
    {
        proc = partition[i];
        idx = proc_to_idx[proc];
        pos = send_ptr[idx] + proc_row_sizes[proc]++;
        proc_row_idx[pos] = i;
    }

    // Now know the number of messages to be sent
    if (num_sends) send_requests.resize(num_sends);
   
    RAPtor_MPI_Allreduce(MPI_IN_PLACE, proc_row_sizes.data(), num_procs, MPI_INT,
            MPI_SUM, MPI_COMM_WORLD);
    int recv_num_rows = proc_row_sizes[rank];

    // Send Matrix Rows to Corresponding Processes
    tag = 23931;
    ctr = 0;
    prev_ctr = 0;
    local_nnz = A_on->nnz;
    if (A_off) local_nnz += A_off->nnz;
    MPI_Pack_size(2*A_on->n_rows + local_nnz, MPI_INT, MPI_COMM_WORLD, &mat_size);
    MPI_Pack_size(local_nnz * A_on->b_size, MPI_DOUBLE, MPI_COMM_WORLD, &size);
    mat_size += size;
    aligned_vector<char> send_buffer(mat_size);
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        for (int j = start; j < end; j++)
        {
            row = proc_row_idx[j];
            global_row = local_row_map[row];

            start_on = A_on->idx1[row];
            end_on = A_on->idx1[row+1];
            on_size = end_on - start_on;
            if (A_off)
            {
                start_off = A_off->idx1[row];
                end_off = A_off->idx1[row+1];
            }
            else
            {
                start_off = 0;
                end_off = 0;
            }
            off_size = end_off - start_off;
            size = on_size + off_size;

            // Send global row
            RAPtor_MPI_Pack(&global_row, 1, RAPtor_MPI_INT, send_buffer.data(),
                    mat_size, &ctr, MPI_COMM_WORLD);
            // Send size of row (num nonzeros in row)
            RAPtor_MPI_Pack(&size, 1, RAPtor_MPI_INT, send_buffer.data(), 
                    mat_size, &ctr, MPI_COMM_WORLD);

            // Add global column indices
            for (int j = start_on; j < end_on; j++)
            {
                col = A_on->idx2[j];
                global_col = on_proc_column_map[col];
                RAPtor_MPI_Pack(&(global_col), 1, RAPtor_MPI_INT, send_buffer.data(),
                        mat_size, &ctr, MPI_COMM_WORLD);
            }
            for (int j = start_off; j < end_off; j++)
            {
                col = A_off->idx2[j];
                global_col = off_proc_column_map[col];
                RAPtor_MPI_Pack(&(global_col), 1, RAPtor_MPI_INT, send_buffer.data(),
                        mat_size, &ctr, MPI_COMM_WORLD);
            }
            
            // Add values associated with each nonzero in row
            RAPtor_MPI_Pack(&(A_on->vals[start_on]), on_size, 
                    RAPtor_MPI_DOUBLE, send_buffer.data(), mat_size, &ctr, MPI_COMM_WORLD);
            if (off_size)
            {
                RAPtor_MPI_Pack(&(A_off->vals[start_off]), off_size, 
                        RAPtor_MPI_DOUBLE, send_buffer.data(), mat_size, &ctr, MPI_COMM_WORLD);
            }
        }

        // Send previously packed matrix
        RAPtor_MPI_Isend(&(send_buffer[prev_ctr]), ctr - prev_ctr, RAPtor_MPI_PACKED,
                proc, tag, MPI_COMM_WORLD, &(send_requests[i]));
        prev_ctr = ctr;
    }

    // Recv Matrix Rows (wait for recv_num_rows of them)
    CSRMatrix* recv_mat = new CSRMatrix(recv_num_rows, recv_num_rows);
    if (recv_num_rows) new_local_rows.resize(recv_num_rows);
    aligned_vector<char> recv_buffer;
    n_rows = 0;
    mat_size = 0;
    while (n_rows < recv_num_rows)
    {
        RAPtor_MPI_Probe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &recv_status);
        RAPtor_MPI_Get_count(&recv_status, RAPtor_MPI_PACKED, &count);
        proc = recv_status.MPI_SOURCE;
        if (count > recv_buffer.size())
            recv_buffer.resize(count);
        RAPtor_MPI_Recv(&(recv_buffer[0]), count, RAPtor_MPI_PACKED, proc, tag,
                MPI_COMM_WORLD, &recv_status);

        ctr = 0;
        while (ctr < count)
        {
            // Get row size, and allocate space in idx2/vals 
            RAPtor_MPI_Unpack(recv_buffer.data(), count, &ctr, &(new_local_rows[n_rows]),
                    1, RAPtor_MPI_INT, MPI_COMM_WORLD);
            RAPtor_MPI_Unpack(recv_buffer.data(), count, &ctr, &size, 1, 
                    RAPtor_MPI_INT, MPI_COMM_WORLD);
            prev_mat_size = mat_size;
            mat_size += size;
            recv_mat->idx1[n_rows+1] = mat_size;
            n_rows++;
            recv_mat->idx2.resize(mat_size);
            recv_mat->vals.resize(mat_size);

            // Unpack row's global column indices
            RAPtor_MPI_Unpack(recv_buffer.data(), count, &ctr, 
                    &(recv_mat->idx2[prev_mat_size]), size,
                    RAPtor_MPI_INT, MPI_COMM_WORLD);

            // Unpack row's values
            RAPtor_MPI_Unpack(recv_buffer.data(), count, &ctr,
                    &(recv_mat->vals[prev_mat_size]), size,
                    RAPtor_MPI_DOUBLE, MPI_COMM_WORLD);
        }
    }
    recv_mat->nnz = recv_mat->idx2.size();
            
    if (num_sends)
    {
        RAPtor_MPI_Waitall(num_sends, send_requests.data(), 
                RAPtor_MPI_STATUSES_IGNORE);
    }

    return recv_mat;

}

ParCSRMatrix* repartition_matrix(ParCSRMatrix* A, int* partition, 
        aligned_vector<int>& new_local_rows, bool make_contig)
{
    int rank, num_procs;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);

    ParCSRMatrix* A_part;
    int first_row, global_row, global_col;
    int start, end, ctr;
    int recv_num_rows;
    double val;
    RAPtor_MPI_Status recv_status;

    aligned_vector<int> proc_row_sizes(num_procs, 0);
    CSRMatrix* recv_mat = send_matrix((CSRMatrix*)A->on_proc, (CSRMatrix*)A->off_proc, partition, 
            A->local_row_map.data(), A->on_proc_column_map.data(),
            A->off_proc_column_map.data(), proc_row_sizes, new_local_rows);

    // Assuming local num cols == num_rows (square)
    first_row = 0;
    for (int i = 0; i < rank; i++)
    {
        first_row += proc_row_sizes[i];
    }
    recv_num_rows = proc_row_sizes[rank];

    A_part = new ParCSRMatrix(A->global_num_rows, A->global_num_rows, 
            recv_num_rows, recv_num_rows, first_row, first_row, 
            A->partition->topology);

    // Create row_ptr
    // Add values/indices to appropriate positions
    std::map<int, int> on_proc_to_local;
    for(int i = 0; i < recv_num_rows; i++)
    {
        global_row = new_local_rows[i];
        on_proc_to_local[global_row] = i;
        A_part->on_proc_column_map.push_back(global_row);
    }
    A_part->local_row_map = A_part->get_on_proc_column_map();
    A_part->on_proc_num_cols = A_part->on_proc_column_map.size();

    ctr = 0;
    A_part->on_proc->idx1[0] = 0;
    A_part->off_proc->idx1[0] = 0;
    for (int i = 0; i < recv_num_rows; i++)
    {
        start = recv_mat->idx1[i];
        end = recv_mat->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            global_col = recv_mat->idx2[j];
            val = recv_mat->vals[j];
            std::map<int, int>::iterator it = on_proc_to_local.find(global_col);
            if (it != on_proc_to_local.end())
            {
                A_part->on_proc->idx2.push_back(it->second);
                A_part->on_proc->vals.push_back(val);
            }
            else
            {
                A_part->off_proc->idx2.push_back(global_col);
                A_part->off_proc->vals.push_back(val);
            }
        }
        A_part->on_proc->idx1[i+1] = A_part->on_proc->idx2.size();
        A_part->off_proc->idx1[i+1] = A_part->off_proc->idx2.size();
    }
    A_part->on_proc->nnz = A_part->on_proc->idx2.size();
    A_part->off_proc->nnz = A_part->off_proc->idx2.size();
    A_part->local_nnz = A_part->on_proc->nnz + A_part->off_proc->nnz;

    aligned_vector<int> off_proc_cols;
    std::copy(A_part->off_proc->idx2.begin(), A_part->off_proc->idx2.end(),
            std::back_inserter(off_proc_cols));
    std::sort(off_proc_cols.begin(), off_proc_cols.end());
    int prev_col = -1;
    std::map<int, int> global_to_local;
    for (aligned_vector<int>::iterator it = off_proc_cols.begin(); 
            it != off_proc_cols.end(); ++it)
    {
        if (*it != prev_col)
        {
            global_to_local[*it] = A_part->off_proc_column_map.size();
            A_part->off_proc_column_map.emplace_back(*it);
            prev_col = *it;
        }
    }
    A_part->off_proc_num_cols = A_part->off_proc_column_map.size();
    delete recv_mat;

    for (aligned_vector<int>::iterator it = A_part->off_proc->idx2.begin();
            it != A_part->off_proc->idx2.end(); ++it)
    {
        *it = global_to_local[*it];
    }

    new_local_rows.resize(A_part->on_proc_num_cols);
    std::copy(A_part->on_proc_column_map.begin(), A_part->on_proc_column_map.end(),
            new_local_rows.begin());

    if (make_contig)
        make_contiguous(A_part);

    return A_part;
}




