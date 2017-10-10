#include "partition.hpp"

int* ptscotch_partition(ParCSRMatrix* A)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Variables for Graph Partitioning
    SCOTCH_Arch archdata;
    SCOTCH_Strat stratdata;
    SCOTCH_Dgraph dgraphdata;
    int* edge_starts;
    int* lcl_indices;
    int* gbl_indices;
    int* partition = NULL;
    int baseval = 0; // Always 0 for C style arrays
    int row_start, row_end;
    int idx, gbl_idx, ctr;
    int err;

    // Allocate Partition Variable (to be returned)
    if (A->local_num_rows) 
    {
        partition = new int[A->local_num_rows];
    }
    
    SCOTCH_archInit(&archdata);
    SCOTCH_archCmplt(&archdata, num_procs);

    // Build Strategy for PT Scotch Partitioner
    if (SCOTCH_stratInit(&stratdata))
    {
        printf("Cannot Initialize Strategy\n");
    }
    if (SCOTCH_stratDgraphMap(&stratdata, "r{seq=b, sep=x}"))
    {
        printf("Cannot create strategy map\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // lcl_verts + 2 so that size is at least 2
    edge_starts = new int[A->local_num_rows+2]; 

    // Know edge_starts size is at least 2
    lcl_indices = new int[A->local_nnz + 1];
    gbl_indices = new int[A->local_nnz + 1];

    // Find matrix edge indices for PT Scotch
    ctr = 0;
    for (int row = 0; row < A->local_num_rows; row++)
    {
        edge_starts[row] = ctr;
        row_start = A->on_proc->idx1[row];
        row_end = A->on_proc->idx1[row+1];
        for (int j = row_start; j < row_end; j++)
        {
            idx = A->on_proc->idx2[j];
            gbl_idx = A->on_proc_column_map[idx];
            lcl_indices[ctr] = idx;
            gbl_indices[ctr] = gbl_idx;
            ctr++;
        }

        if (A->offd_num_cols)
        {
            row_start = A->off_proc->idx1[row];
            row_end = A->off_proc->idx1[row+1];
            for (int j = row_start; j < row_end; j++)
            {
                idx = A->off_proc->idx2[j];
                gbl_idx = A->off_proc_column_map[idx];
                lcl_indices[ctr] = idx;
                gbl_indices[ctr] = gbl_idx;
                ctr++;
            }
        }
    }
    edge_starts[A->local_num_rows] = A->local_nnz;

    // Build PT Scotch Graph
    if (SCOTCH_dgraphInit(&dgraphdata, MPI_COMM_WORLD))
    {
        printf("Cannot Properly Init DGRAPH\n");
    }

    if (SCOTCH_dgraphBuild(&dgraphdata,
           baseval,            // Offset 0 in C
           A->local_num_rows,          // Number of local vertices
           A->local_num_rows,      // Maximum number of local vertices to be created
           edge_starts,        // Local adjacency index array (row_starts)
           &(edge_starts[1]),          // (Optional) Local adjacency end index array (row_ends)
           NULL,               // (Optional) Local vertex load array
           NULL,
           A->local_nnz,            // Total number of edges (1x for each direction)
           A->local_nnz,            // Minimum size of edge array required to encompass all adjacency values
           gbl_indices,        // Local adjacency array which stores global indices
           lcl_indices,        // (Optional)
           //NULL,
           NULL))              // (Optional) arc load array
    {
        printf("Cannot Build DGraph\n");
    }

    int global_verts, local_verts, global_edges, local_edges;
    SCOTCH_dgraphSize(&dgraphdata,
                      &global_verts,
                      &local_verts,
                      &global_edges,
                      &local_edges);


    // Check that Graph Object was build correctly
    if (SCOTCH_dgraphCheck (&dgraphdata)) printf("Error!\n");
    if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS)
    {
        printf("Error in MPI Barrier\n");
    }

    // Partition Graph with PT Scotch... Variable partition will contain
    // which process each local row should be stored on
    SCOTCH_dgraphMap(&dgraphdata, &archdata, &stratdata, partition);
//    SCOTCH_dgraphPart(&dgraphdata, num_procs, &stratdata, partition);
    if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS)
    {
        printf("Error in MPI Barrier\n");
    }

    // Free PT Scotch Variable
    SCOTCH_dgraphExit(&dgraphdata);
    SCOTCH_stratExit(&stratdata);
    SCOTCH_archExit(&archdata);

    delete[] edge_starts;
    delete[] lcl_indices;
    delete[] gbl_indices;

    return partition;
}

ParCSRMatrix* repartition_matrix(ParCSRMatrix* A, int* partition)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    struct PairData
    {
        double val;
        int index;
    };

    ParCSRMatrix* A_part;
    std::vector<int> send_row_buffer;
    std::vector<int> recv_row_buffer;
    std::vector<PairData> send_buffer;
    std::vector<PairData> recv_buffer;
    
    int proc;
    int send_row_size;
    int num_sends;
    int row_key = 370284;
    int msg_avail, finished;
    int count;
    MPI_Status recv_status;
    MPI_Request barrier_request;
    std::vector<int> send_procs;
    std::vector<int> send_ptr;
    std::vector<int> proc_sizes(num_procs, 0);
    std::vector<int> proc_to_idx(num_procs);
    std::vector<int> send_requests;
    std::vector<int> recv_rows;
    std::vector<int> recv_row_sizes;
    std::vector<int> recv_procs;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        proc = partition[i];
        proc_sizes[proc]++;
    }
    send_row_size = 0;
    send_ptr.push_back(0);
    for (int i = 0; i < num_procs; i++)
    {
        if (proc_sizes[i])
        {
            send_row_size += proc_sizes[i];
            proc_to_idx[i] = send_procs.size();
            send_procs.push_back(i);
            send_ptr.push_back(send_row_size);
            proc_sizes[i] = 0;
        }
    }
    num_sends = send_procs.size();
    if (num_sends)
    {
        send_requests.resize(num_sends);
    }
    send_row_buffer.resize(2*send_row_size);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        proc = partition[i];
        proc_idx = proc_to_idx[proc];
        idx = send_ptr[proc_idx] + proc_sizes[proc]++;
        send_row_buffer[2*idx] = A->local_row_map[i];
        
        row_size = (A->on_proc->idx1[i+1] - A->on_proc->idx1[i]) + 
            (A->off_proc->idx1[i+1] - A->off_proc->idx1[i]);
        send_row_buffer[2*idx+1] = row_size;
    }

    // Send to proc p the rows and row sizes that will be sent to it
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        MPI_Issend(&(send_row_buffer[2*start]), 2*(end - start), MPI_INT, proc,
                row_key, MPI_COMM_WORLD, &send_requests[i]);
    }

    // Dynamically receive rows and corresponding sizes  
    if (num_sends)
    {
        MPI_Testall(num_sends, send_requests.data(), &finished, MPI_STATUSES_IGNORE);
        while (!finished)
        {
            MPI_Iprobe(MPI_ANY_SOURCE, row_key, MPI_COMM_WORLD, &msg_avail, &recv_status);
            if (msg_avail)
            {
                proc = recv_status.MPI_SOURCE;
                MPI_Get_count(&recv_status, MPI_INT, &count);
                if (count > recv_row_buffer.size())
                {
                    recv_row_buffer.resize(count);
                }
                MPI_Recv(recv_row_buffer.data(), count, MPI_INT, proc, row_key,
                        MPI_COMM_WORLD, &recv_status);
                for (int i = 0; i < count; i += 2)
                {
                    recv_rows.push_back(recv_row_buffer[i]);
                    recv_row_sizes.push_back(recv_row_buffer[i+1]);
                }
                recv_procs.push_back(proc);
            }
            MPI_Testall(num_sends, send_requests.data(), &finished, MPI_STATUSES_IGNORE);
        }
    }
    MPI_Ibarrier(MPI_COMM_WORLD, &barrier_request);
    MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
    while (!finished)
    {
        MPI_Iprobe(MPI_ANY_SOURCE, row_key, MPI_COMM_WORLD, &msg_avail, &recv_status);
        if (msg_avail)
        {
            proc = recv_status.MPI_SOURCE;
            MPI_Get_count(&recv_status, MPI_INT, &count);
            if (count > recv_row_buffer.size())
            {
                recv_row_buffer.resize(count);
            }
            MPI_Recv(recv_row_buffer.data(), count, MPI_INT, proc, row_key,
                    MPI_COMM_WORLD, &recv_status);
            for (int i = 0; i < count; i += 2)
            {
                recv_rows.push_back(recv_row_buffer[i]);
                recv_row_sizes.push_back(recv_row_buffer[i+1]);
            }
            recv_procs.push_back(proc);
        }
        MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
    }

    // Find permutation of rows (smallest to largest)
    std::vector<int> p;
    std::vector<bool> done;
    int num_rows = recv_rows.size();
    int num_recvs = recv_procs.size();
    if (num_rows)
    {
        p.resize(num_rows);
        done.resize(num_rows, false)
        std::iota(p.begin(), p.end(), 0);
        std::sort(p.begin(), p.end(), 
                [&](const int i, const int j)
                {
                    return recv_rows[i] < recv_rows[j];
                });
    }

    // Sort rows (smallest to largest) and corresponding sizes
    for (int i = 0; i < num_rows; i++)
    {
        if (done[i]) continue;

        done[i] = true;
        prev_k = i;
        k = p[i];
        while (i != k)
        {
            std::swap(recv_rows[prev_k], recv_rows[k]);
            std::swap(recv_row_sizes[prev_k], recv_row_sizes[k]);
            done[k] = true;
            prev_k = k;
            k = p[k];
        }
    }

    // Assuming local num cols == num_rows (square)
    std::vector<int> proc_sizes(num_procs);
    MPI_Allgather(&(num_rows), 1, MPI_INT, proc_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    first_row = 0;
    for (int i = 0; i < rank; i++)
    {
        first_row += proc_sizes[i];
    }

    A_part = new ParCSRMatrix(A->global_num_rows, A->global_num_cols, num_rows, num_rows, first_row, first_row);

    // Form indptr for on and off proc matrices
    A->on_proc->idx1[0] = 0;
    A->off_proc->idx1[0] = 0;
    for (int i = 0; i < num_rows; i++)
    {
        A->on_proc->idx1[i+1] = A->on_proc->idx1[i] + recv_row_sizes[i];
        A->off_proc->idx1[i+1] = A->off_proc->idx1[i] + recv_row_sizes[i];
    }

    // Form vector of pair data (col indices and values) for sending to each
    // proc
    










    int global_row, global_col;
    int row_start, row_end;
    int proc, ctr;
    double value;

    int num_sends, count;
    int send_start, send_end;
    int tag = 3456;
    std::vector<int> send_procs;
    std::vector<int> send_row_starts;
    std::vector<int> send_data_starts;
    std::vector<int>* send_rows;
    MPI_Request* send_requests;
    MPI_Status recv_status;
    coo_data* send_data;

    int* send_proc_bool;
    int* proc_num_recvs;

    // MPI_Datatype variables (for sending coo_data structs}
    send_rows = new std::vector<int>[num_procs];
    send_data = new coo_data[A->local_nnz];

    // Go through every value in A and add it to new_mat if 
    // (proc = partition[value]) == rank.  Otherwise, store 
    // row to be sent to proc.
    if (A->offd_num_cols) A->offd->convert(CSR);
    for (int i = 0; i < A->local_rows; i++)
    {
        proc = partition[i];
        if (proc != rank)
        {
            send_rows[proc].push_back(i);
        }
        else
        {
            global_row = A->first_row + i;
            row_start = A->diag->indptr[i];
            row_end = A->diag->indptr[i+1];
            for (int j = row_start; j < row_end; j++)
            {
                global_col = A->first_col_diag + A->diag->indices[j];
                new_mat.push_back({global_row, global_col, A->diag->data[j]});
            }

            if (A->offd_num_cols)
            {
                row_start = A->offd->indptr[i];
                row_end = A->offd->indptr[i+1];
                for (int j = row_start; j < row_end; j++)
                {
                    global_col = A->local_to_global[A->offd->indices[j]];
                    new_mat.push_back({global_row, global_col, A->offd->data[j]});
                }
            }
        }
    }

    // Create send_row_starts and send procs, to indicate
    // how many rows should be sent to each proc i talk to
    ctr = 0;
    send_row_starts.push_back(ctr);
    for (int i = 0; i < num_procs; i++)
    {
        if (send_rows[i].size())
        {
            send_procs.push_back(i);
            ctr += send_rows[i].size();
            send_row_starts.push_back(ctr);
        }
    }
    num_sends = send_procs.size();

    // Create send_data_starts and send_data to store which
    // coo_data must be sent to each process in send_procs
    ctr = 0;
    send_data_starts.push_back(ctr);
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        std::vector<int>& proc_rows = send_rows[proc];
        for (std::vector<int>::iterator it = proc_rows.begin();
                it != proc_rows.end(); ++it)
        {
            global_row = *it + A->first_row;
            row_start = A->diag->indptr[*it];
            row_end = A->diag->indptr[*it+1];
            for (int j = row_start; j < row_end; j++)
            {
                global_col = A->first_col_diag + A->diag->indices[j];
                value = A->diag->data[j];
                send_data[ctr++] = {global_row, global_col, value};
            }

            if (A->offd_num_cols)
            {
                row_start = A->offd->indptr[*it];
                row_end = A->offd->indptr[*it+1];
                for (int j = row_start; j < row_end; j++)
                {
                    global_col = A->local_to_global[A->offd->indices[j]];
                    value = A->offd->data[j];
                    send_data[ctr++] = {global_row, global_col, value};
                }
            }
        }
        send_data_starts.push_back(ctr);
    }
    if (A->offd_num_cols) A->offd->convert(CSC);

    // Create MPI_Datatype for sending coo_data
    MPI_Type_extent(MPI_INT, &intex);
    displacements[0] = static_cast<MPI_Aint>(0);
    displacements[1] = 2*intex;
    MPI_Type_struct(2, blocks, displacements, types, &coo_type);
    MPI_Type_commit(&coo_type);

    // Determine how many messages I recv (with matrix rows)
    send_proc_bool = new int[num_procs]();
    proc_num_recvs = new int[num_procs];
    int num_recvs = 0;
    for (int i = 0; i < num_sends; i++)
    {
        send_proc_bool[send_procs[i]] = 1;
    }
    MPI_Allreduce(send_proc_bool, proc_num_recvs, num_procs, 
            MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    num_recvs = proc_num_recvs[rank];

    send_requests = new MPI_Request[num_sends];
    for (int i = 0; i < num_sends; i++)
    {
        send_start = send_data_starts[i];
        send_end = send_data_starts[i+1];
        MPI_Isend(&(send_data[send_start]), send_end - send_start, coo_type, 
                send_procs[i], tag, MPI_COMM_WORLD, &(send_requests[i]));
    }

    for (int i = 0; i < num_recvs; i++)
    {
        MPI_Probe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &recv_status);
        MPI_Get_count(&recv_status, coo_type, &count);
        coo_data recv_buffer[count];
        MPI_Recv(&recv_buffer, count, coo_type, recv_status.MPI_SOURCE, tag,
                MPI_COMM_WORLD, &recv_status);
        for (int j = 0; j < count; j++)
        {
            coo_data tmp = recv_buffer[j];
            new_mat.push_back({tmp.row, tmp.col, tmp.value});
        }
    }
    MPI_Waitall(num_sends, send_requests, MPI_STATUS_IGNORE);

    MPI_Type_free(&coo_type);
    delete[] send_requests;
    delete[] send_rows;
    delete[] send_proc_bool;
    delete[] proc_num_recvs;
    delete[] send_data;
}

ParCSRMatrix* repartition_matrix(ParCSRMatrix* A)
{
    return repartition_matrix(A, ptscotch_partition(A));
}


