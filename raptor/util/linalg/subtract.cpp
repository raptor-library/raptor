// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/types.hpp"
#include "core/par_matrix.hpp"

using namespace raptor;

ParCSRMatrix* ParCSRMatrix::subtract(ParCSRMatrix* B)
{
    ParCSRMatrix* C = new ParCSRMatrix(global_num_rows, global_num_cols,
            local_num_rows, local_num_cols, first_local_row, first_local_col);

    delete C->on_proc;
    delete C->off_proc;

    int start, end;
    int idx, idx_B;

    // This one will work fine because columns are not condensed
    C->on_proc = on_proc->subtract(B->on_proc);

    // For C->off_proc... need to edit CSRMatrix::subtract to take into 
    // account condensed cols
    C->off_proc = off_proc->subtract(B->off_proc);

    C->local_nnz = C->on_proc->nnz + C->off_proc->nnz;
    C->off_proc_column_map = C->off_proc->get_col_list();
    C->off_proc_num_cols = C->off_proc_column_map.size();

    // Communication package is just a sum of the communication 
    // packages of original ParCSRMat and B...
    C->comm = new ParComm();
    std::set<int> proc_send_set;
    for (std::vector<int>::iterator it = comm->send_data->procs.begin();
            it != comm->send_data->procs.end(); ++it)
    {
        proc_send_set.insert(*it);
    }
    for (std::vector<int>::iterator it = B->comm->send_data->procs.begin();
            it != B->comm->send_data->procs.end(); ++it)
    {
        proc_send_set.insert(*it);
    }
    idx = 0;
    idx_B = 0;
    C->comm->send_data->indptr.push_back(0);
    for (std::set<int>::iterator it = proc_send_set.begin(); 
            it != proc_send_set.end(); ++it)
    {
        std::set<int> send_idx_set;
        if (idx < comm->send_data->num_msgs)
        {
            if (comm->send_data->procs[idx] == *it)
            {
                start = comm->send_data->indptr[idx];
                end = comm->send_data->indptr[idx+1];
                for (int j = start; j < end; j++)
                {
                    send_idx_set.insert(comm->send_data->indices[j]);
                }
                idx++;
            }
        }
        if (idx_B < B->comm->send_data->num_msgs)
        {
            if (B->comm->send_data->procs[idx_B] == *it)
            {
                start = B->comm->send_data->indptr[idx_B];
                end = B->comm->send_data->indptr[idx_B+1];
                for (int j = start; j < end; j++)
                {
                    send_idx_set.insert(B->comm->send_data->indices[j]);
                }
            }
        }
        
        C->comm->send_data->procs.push_back(*it);
        C->comm->send_data->indptr.push_back(send_idx_set.size());
        for (std::set<int>::iterator it = send_idx_set.begin();
                it != send_idx_set.end(); ++it)
        {
            C->comm->send_data->indices.push_back(*it);
        }
    }
    C->comm->send_data->num_msgs = C->comm->send_data->procs.size();
    C->comm->send_data->size_msgs = C->comm->send_data->indices.size();
    C->comm->send_data->finalize();


    std::set<int> proc_recv_set;
    for (std::vector<int>::iterator it = comm->recv_data->procs.begin();
            it != comm->recv_data->procs.end(); ++it)
    {
        proc_recv_set.insert(*it);
    }
    for (std::vector<int>::iterator it = B->comm->recv_data->procs.begin();
            it != B->comm->recv_data->procs.end(); ++it)
    {
        proc_recv_set.insert(*it);
    }
    idx = 0;
    idx_B = 0;
    C->comm->recv_data->indptr.push_back(0);
    for (std::set<int>::iterator it = proc_recv_set.begin(); 
            it != proc_recv_set.end(); ++it)
    {
        std::set<int> recv_idx_set;
        if (idx < comm->recv_data->num_msgs)
        {
            if (comm->recv_data->procs[idx] == *it)
            {
                start = comm->recv_data->indptr[idx];
                end = comm->recv_data->indptr[idx+1];
                for (int j = start; j < end; j++)
                {
                    recv_idx_set.insert(comm->recv_data->indices[j]);
                }
                idx++;
            }
        }
        if (idx_B < B->comm->recv_data->num_msgs)
        {
            if (B->comm->recv_data->procs[idx_B] == *it)
            {
                start = B->comm->recv_data->indptr[idx_B];
                end = B->comm->recv_data->indptr[idx_B+1];
                for (int j = start; j < end; j++)
                {
                    recv_idx_set.insert(B->comm->recv_data->indices[j]);
                }
            }
        }
        
        C->comm->recv_data->procs.push_back(*it);
        C->comm->recv_data->indptr.push_back(recv_idx_set.size());
        for (std::set<int>::iterator it = recv_idx_set.begin();
                it != recv_idx_set.end(); ++it)
        {
            C->comm->recv_data->indices.push_back(*it);
        }
    }
    C->comm->recv_data->num_msgs = C->comm->recv_data->procs.size();
    C->comm->recv_data->size_msgs = C->comm->recv_data->indices.size();
    C->comm->recv_data->finalize();

    return C;
}

ParCSRMatrix* ParCSRMatrix::subtract(ParCSCMatrix* B)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        printf("Converting matrix to ParCSR before subtracting\n");
    }

    ParCSRMatrix* B_csr = new ParCSRMatrix(B);
    ParCSRMatrix* C = subtract(B);

    delete B_csr;
    return C;
}

ParCSRMatrix* ParCSRMatrix::subtract(ParCOOMatrix* B)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        printf("Converting matrix to ParCSR before subtracting\n");
    }

    ParCSRMatrix* B_csr = new ParCSRMatrix(B);
    ParCSRMatrix* C = subtract(B);

    delete B_csr;
    return C;
}

ParCSRMatrix* ParMatrix::subtract(ParCSRMatrix* B)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        printf("Subtraction Not Implemented for these matrix types\n");
    }
    return NULL;
}

ParCSRMatrix* ParMatrix::subtract(ParCSCMatrix* B)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        printf("Subtraction Not Implemented for these matrix types\n");
    }
    return NULL;
}

ParCSRMatrix* ParMatrix::subtract(ParCOOMatrix* B)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        printf("Subtraction Not Implemented for these matrix types\n");
    }
    return NULL;
}
